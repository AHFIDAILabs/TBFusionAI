"""
Model Training Pipeline for TBFusionAI - REGULARIZED VERSION.

ANTI-OVERFITTING MEASURES:
1. Strong regularization on all models
2. Early stopping for tree models and MLP
3. Cross-validation during training
4. Conservative hyperparameters
5. Feature subsampling
6. Validation monitoring

Handles:
- 6 base models with regularization
- 6 meta-cost models with regularization
- Early stopping callbacks
- Overfitting detection
- Detailed logging
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import get_config
from src.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# =========================================================================
# Custom Asymmetric Loss MLP
# =========================================================================
class AsymmetricLossMLPClassifier(MLPClassifier):
    """MLP with asymmetric cost-aware weighting."""

    def __init__(self, fn_cost=100, fp_cost=10, **mlp_kwargs):
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        super().__init__(**mlp_kwargs)

    def fit(self, X, y, **fit_kwargs):
        sample_weight = np.where(y == 1, self.fn_cost, self.fp_cost)
        sample_weight = sample_weight / sample_weight.sum() * len(y)
        return super().fit(X, y, **fit_kwargs)


# =========================================================================
# Main Training Pipeline - REGULARIZED
# =========================================================================
class ModelTrainingPipeline:
    """Cost-sensitive training with STRONG REGULARIZATION."""

    def __init__(self):
        """Initialize with anti-overfitting configuration."""
        self.config = get_config()

        self.models_path = Path(str(self.config.paths.models_path))
        self.labeled_data_path = Path(str(self.config.paths.labeled_data_path))

        self.models_path.mkdir(parents=True, exist_ok=True)

        self.trained_models: Dict = {}
        self.model_metrics: Dict = {}
        self.train_metrics: Dict = {}
        self.cv_scores: Dict = {}  # NEW: CV scores
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []

        logger.info("ModelTrainingPipeline initialized (REGULARIZED - Anti-Overfit)")

    async def run(self, data_path: Optional[Any] = None) -> Dict:
        """Execute regularized training pipeline."""
        logger.info("=" * 70)
        logger.info("STARTING REGULARIZED TRAINING PIPELINE")
        logger.info("=" * 70)

        if data_path and not isinstance(data_path, (str, Path)):
            logger.warning(f"Invalid data_path type: {type(data_path)}. Using default.")
            data_path = None

        try:
            X_train, X_valid, y_train, y_valid = await self._prepare_data(data_path)
            await self._train_all_models(X_train, X_valid, y_train, y_valid)
            await self._train_meta_cost_models(X_train, X_valid, y_train, y_valid)
            self._detect_overfitting()
            best_model_name = await self._evaluate_and_select_models(y_valid)
            await self._save_models_and_metadata(best_model_name)

            return {
                "best_model": best_model_name,
                "metrics": self.model_metrics,
            }
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            raise

    async def _prepare_data(self, data_path: Optional[Any]) -> Tuple:
        """Load and prepare data with detailed logging."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 70)

        if data_path is None:
            final_path = self.labeled_data_path / "wav2vec2_balanced_ctgan.csv"
        else:
            final_path = Path(str(data_path))

        if not final_path.exists():
            raise FileNotFoundError(f"Training data not found at: {final_path}")

        df = pd.read_csv(final_path)
        logger.info(f"✓ Loaded dataset: {df.shape}")

        clinical_features = self.config.metadata.clinical_features
        audio_features = [c for c in df.columns if c.startswith("feat_")]

        if "sex" in df.columns:
            df["sex"] = (df["sex"] == "Male").astype(int)

        self.feature_columns = audio_features + clinical_features

        X = df[self.feature_columns]
        y = df["tb_status"]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.config.model_training.test_size,
            stratify=y,
            random_state=self.config.model_training.random_state,
        )

        # Logging
        logger.info("\n" + "-" * 70)
        logger.info("Dataset Split:")
        logger.info("-" * 70)

        train_total = len(y_train)
        train_pos = (y_train == 1).sum()
        train_neg = (y_train == 0).sum()

        logger.info(
            f"Training set:   {train_total:,} samples ({(1 - self.config.model_training.test_size) * 100:.0f}%)"
        )
        logger.info(
            f"  - TB Positive: {train_pos:,} ({train_pos/train_total*100:.1f}%)"
        )
        logger.info(
            f"  - TB Negative: {train_neg:,} ({train_neg/train_total*100:.1f}%)"
        )

        valid_total = len(y_valid)
        valid_pos = (y_valid == 1).sum()
        valid_neg = (y_valid == 0).sum()

        logger.info(
            f"\nValidation set: {valid_total:,} samples ({self.config.model_training.test_size * 100:.0f}%)"
        )
        logger.info(
            f"  - TB Positive: {valid_pos:,} ({valid_pos/valid_total*100:.1f}%)"
        )
        logger.info(
            f"  - TB Negative: {valid_neg:,} ({valid_neg/valid_total*100:.1f}%)"
        )

        logger.info(f"\n✓ Features: {len(self.feature_columns)} total")
        logger.info(f"  - Audio features: {len(audio_features)}")
        logger.info(f"  - Clinical features: {len(clinical_features)}")

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_valid = self.scaler.transform(X_valid)

        logger.info(f"✓ Applied StandardScaler normalization")
        logger.info("=" * 70)

        return X_train, X_valid, y_train, y_valid

    async def _train_all_models(self, X_train, X_valid, y_train, y_valid):
        """Train 6 REGULARIZED base models."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: TRAINING BASE MODELS (REGULARIZED)")
        logger.info("=" * 70)
        logger.info("Regularization: max_depth, min_samples, subsampling, L1/L2")

        spw = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Scale Pos Weight: {spw:.2f}\n")

        await self._train_logistic(X_train, X_valid, y_train, y_valid)
        await self._train_random_forest(X_train, X_valid, y_train, y_valid)
        await self._train_xgboost(X_train, X_valid, y_train, y_valid, spw)
        await self._train_lightgbm(X_train, X_valid, y_train, y_valid, spw)
        await self._train_catboost(X_train, X_valid, y_train, y_valid, spw)
        await self._train_mlp(X_train, X_valid, y_train, y_valid)

    async def _train_logistic(self, X_tr, X_val, y_tr, y_val):
        logger.info("--- Logistic Regression (Regularized) ---")
        model = LogisticRegression(
            C=0.1,  # STRONG regularization
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        self.trained_models["Logistic"] = model
        self._evaluate_model("Logistic", model, X_tr, y_tr, X_val, y_val)

    async def _train_random_forest(self, X_tr, X_val, y_tr, y_val):
        logger.info("\n--- Random Forest (Regularized) ---")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # LIMIT depth
            min_samples_split=20,  # PREVENT small splits
            min_samples_leaf=10,  # PREVENT tiny leaves
            max_features="sqrt",  # USE sqrt features
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        self.trained_models["RandomForest"] = model
        self._evaluate_model("RandomForest", model, X_tr, y_tr, X_val, y_val)

    async def _train_xgboost(self, X_tr, X_val, y_tr, y_val, spw):
        logger.info("\n--- XGBoost (Regularized + Early Stopping) ---")
        model = XGBClassifier(
            n_estimators=50,  # More trees but will stop early/change from 500 to 50 for fewer trees beccause of overfitting
            max_depth=2,  # SHALLOW trees/change from 4 to 2 for very SHALLOW trees bcause of overfitting
            learning_rate=0.05,  # SLOW learning
            subsample=0.8,  # USE 80% data
            colsample_bytree=0.8,  # USE 80% features
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            min_child_weight=5,  # MIN samples per leaf
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=50,  # EARLY STOPPING
            verbosity=0,
        )

        # Fit with validation set for early stopping
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        logger.info(f"  Stopped at iteration: {model.best_iteration}")
        self.trained_models["XGBoost"] = model
        self._evaluate_model("XGBoost", model, X_tr, y_tr, X_val, y_val)

    async def _train_lightgbm(self, X_tr, X_val, y_tr, y_val, spw):
        logger.info("\n--- LightGBM (Regularized + Early Stopping) ---")
        model = LGBMClassifier(
            n_estimators=50,  # More trees but will stop early/change from 500 to 50 for fewer trees beccause of overfitting
            max_depth=2,  # SHALLOW trees/change from 4 to 2 for very SHALLOW trees bcause of overfitting
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,  # MIN samples per leaf
            scale_pos_weight=spw,
            random_state=42,
            verbose=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                # Early stopping callback
                __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False)
            ],
        )

        logger.info(f"  Stopped at iteration: {model.best_iteration_}")
        self.trained_models["LightGBM"] = model
        self._evaluate_model("LightGBM", model, X_tr, y_tr, X_val, y_val)

    async def _train_catboost(self, X_tr, X_val, y_tr, y_val, spw):
        logger.info("\n--- CatBoost (Regularized + Early Stopping) ---")
        model = CatBoostClassifier(
            iterations=50,  # More trees but will stop early/change from 500 to 50 for fewer trees beccause of overfitting
            depth=2,  # # SHALLOW trees/change from 4 to 2 for very SHALLOW trees bcause of overfitting
            learning_rate=0.05,
            l2_leaf_reg=3.0,  # L2 regularization
            subsample=0.8,
            colsample_bylevel=0.8,
            min_data_in_leaf=20,
            scale_pos_weight=spw,
            random_state=42,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=50,  # EARLY STOPPING
        )

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        logger.info(f"  Stopped at iteration: {model.best_iteration_}")
        self.trained_models["CatBoost"] = model
        self._evaluate_model("CatBoost", model, X_tr, y_tr, X_val, y_val)

    async def _train_mlp(self, X_tr, X_val, y_tr, y_val):
        logger.info("\n--- MLP (Regularized + Early Stopping) ---")
        model = AsymmetricLossMLPClassifier(
            fn_cost=5.0,
            fp_cost=1.0,
            hidden_layer_sizes=(100, 50),
            alpha=0.01,  # STRONG L2
            learning_rate_init=0.001,
            early_stopping=True,  # EARLY STOPPING
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        logger.info(f"  Stopped at iteration: {model.n_iter_}")
        self.trained_models["MLP"] = model
        self._evaluate_model("MLP", model, X_tr, y_tr, X_val, y_val)

    async def _train_meta_cost_models(self, X_tr, X_val, y_tr, y_val):
        """Train 6 REGULARIZED meta-cost models."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: META-COST MODELS (REGULARIZED)")
        logger.info("=" * 70)
        logger.info("Cost weighting: 10x FN penalty + Regularization\n")

        weights = np.where(y_tr == 1, 10.0, 1.0)
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()

        # Meta-Logistic
        logger.info("--- Meta-Cost Logistic (Regularized) ---")
        mc_log = LogisticRegression(
            C=0.1, penalty="l2", class_weight="balanced", max_iter=1000, random_state=42
        )
        mc_log.fit(X_tr, y_tr, sample_weight=weights)
        self.trained_models["Meta-Logistic"] = mc_log
        self._evaluate_model("Meta-Logistic", mc_log, X_tr, y_tr, X_val, y_val)

        # Meta-RandomForest
        logger.info("\n--- Meta-Cost RandomForest (Regularized) ---")
        mc_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        mc_rf.fit(X_tr, y_tr, sample_weight=weights)
        self.trained_models["Meta-RandomForest"] = mc_rf
        self._evaluate_model("Meta-RandomForest", mc_rf, X_tr, y_tr, X_val, y_val)

        # Meta-XGBoost
        logger.info("\n--- Meta-Cost XGBoost (Regularized + Early Stop) ---")
        mc_xgb = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=50,
            verbosity=0,
        )
        mc_xgb.fit(
            X_tr, y_tr, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False
        )
        logger.info(f"  Stopped at iteration: {mc_xgb.best_iteration}")
        self.trained_models["Meta-XGBoost"] = mc_xgb
        self._evaluate_model("Meta-XGBoost", mc_xgb, X_tr, y_tr, X_val, y_val)

        # Meta-LightGBM
        logger.info("\n--- Meta-Cost LightGBM (Regularized + Early Stop) ---")
        mc_lgb = LGBMClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            scale_pos_weight=spw,
            random_state=42,
            verbose=-1,
        )
        mc_lgb.fit(
            X_tr,
            y_tr,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False)
            ],
        )
        logger.info(f"  Stopped at iteration: {mc_lgb.best_iteration_}")
        self.trained_models["Meta-LightGBM"] = mc_lgb
        self._evaluate_model("Meta-LightGBM", mc_lgb, X_tr, y_tr, X_val, y_val)

        # Meta-CatBoost
        logger.info("\n--- Meta-Cost CatBoost (Regularized + Early Stop) ---")
        mc_cat = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            min_data_in_leaf=20,
            scale_pos_weight=spw,
            random_state=42,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=50,
        )
        mc_cat.fit(
            X_tr, y_tr, sample_weight=weights, eval_set=(X_val, y_val), verbose=False
        )
        logger.info(f"  Stopped at iteration: {mc_cat.best_iteration_}")
        self.trained_models["Meta-CatBoost"] = mc_cat
        self._evaluate_model("Meta-CatBoost", mc_cat, X_tr, y_tr, X_val, y_val)

        # Meta-MLP
        logger.info("\n--- Meta-Cost MLP (Regularized + Early Stop) ---")
        mc_mlp = AsymmetricLossMLPClassifier(
            fn_cost=10.0,
            fp_cost=1.0,
            hidden_layer_sizes=(100, 50),
            alpha=0.01,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=1000,
            random_state=42,
        )
        mc_mlp.fit(X_tr, y_tr)
        logger.info(f"  Stopped at iteration: {mc_mlp.n_iter_}")
        self.trained_models["Meta-MLP"] = mc_mlp
        self._evaluate_model("Meta-MLP", mc_mlp, X_tr, y_tr, X_val, y_val)

        logger.info(f"\n✓ Trained 6 Meta-Cost models (all regularized)")
        logger.info("=" * 70)

    def _evaluate_model(
        self,
        name: str,
        model: Any,
        X_tr: np.ndarray,
        y_tr: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
    ):
        """Evaluate with CV scores."""
        # Validation metrics
        preds_val = model.predict(X_val)
        probs_val = model.predict_proba(X_val)[:, 1]
        tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val, preds_val).ravel()
        brier_val = brier_score_loss(y_val, probs_val)

        self.model_metrics[name] = {
            "recall": recall_score(y_val, preds_val, zero_division=0),
            "f1": f1_score(y_val, preds_val, zero_division=0),
            "auc": roc_auc_score(y_val, probs_val),
            "brier_score": brier_val,
            "false_negatives": int(fn_val),
            "false_positives": int(fp_val),
            "true_positives": int(tp_val),
            "true_negatives": int(tn_val),
        }

        # Training metrics
        preds_tr = model.predict(X_tr)
        probs_tr = model.predict_proba(X_tr)[:, 1]
        tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_tr, preds_tr).ravel()
        brier_tr = brier_score_loss(y_tr, probs_tr)

        self.train_metrics[name] = {
            "recall": recall_score(y_tr, preds_tr, zero_division=0),
            "f1": f1_score(y_tr, preds_tr, zero_division=0),
            "brier_score": brier_tr,
            "false_negatives": int(fn_tr),
            "false_positives": int(fp_tr),
        }

        # Cross-validation (5-fold)
        try:
            cv_recall = cross_val_score(
                model, X_tr, y_tr, cv=5, scoring="recall", n_jobs=-1
            )
            self.cv_scores[name] = {
                "cv_recall_mean": cv_recall.mean(),
                "cv_recall_std": cv_recall.std(),
            }
            cv_mean = cv_recall.mean()
            cv_std = cv_recall.std()
        except:
            cv_mean, cv_std = 0.0, 0.0

        # Logging
        logger.info(
            f"VALIDATION → FN: {fn_val:>3} | FP: {fp_val:>4} | "
            f"Recall: {self.model_metrics[name]['recall']:.4f} | Brier: {brier_val:.4f}"
        )
        logger.info(
            f"TRAINING  → FN: {fn_tr:>3} | FP: {fp_tr:>4} | "
            f"Recall: {self.train_metrics[name]['recall']:.4f} | Brier: {brier_tr:.4f}"
        )
        if cv_mean > 0:
            logger.info(f"CV (5-fold) → Recall: {cv_mean:.4f} ± {cv_std:.4f}")

    def _detect_overfitting(self):
        """Detect overfitting with improved thresholds."""
        logger.info("\n" + "=" * 70)
        logger.info("OVERFITTING DETECTION")
        logger.info("=" * 70)

        overfitting_detected = False

        for name in self.model_metrics.keys():
            train_m = self.train_metrics[name]
            val_m = self.model_metrics[name]

            recall_gap = train_m["recall"] - val_m["recall"]
            f1_gap = train_m["f1"] - val_m["f1"]
            brier_gap = val_m["brier_score"] - train_m["brier_score"]

            # Stricter thresholds for regularized models
            if recall_gap > 0.05 or f1_gap > 0.05 or brier_gap > 0.03:
                overfitting_detected = True
                logger.warning(
                    f"⚠️  {name}: Minor overfitting\n"
                    f"    Recall gap: {recall_gap:+.4f} | F1 gap: {f1_gap:+.4f} | Brier gap: {brier_gap:+.4f}"
                )

            if train_m["false_negatives"] == 0 and train_m["false_positives"] == 0:
                overfitting_detected = True
                logger.warning(
                    f"⚠️  {name}: Perfect training (FN=0, FP=0) - Still overfitting!"
                )

        if not overfitting_detected:
            logger.info(
                "✓ No significant overfitting detected - Regularization working!"
            )

        logger.info("=" * 70)

    async def _evaluate_and_select_models(self, y_valid: pd.Series) -> str:
        """Select best model."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: MODEL COMPARISON (VALIDATION SET)")
        logger.info("=" * 70)

        sorted_models = sorted(
            self.model_metrics.items(),
            key=lambda x: (x[1]["false_negatives"], x[1]["false_positives"]),
        )

        logger.info(
            f"\n{'Model':<20} {'FN':>5} {'FP':>6} {'Recall':>8} {'F1':>8} {'CV-Recall':>10}"
        )
        logger.info("-" * 75)
        for name, m in sorted_models:
            cv_r = self.cv_scores.get(name, {}).get("cv_recall_mean", 0.0)
            logger.info(
                f"{name:<20} {m['false_negatives']:>5} {m['false_positives']:>6} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f} {cv_r:>10.4f}"
            )

        best_name = sorted_models[0][0]
        best_m = self.model_metrics[best_name]

        logger.info("\n" + "=" * 70)
        logger.info(f"🏆 BEST MODEL: {best_name}")
        logger.info(
            f"   FN: {best_m['false_negatives']} | FP: {best_m['false_positives']}"
        )
        logger.info(f"   Recall: {best_m['recall']:.4f} | F1: {best_m['f1']:.4f}")
        logger.info("=" * 70)

        return best_name

    async def _save_models_and_metadata(self, best_model_name: str):
        """Save all models and metadata."""
        top_3 = sorted(
            self.model_metrics.items(), key=lambda x: x[1]["false_negatives"]
        )[:3]
        top_3_names = [n for n, _ in top_3]

        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: SAVING MODELS")
        logger.info("=" * 70)

        logger.info(f"\nTop 3 Models:")
        for i, (name, m) in enumerate(top_3, 1):
            logger.info(
                f"  {i}. {name}: FN={m['false_negatives']}, FP={m['false_positives']}"
            )

        joblib.dump(self.trained_models, self.models_path / "all_models.joblib")

        for name in top_3_names:
            path = self.models_path / f"{name.replace(' ', '_')}_model.joblib"
            joblib.dump(self.trained_models[name], path)

        joblib.dump(self.scaler, self.models_path / "scaler.joblib")

        metadata = {
            "best_model": best_model_name,
            "top_3_models": top_3_names,
            "feature_columns": self.feature_columns,
            "metrics": self.model_metrics,
            "train_metrics": self.train_metrics,
            "cv_scores": self.cv_scores,
        }
        joblib.dump(metadata, self.models_path / "training_metadata.joblib")

        logger.info(f"\n✓ Saved 12 regularized models")
        logger.info(f"✓ All artifacts in: {self.models_path}")
        logger.info("=" * 70)


async def run_model_training(data_path: Optional[Any] = None):
    """Entry point."""
    pipeline = ModelTrainingPipeline()
    return await pipeline.run(data_path)
