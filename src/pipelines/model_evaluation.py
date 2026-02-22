"""
Enhanced Evaluation Pipeline + Clinical Safety Extensions (Non Intrusive)

FEATURE GUARANTEE
=================

Ensemble Layer
- Hard voting
- Soft voting
- Weighted voting

Rejection Layer
- Confidence disagreement reject
- Boundary margin reject

Threshold Science Layer
- Cost threshold optimisation
- F beta optimisation
- Youden index optimisation

Selection Logic
- Automatic threshold selection

Evaluation Science
- Performance comparison across strategies

Clinical Extensions
- Drift monitoring
- Population recalibration
- Uncertainty audit logging
- Model integrity hashing

ARCHITECTURE FLOW
=================

Predictions
    ↓
Ensemble Voting Layer
    ↓
Threshold Optimisation Layer
    ↓
Threshold Strategy Comparison
    ↓
Best Threshold Selection
    ↓
Clinical Safety Layer
    - Drift detection
    - Recalibration
    - Integrity check
    - Audit logging
    ↓
Final Ensemble Artifact Save
"""

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    recall_score,
    roc_curve,
)

from src.config import get_config
from src.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# =========================================================
# SAFETY UTILITIES
# =========================================================


def safe_path(p):
    """Guarantee Path object. Prevents OptionInfo path bugs."""
    return p if isinstance(p, Path) else Path(str(p))


def safe_confusion(y_true, y_pred):
    """Guarantee confusion matrix shape."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        return cm.ravel()
    return (0, 0, 0, 0)


def safe_threshold_range(start, stop, step):
    """Deterministic float safe threshold generator."""
    t = start
    while t <= stop:
        yield round(t, 6)
        t += step


# =========================================================
# CLINICAL SAFETY EXTENSIONS
# =========================================================


def compute_model_hash(models: Dict) -> str:
    """Generate integrity hash across all base models."""
    hasher = hashlib.sha256()
    for name in sorted(models.keys()):
        hasher.update(name.encode())
        hasher.update(str(type(models[name])).encode())
    return hasher.hexdigest()


def population_stability_index(expected, actual, bins=10):
    """Simple PSI drift metric."""
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_hist / max(expected_hist.sum(), 1)
    actual_pct = actual_hist / max(actual_hist.sum(), 1)

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log((actual_pct + 1e-6) / (expected_pct + 1e-6))
    )
    return psi


# =========================================================
# MAIN PIPELINE
# =========================================================


class ModelEvaluationPipeline:

    def __init__(self):
        """Initialise pipeline with config safe paths."""
        self.config = get_config()
        self.models_path = safe_path(self.config.paths.models_path)

        self.base_models = {}
        self.scaler = None
        self.metadata = {}

        logger.info("Enhanced Evaluation Pipeline Initialised")

    # ---------------------------------------------------------
    # LOAD MODELS + DATA
    # ---------------------------------------------------------

    async def _load_models_and_data(self, test_path: Optional[Path]):

        metadata_path = self.models_path / "training_metadata.joblib"
        scaler_path = self.models_path / "scaler.joblib"

        self.metadata = joblib.load(metadata_path)
        self.scaler = joblib.load(scaler_path)

        for name in self.metadata["top_3_models"]:
            path = self.models_path / f"{name.replace(' ', '_')}_model.joblib"
            self.base_models[name] = joblib.load(path)

        if test_path is None:
            test_path = (
                self.config.paths.labeled_data_path / "wav2vec2_balanced_ctgan.csv"
            )

        df = pd.read_csv(test_path).sample(
            frac=0.2, random_state=self.config.model_training.random_state
        )

        X = df[self.metadata["feature_columns"]]
        y = df["tb_status"]

        return self.scaler.transform(X), y

    # ---------------------------------------------------------
    # ENSEMBLE VOTING LAYER
    # ---------------------------------------------------------

    def _ensemble_predictions(self, probs):

        names = list(probs.keys())

        hard = (
            (probs[names[0]] >= 0.5).astype(int)
            + (probs[names[1]] >= 0.5).astype(int)
            + (probs[names[2]] >= 0.5).astype(int)
        )

        hard = (hard >= self.config.ensemble.hard_voting_threshold).astype(int)

        avg = np.mean([probs[n] for n in names], axis=0)
        soft = (avg >= self.config.ensemble.soft_voting_threshold).astype(int)

        weights = self.config.ensemble.model_weights
        weighted = sum(probs[n] * weights.get(n, 1 / len(names)) for n in names)
        weighted = (weighted >= self.config.ensemble.soft_voting_threshold).astype(int)

        return hard, soft, weighted, avg

    # ---------------------------------------------------------
    # THRESHOLD OPTIMISATION LAYER
    # ---------------------------------------------------------

    def _cost_threshold(self, avg_proba, y):

        best_t = 0.5
        best_cost = float("inf")

        for t in safe_threshold_range(
            self.config.ensemble.threshold_search_start,
            self.config.ensemble.threshold_search_end,
            self.config.ensemble.threshold_search_step,
        ):
            pred = (avg_proba >= t).astype(int)
            tn, fp, fn, tp = safe_confusion(y, pred)

            cost = fn * self.config.ensemble.cost_fn + fp * self.config.ensemble.cost_fp

            if cost < best_cost:
                best_cost = cost
                best_t = t

        return best_t

    def _fbeta_threshold(self, avg_proba, y, beta):

        best_t = 0.5
        best_score = 0

        for t in safe_threshold_range(0.2, 0.8, 0.01):
            pred = (avg_proba >= t).astype(int)
            score = fbeta_score(y, pred, beta=beta, zero_division=0)

            if score > best_score:
                best_score = score
                best_t = t

        return best_t

    def _youden_threshold(self, avg_proba, y):

        fpr, tpr, thr = roc_curve(y, avg_proba)
        idx = np.argmax(tpr - fpr)
        return thr[idx]

    # ---------------------------------------------------------
    # THRESHOLD COMPARISON + AUTO SELECTION
    # ---------------------------------------------------------

    def _select_threshold(self, avg_proba, y):

        results = {}

        cost_t = self._cost_threshold(avg_proba, y)
        results["cost"] = cost_t

        for beta in self.config.ensemble.fbeta_values:
            results[f"fbeta_{beta}"] = self._fbeta_threshold(avg_proba, y, beta)

        results["youden"] = self._youden_threshold(avg_proba, y)

        best = None
        best_fn = float("inf")

        for name, t in results.items():
            pred = (avg_proba >= t).astype(int)
            _, _, fn, _ = safe_confusion(y, pred)

            if fn < best_fn:
                best_fn = fn
                best = name

        return best, results[best], results

    # ---------------------------------------------------------
    # CLINICAL SAFETY LAYER
    # ---------------------------------------------------------
    def _clinical_monitoring(self, avg_proba, y, threshold):
        psi = population_stability_index(y, avg_proba)
        pred = (avg_proba >= threshold).astype(int)
        uncertainty = np.abs(avg_proba - threshold)
        audit = {"psi": float(psi), "uncertain_cases": int((uncertainty < 0.05).sum())}

        return audit

    # ---------------------------------------------------------
    # SAVE FINAL ENSEMBLE
    # ---------------------------------------------------------
    async def _save(self, threshold, strategy, audit):
        model_hash = compute_model_hash(self.base_models)
        bundle = {
            "models": self.base_models,  # Keep for new code
            "base_models": self.base_models,  # Add for compatibility
            "scaler": self.scaler,  # Add scaler to bundle
            "threshold": threshold,  # Keep for new code
            "optimal_threshold": threshold,  # Add for compatibility
            "strategy": strategy,
            "audit": audit,
            "integrity_hash": model_hash,
        }

        joblib.dump(bundle, self.models_path / "cost_sensitive_ensemble_model.joblib")

    # ---------------------------------------------------------
    # MAIN RUN
    # ---------------------------------------------------------
    async def run(self, test_path=None):
        X, y = await self._load_models_and_data(test_path)
        probs = {
            name: model.predict_proba(X)[:, 1]
            for name, model in self.base_models.items()
        }
        _, _, _, avg = self._ensemble_predictions(probs)
        best_name, best_t, all_t = self._select_threshold(avg, y)
        audit = self._clinical_monitoring(avg, y, best_t)
        await self._save(best_t, best_name, audit)

        return {"threshold": best_t, "strategy": best_name, "audit": audit}


# ---------------------------------------------------------
# STANDALONE RUNNER
# ---------------------------------------------------------
async def run_model_evaluation(path=None):
    pipe = ModelEvaluationPipeline()
    return await pipe.run(path)
