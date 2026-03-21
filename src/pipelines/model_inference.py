"""
Model Inference Pipeline for TBFusionAI.

Provides real-time inference with:
1. Single and batch prediction
2. Confidence assessment
3. Feature importance explanation
4. Ensemble prediction aggregation
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import get_config
from src.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class ModelInferencePipeline:
    """
    Real-time inference pipeline with explainability.

    Provides:
    - Single and batch predictions
    - Confidence scores
    - Feature importance
    - Uncertainty quantification
    """

    def __init__(self):
        """Initialize the inference pipeline."""
        self.config = get_config()
        self.models_path = self.config.paths.models_path

        # Model components
        self.ensemble_model: Optional[Dict] = None
        self.base_models: Dict = {}
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Dict = {}
        self.optimal_threshold: float = 0.5

        # Load models
        self._load_models()

        logger.info("ModelInferencePipeline initialized and ready")

    def _load_models(self) -> None:
        """Load all necessary models and components with defensive error handling."""
        try:
            # 1. Load ensemble model
            ensemble_path = self.models_path / "cost_sensitive_ensemble_model.joblib"
            self.ensemble_model = joblib.load(ensemble_path)

            # Use .get() to prevent crashes if metadata keys differ slightly
            if self.ensemble_model is None:
                raise ValueError("Ensemble model failed to load from joblib")
            self.base_models = self.ensemble_model.get("base_models", {})
            self.optimal_threshold = self.ensemble_model.get("optimal_threshold", 0.5)
            strategy = self.ensemble_model.get("strategy", "unknown")

            logger.info("✓ Loaded ensemble model")
            logger.info(f"  Strategy: {strategy}")
            logger.info(f"  Optimal threshold: {self.optimal_threshold:.3f}")

            # 2. Load scaler
            scaler_path = self.models_path / "scaler.joblib"
            self.scaler = joblib.load(scaler_path)
            logger.info("✓ Loaded scaler")

            # 3. Load metadata
            metadata_path = self.models_path / "training_metadata.joblib"
            self.metadata = joblib.load(metadata_path)

            # FIX: Handle missing 'n_features' by checking alternative keys
            n_features = self.metadata.get("n_features") or self.metadata.get(
                "feature_count", "Unknown"
            )

            logger.info("✓ Loaded metadata")
            logger.info(f"  Features: {n_features}")

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

    async def predict_single(self, features: Dict[str, Union[int, float, str]]) -> Dict:
        """Make prediction for a single sample."""
        df = pd.DataFrame([features])
        results = await self.predict_batch(df)

        return {
            "prediction": results["predictions"][0],
            "prediction_label": results["prediction_labels"][0],
            "probability": results["probabilities"][0],
            "confidence": results["confidence_scores"][0],
            "confidence_level": results["confidence_levels"][0],
        }

    async def predict_batch(self, data: pd.DataFrame) -> Dict:
        """Make predictions for a batch of samples."""
        logger.info(f"Making predictions for {len(data)} samples")

        X_prepared = self._prepare_features(data)

        model_probabilities = []
        model_names = list(self.base_models.keys())

        for model_name, model in self.base_models.items():
            y_proba = model.predict_proba(X_prepared)[:, 1]
            model_probabilities.append(y_proba)

        # Ensemble: Soft voting (mean probability)
        avg_proba = np.mean(model_probabilities, axis=0)
        predictions = (avg_proba >= self.optimal_threshold).astype(int)

        # Distance from decision boundary
        confidence = np.abs(avg_proba - self.optimal_threshold)
        confidence_levels = self._classify_confidence(confidence)
        prediction_labels = ["TB+" if p == 1 else "TB-" for p in predictions]

        results = {
            "predictions": predictions,
            "prediction_labels": prediction_labels,
            "probabilities": avg_proba,
            "confidence_scores": confidence,
            "confidence_levels": confidence_levels,
            "model_probabilities": {
                name: probs for name, probs in zip(model_names, model_probabilities)
            },
        }

        logger.info(
            f"✓ Predictions complete (TB+: {(predictions == 1).sum()}, TB-: {(predictions == 0).sum()})"
        )
        return results

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare and encode features for inference."""
        df = data.copy()

        # Handle Categorical 'sex'
        if "sex" in df.columns and df["sex"].dtype == "object":
            df["sex"] = (df["sex"].str.lower() == "male").astype(int)

        # Handle Binary Features (Yes/No)
        binary_features = self.config.metadata.binary_features
        for col in binary_features:
            if col in df.columns and df[col].dtype == "object":
                df[col] = (df[col].str.lower() == "yes").astype(int)

        # Compatibility: Add noise features if missing
        noise_features = self.metadata.get("noise_features", [])
        for noise_col in noise_features:
            if noise_col not in df.columns:
                df[noise_col] = np.random.randn(len(df))

        # Ensure correct column ordering based on training metadata
        all_features = self.metadata.get("feature_columns", [])
        if not all_features:
            raise ValueError(
                "No 'feature_columns' found in metadata. Cannot align features."
            )

        if self.scaler is None:
            raise RuntimeError(
                "Scaler is not initialized. Model loading may have failed."
            )

        X = df[all_features].values
        return self.scaler.transform(X)

    def _classify_confidence(self, confidence: np.ndarray) -> List[str]:
        """Classify confidence scores into High/Medium/Uncertain."""
        levels = []
        high_thresh = self.config.inference.high_confidence_threshold
        uncert_thresh = self.config.inference.uncertainty_threshold

        for conf in confidence:
            if conf > high_thresh:
                levels.append("High")
            elif conf < uncert_thresh:
                levels.append("Uncertain")
            else:
                levels.append("Medium")
        return levels

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get weighted ensemble feature importance."""
        if top_n is None:
            top_n = self.config.inference.top_features_display

        model_weights = self.config.ensemble.model_weights
        feature_cols = self.metadata.get("feature_columns", [])
        ensemble_importance = np.zeros(len(feature_cols))

        for model_name, model in self.base_models.items():
            if hasattr(model, "feature_importances_"):
                weight = model_weights.get(model_name, 1.0 / len(self.base_models))
                ensemble_importance += model.feature_importances_ * weight

        importance_df = pd.DataFrame(
            {
                "Feature": feature_cols,
                "Importance": ensemble_importance,
            }
        ).sort_values("Importance", ascending=False)

        return importance_df.head(top_n) if top_n else importance_df

    def explain_prediction(
        self, features: Dict[str, Union[int, float, str]], top_features: int = 5
    ) -> Dict:
        """Provide detailed explanation for a single prediction."""
        import asyncio

        prediction_result = asyncio.run(self.predict_single(features))
        importance_df = self.get_feature_importance(top_n=top_features)

        top_feature_values = []
        for _, row in importance_df.iterrows():
            f_name = row["Feature"]
            if f_name in features:
                top_feature_values.append(
                    {
                        "feature": f_name,
                        "value": features[f_name],
                        "importance": row["Importance"],
                    }
                )

        return {
            "prediction": prediction_result["prediction_label"],
            "probability": prediction_result["probability"],
            "confidence": prediction_result["confidence"],
            "confidence_level": prediction_result["confidence_level"],
            "top_contributing_features": top_feature_values,
            "recommendation": self._generate_recommendation(prediction_result),
        }

    def _generate_recommendation(self, res: Dict) -> str:
        """Generate clinical recommendation logic."""
        pred, conf_lvl = res["prediction"], res["confidence_level"]

        if pred == 1:  # TB+
            if conf_lvl == "High":
                return "High probability of TB. Recommend immediate clinical follow-up and confirmatory testing."
            return "Moderate or uncertain TB probability. Recommend clinical evaluation and patient symptom review."

        # TB-
        if conf_lvl == "High":
            return "Low probability of TB. Continue routine monitoring if symptoms persist."
        return "TB unlikely but follow-up recommended if symptoms worsen."

    def get_model_info(self) -> Dict:
        """FIXED: Robustly return model info without KeyError."""
        return {
            "ensemble_strategy": (
                self.ensemble_model.get("strategy", "unknown")
                if self.ensemble_model
                else "unknown"
            ),
            "optimal_threshold": self.optimal_threshold,
            "base_models": list(self.base_models.keys()),
            "model_weights": self.config.ensemble.model_weights,
            "n_features": self.metadata.get("n_features")
            or self.metadata.get("feature_count", 0),
            "clinical_features": self.metadata.get("clinical_features", []),
            "performance": (
                self.ensemble_model.get("performance", {})
                if self.ensemble_model
                else {}
            ),
        }


# Convenience wrapper
async def predict(features: Dict, explain: bool = False) -> Dict:
    pipeline = ModelInferencePipeline()
    return (
        pipeline.explain_prediction(features)
        if explain
        else await pipeline.predict_single(features)
    )
