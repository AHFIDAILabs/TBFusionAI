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
        """Load all necessary models and components."""
        try:
            # Load ensemble model
            ensemble_path = self.models_path / "cost_sensitive_ensemble_model.joblib"
            self.ensemble_model = joblib.load(ensemble_path)
            self.base_models = self.ensemble_model["base_models"]
            self.optimal_threshold = self.ensemble_model["optimal_threshold"]

            logger.info("✓ Loaded ensemble model")
            logger.info(f"  Strategy: {self.ensemble_model['strategy']}")
            logger.info(f"  Optimal threshold: {self.optimal_threshold:.3f}")

            # Load scaler
            scaler_path = self.models_path / "scaler.joblib"
            self.scaler = joblib.load(scaler_path)
            logger.info("✓ Loaded scaler")

            # Load metadata
            metadata_path = self.models_path / "training_metadata.joblib"
            self.metadata = joblib.load(metadata_path)
            logger.info("✓ Loaded metadata")
            logger.info(f"  Features: {self.metadata['n_features']}")

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

    async def predict_single(self, features: Dict[str, Union[int, float, str]]) -> Dict:
        """
        Make prediction for a single sample.

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with prediction, probability, and confidence
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Make prediction
        results = await self.predict_batch(df)

        # Return first (and only) result
        return {
            "prediction": results["predictions"][0],
            "prediction_label": results["prediction_labels"][0],
            "probability": results["probabilities"][0],
            "confidence": results["confidence_scores"][0],
            "confidence_level": results["confidence_levels"][0],
        }

    async def predict_batch(self, data: pd.DataFrame) -> Dict:
        """
        Make predictions for a batch of samples.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary with batch predictions and metadata
        """
        logger.info(f"Making predictions for {len(data)} samples")

        # Prepare features
        X_prepared = self._prepare_features(data)

        # Get predictions from all base models
        model_probabilities = []
        model_names = list(self.base_models.keys())

        for model_name, model in self.base_models.items():
            y_proba = model.predict_proba(X_prepared)[:, 1]
            model_probabilities.append(y_proba)

        # Ensemble: Average probabilities (soft voting)
        avg_proba = np.mean(model_probabilities, axis=0)

        # Apply optimal cost-sensitive threshold
        predictions = (avg_proba >= self.optimal_threshold).astype(int)

        # Calculate confidence (distance from decision boundary)
        confidence = np.abs(avg_proba - self.optimal_threshold)

        # Classify confidence levels
        confidence_levels = self._classify_confidence(confidence)

        # Prepare prediction labels
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

        logger.info(f"✓ Predictions complete")
        logger.info(f"  TB+: {(predictions == 1).sum()}")
        logger.info(f"  TB-: {(predictions == 0).sum()}")

        return results

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for inference.

        Args:
            data: Raw feature DataFrame

        Returns:
            Scaled feature array
        """
        df = data.copy()

        # Encode categorical features
        if "sex" in df.columns:
            if df["sex"].dtype == "object":
                df["sex"] = (df["sex"] == "Male").astype(int)

        # Encode binary features
        binary_features = self.config.metadata.binary_features
        for col in binary_features:
            if col in df.columns and df[col].dtype == "object":
                df[col] = (df[col] == "Yes").astype(int)

        # Add noise features if missing (for compatibility)
        noise_features = self.metadata.get("noise_features", [])
        for noise_col in noise_features:
            if noise_col not in df.columns:
                df[noise_col] = np.random.randn(len(df))

        # Select features in correct order
        all_features = self.metadata["feature_columns"]
        X = df[all_features].values

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def _classify_confidence(self, confidence: np.ndarray) -> List[str]:
        """
        Classify confidence scores into levels.

        Args:
            confidence: Array of confidence scores

        Returns:
            List of confidence level labels
        """
        levels = []

        for conf in confidence:
            if conf > self.config.inference.high_confidence_threshold:
                levels.append("High")
            elif conf < self.config.inference.uncertainty_threshold:
                levels.append("Uncertain")
            else:
                levels.append("Medium")

        return levels

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get ensemble feature importance.

        Args:
            top_n: Number of top features to return. If None, returns all.

        Returns:
            DataFrame with feature importance rankings
        """
        if top_n is None:
            top_n = self.config.inference.top_features_display

        # Get feature importance from all models
        model_weights = self.config.ensemble.model_weights
        model_names = list(self.base_models.keys())

        # Calculate weighted ensemble importance
        ensemble_importance = np.zeros(len(self.metadata["feature_columns"]))

        for model_name, model in self.base_models.items():
            if hasattr(model, "feature_importances_"):
                weight = model_weights.get(model_name, 1.0 / len(self.base_models))
                ensemble_importance += model.feature_importances_ * weight

        # Create DataFrame
        importance_df = pd.DataFrame(
            {
                "Feature": self.metadata["feature_columns"],
                "Importance": ensemble_importance,
            }
        ).sort_values("Importance", ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def explain_prediction(
        self, features: Dict[str, Union[int, float, str]], top_features: int = 5
    ) -> Dict:
        """
        Provide detailed explanation for a single prediction.

        Args:
            features: Feature dictionary
            top_features: Number of top contributing features to show

        Returns:
            Dictionary with prediction explanation
        """
        # Make prediction
        import asyncio

        prediction_result = asyncio.run(self.predict_single(features))

        # Get feature importance
        importance_df = self.get_feature_importance(top_n=top_features)

        # Get feature values for top features
        top_feature_values = []
        for _, row in importance_df.iterrows():
            feature_name = row["Feature"]
            if feature_name in features:
                top_feature_values.append(
                    {
                        "feature": feature_name,
                        "value": features[feature_name],
                        "importance": row["Importance"],
                    }
                )

        explanation = {
            "prediction": prediction_result["prediction_label"],
            "probability": prediction_result["probability"],
            "confidence": prediction_result["confidence"],
            "confidence_level": prediction_result["confidence_level"],
            "top_contributing_features": top_feature_values,
            "recommendation": self._generate_recommendation(prediction_result),
        }

        return explanation

    def _generate_recommendation(self, prediction_result: Dict) -> str:
        """
        Generate clinical recommendation based on prediction.

        Args:
            prediction_result: Prediction result dictionary

        Returns:
            Recommendation text
        """
        prediction = prediction_result["prediction"]
        confidence = prediction_result["confidence"]
        confidence_level = prediction_result["confidence_level"]

        if prediction == 1:  # TB+
            if confidence_level == "High":
                return (
                    "High probability of TB detected. "
                    "Recommend immediate clinical follow-up and confirmatory testing. "
                    "This patient should be prioritized for diagnostic evaluation."
                )
            elif confidence_level == "Medium":
                return (
                    "Moderate probability of TB detected. "
                    "Recommend clinical evaluation and follow-up testing. "
                    "Consider patient's symptoms and risk factors."
                )
            else:  # Uncertain
                return (
                    "TB possible but prediction uncertain. "
                    "Recommend comprehensive clinical assessment. "
                    "Additional testing may be needed for definitive diagnosis."
                )
        else:  # TB-
            if confidence_level == "High":
                return (
                    "Low probability of TB. "
                    "Continue routine monitoring if symptomatic. "
                    "Re-evaluate if symptoms worsen or persist."
                )
            else:
                return (
                    "TB unlikely but follow-up recommended. "
                    "Monitor symptoms and consider re-testing if concerns persist. "
                    "Clinical judgment should guide next steps."
                )

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded models.

        Returns:
            Dictionary with model information
        """
        return {
            "ensemble_strategy": self.ensemble_model["strategy"],
            "optimal_threshold": self.optimal_threshold,
            "base_models": list(self.base_models.keys()),
            "model_weights": self.config.ensemble.model_weights,
            "n_features": self.metadata["n_features"],
            "clinical_features": self.metadata["clinical_features"],
            "performance": self.ensemble_model["performance"],
        }


# Convenience function for quick inference
async def predict(
    features: Dict[str, Union[int, float, str]], explain: bool = False
) -> Dict:
    """
    Quick prediction function.

    Args:
        features: Dictionary of feature values
        explain: Whether to include detailed explanation

    Returns:
        Prediction result dictionary
    """
    pipeline = ModelInferencePipeline()

    if explain:
        return pipeline.explain_prediction(features)
    else:
        return await pipeline.predict_single(features)


if __name__ == "__main__":
    import asyncio

    # Example usage
    sample_features = {
        "age": 45,
        "sex": "Male",
        "reported_cough_dur": 21,
        "tb_prior": "No",
        "hemoptysis": "No",
        "weight_loss": "Yes",
        "fever": "Yes",
        "night_sweats": "No",
        # Audio features would be extracted from audio file
        **{f"feat_{i}": np.random.randn() for i in range(768)},
    }

    # Make prediction with explanation
    pipeline = ModelInferencePipeline()
    result = pipeline.explain_prediction(sample_features)

    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION RESULT")
    logger.info("=" * 70)
    logger.info(f"Prediction: {result['prediction']}")
    logger.info(f"Probability: {result['probability']:.4f}")
    logger.info(
        f"Confidence: {result['confidence']:.4f} ({result['confidence_level']})"
    )
    logger.info(f"\nRecommendation:")
    logger.info(f"  {result['recommendation']}")
