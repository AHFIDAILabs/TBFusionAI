"""
High-level Predictor for TBFusionAI.

Provides unified interface for TB prediction combining:
- Audio preprocessing
- Feature extraction
- Ensemble prediction
- Result interpretation
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from src.config import get_config
from src.logger import get_logger
from src.models.ensemble_model import EnsembleModel
from src.models.preprocessor import AudioPreprocessor, FeaturePreprocessor

logger = get_logger(__name__)


class TBPredictor:
    """
    High-level TB prediction interface.

    Orchestrates the complete prediction pipeline:
    1. Audio preprocessing
    2. Feature extraction
    3. Ensemble prediction
    4. Result interpretation
    """

    def __init__(self):
        """Initialize the TB predictor."""
        self.config = get_config()

        # Initialize components
        self.audio_preprocessor = AudioPreprocessor()
        self.feature_preprocessor = FeaturePreprocessor()
        self.ensemble_model = EnsembleModel()

        # Load metadata
        import joblib

        metadata_path = self.config.paths.models_path / "training_metadata.joblib"
        self.metadata = joblib.load(metadata_path)

        logger.info("TBPredictor initialized and ready")

    async def predict_from_audio(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO],
        clinical_features: Dict[str, Union[int, float, str]],
        generate_spectrogram: bool = True,
        validate_quality: bool = False,
    ) -> Dict:
        """
        Make TB prediction from audio file and clinical features.

        Args:
            audio_input: Audio file path, bytes, or BytesIO
            clinical_features: Dictionary of clinical features
            generate_spectrogram: Whether to generate spectrogram image

        Returns:
            Dictionary with prediction results
        """
        logger.info("Starting prediction from audio")

        try:
            # Step 1: Extract audio features
            logger.info("Extracting audio features...")
            audio_features = self.audio_preprocessor.extract_features(
                audio_input, validate_quality
            )

            # Step 2: Generate spectrogram if requested
            spectrogram_image = None
            spectrogram_base64 = None

            if generate_spectrogram:
                logger.info("Generating spectrogram...")
                spectrogram_image = self.audio_preprocessor.generate_spectrogram(
                    audio_input
                )

                # Convert to base64
                buffer = io.BytesIO()
                spectrogram_image.save(buffer, format="PNG")
                spectrogram_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Step 3: Combine features
            logger.info("Preparing features...")
            all_features = self.feature_preprocessor.prepare_feature_dict(
                clinical_features, audio_features
            )

            # Step 4: Make prediction
            logger.info("Making prediction...")
            result = await self.predict_from_features(all_features)

            # Step 5: Add spectrogram to result
            if spectrogram_base64:
                result["spectrogram_base64"] = spectrogram_base64

            logger.info(f"✓ Prediction complete: {result['prediction']}")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    async def predict_from_features(
        self, features: Dict[str, Union[int, float]]
    ) -> Dict:
        """
        Make TB prediction from prepared features.

        Args:
            features: Dictionary of all features (clinical + audio)

        Returns:
            Dictionary with prediction results
        """
        # Validate features
        is_valid, error_msg = self.feature_preprocessor.validate_features(features)
        if not is_valid:
            raise ValueError(error_msg)

        # Prepare feature array
        X = self._prepare_feature_array(features)

        # Scale features
        X_scaled = self.ensemble_model.scaler.transform(X)

        # Make prediction
        predictions, probabilities, confidence_scores = (
            self.ensemble_model.predict_with_confidence(X_scaled)
        )

        # Extract single prediction
        prediction = predictions[0]
        probability = probabilities[0]
        confidence = confidence_scores[0]

        # Classify confidence level
        confidence_level = self._classify_confidence(confidence)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            prediction, probability, confidence_level
        )

        # Generate disclaimer
        disclaimer = self._generate_disclaimer()

        # Prepare result
        result = {
            "prediction": "Probable TB" if prediction == 1 else "Unlikely TB",
            "prediction_class": int(prediction),
            "probability": float(probability),
            "confidence": float(confidence),
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "disclaimer": disclaimer,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "strategy": self.ensemble_model.strategy,
                "threshold": self.ensemble_model.optimal_threshold,
            },
        }

        return result

    def _prepare_feature_array(
        self, features: Dict[str, Union[int, float]]
    ) -> np.ndarray:
        """
        Prepare feature array in correct order.

        Args:
            features: Feature dictionary

        Returns:
            Feature array with shape (1, n_features)
        """
        feature_columns = self.metadata["feature_columns"]

        # Add noise features if missing (for compatibility)
        for noise_col in self.metadata.get("noise_features", []):
            if noise_col not in features:
                features[noise_col] = 0.0

        # Extract features in correct order
        feature_values = []
        for col in feature_columns:
            if col in features:
                feature_values.append(float(features[col]))
            else:
                # Use default value if missing
                feature_values.append(0.0)
                logger.warning(f"Feature '{col}' not found, using default value 0.0")

        # Convert to array
        X = np.array(feature_values).reshape(1, -1)

        return X

    def _classify_confidence(self, confidence: float) -> str:
        """
        Classify confidence score into level.

        Args:
            confidence: Confidence score

        Returns:
            Confidence level string
        """
        if confidence > self.config.inference.high_confidence_threshold:
            return "High"
        elif confidence < self.config.inference.uncertainty_threshold:
            return "Uncertain"
        else:
            return "Medium"

    def _generate_recommendation(
        self, prediction: int, probability: float, confidence_level: str
    ) -> str:
        """
        Generate clinical recommendation based on prediction.

        Args:
            prediction: Prediction class (0 or 1)
            probability: Prediction probability
            confidence_level: Confidence level

        Returns:
            Recommendation text
        """
        if prediction == 1:  # TB+
            if confidence_level == "High":
                return (
                    "High probability of TB detected. "
                    "Immediate clinical follow-up and confirmatory testing recommended. "
                    "This patient should be prioritised for diagnostic evaluation including "
                    "sputum microscopy, culture, and chest X-ray."
                )
            elif confidence_level == "Medium":
                return (
                    "Moderate probability of TB detected. "
                    "Clinical evaluation and follow-up testing strongly recommended. "
                    "Consider patient's symptoms, exposure history, and risk factors. "
                    "Proceed with standard TB diagnostic workup."
                )
            else:  # Uncertain
                return (
                    "TB possible but prediction uncertain. "
                    "Comprehensive clinical assessment recommended. "
                    "Multiple diagnostic tests may be needed for definitive diagnosis. "
                    "Consider repeat testing if symptoms persist."
                )
        else:  # TB-
            if confidence_level == "High":
                return (
                    "Low probability of TB. "
                    "If patient is symptomatic, continue routine monitoring. "
                    "Re-evaluate if symptoms worsen, persist, or new symptoms develop. "
                    "Consider alternative diagnoses if appropriate."
                )
            else:
                return (
                    "TB unlikely but follow-up recommended if symptomatic. "
                    "Monitor patient closely and consider re-testing if concerns persist. "
                    "Clinical judgment should guide next steps based on presentation. "
                    "Rule out other respiratory conditions."
                )

    def _generate_disclaimer(self) -> str:
        """
        Generate standard disclaimer.

        Returns:
            Disclaimer text
        """
        return (
            "This is an AI-assisted screening tool and NOT a diagnostic device. "
            "Results should always be verified by qualified healthcare professionals. "
            "Clinical diagnosis of TB requires confirmatory laboratory testing including "
            "sputum microscopy, culture, or molecular tests (e.g., GeneXpert). "
            "This tool is intended to support, not replace, clinical judgment."
        )

    def explain_prediction(
        self, features: Dict[str, Union[int, float]], top_n: int = 5
    ) -> Dict:
        """
        Provide detailed explanation for prediction.

        Args:
            features: Feature dictionary
            top_n: Number of top features to show

        Returns:
            Dictionary with prediction explanation
        """
        # Get feature importance from ensemble
        feature_importance = self.ensemble_model.get_feature_importance(
            self.metadata["feature_columns"]
        )

        # Get top N features that are in input
        top_features = []
        for feature_name, importance in list(feature_importance.items())[:top_n]:
            if feature_name in features:
                top_features.append(
                    {
                        "feature": feature_name,
                        "value": features[feature_name],
                        "importance": importance,
                    }
                )

        return {
            "top_contributing_features": top_features,
            "feature_importance_available": len(feature_importance) > 0,
        }

    # get_model_info method for src/models/predictor.py
    def get_model_info(self) -> Dict:
        """
        Get information about the prediction system.

        Returns:
            Dictionary with system information
        """
        ensemble_info = self.ensemble_model.get_model_info()

        return {
            "model_version": "1.0.0",
            "ensemble_strategy": ensemble_info["strategy"],
            "base_models": ensemble_info["base_models"],
            "optimal_threshold": ensemble_info["optimal_threshold"],
            # Use .get() with fallback to avoid KeyError
            "feature_count": self.metadata.get(
                "n_features", len(self.metadata.get("feature_columns", []))
            ),
            # Use .get() for safety
            "clinical_features": self.metadata.get("clinical_features", []),
            "performance": ensemble_info.get("performance", {}),
        }


# Convenience function
async def predict(
    audio_input: Union[str, Path, bytes, io.BytesIO],
    clinical_features: Dict[str, Union[int, float, str]],
    generate_spectrogram: bool = True,
) -> Dict:
    """
    Quick prediction function.

    Args:
        audio_input: Audio file path, bytes, or BytesIO
        clinical_features: Dictionary of clinical features
        generate_spectrogram: Whether to generate spectrogram

    Returns:
        Prediction result dictionary
    """
    predictor = TBPredictor()
    return await predictor.predict_from_audio(
        audio_input, clinical_features, generate_spectrogram
    )


if __name__ == "__main__":
    import asyncio

    # Example usage
    async def test_predictor():
        predictor = TBPredictor()

        # Get model info
        info = predictor.get_model_info()

        logger.info("\nTB Predictor Info:")
        logger.info(f"  Model Version: {info['model_version']}")
        logger.info(f"  Strategy: {info['ensemble_strategy']}")
        logger.info(f"  Base Models: {info['base_models']}")
        logger.info(f"  Threshold: {info['optimal_threshold']:.3f}")
        logger.info("\nPredictor ready for inference")

    asyncio.run(test_predictor())
