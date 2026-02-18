"""
Ensemble Model for TBFusionAI.

Implements advanced ensemble strategies:
- Soft voting with weighted probabilities
- Cost-sensitive threshold optimization
- Confidence-based predictions
"""

from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel:
    """
    Advanced ensemble model with cost-sensitive optimization.
    
    Combines multiple base models using:
    - Weighted soft voting
    - Cost-sensitive threshold
    - Confidence assessment
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ensemble model.
        
        Args:
            model_path: Path to saved ensemble model. If None, uses default.
        """
        self.config = get_config()
        
        if model_path is None:
            model_path = self.config.paths.models_path / "cost_sensitive_ensemble_model.joblib"
        
        # Load ensemble model
        self.ensemble_data = joblib.load(model_path)
        
        # Handle both key names for compatibility
        if 'base_models' in self.ensemble_data:
            self.base_models = self.ensemble_data['base_models']
        elif 'models' in self.ensemble_data:
            self.base_models = self.ensemble_data['models']
        else:
            raise KeyError(
                f"Bundle missing 'models' or 'base_models' key. "
                f"Available keys: {list(self.ensemble_data.keys())}"
            )
        
        # Handle both threshold key names
        if 'optimal_threshold' in self.ensemble_data:
            self.optimal_threshold = self.ensemble_data['optimal_threshold']
        elif 'threshold' in self.ensemble_data:
            self.optimal_threshold = self.ensemble_data['threshold']
        else:
            self.optimal_threshold = 0.5  # Fallback default

        self.strategy = self.ensemble_data.get('strategy', 'ensemble')
        self.model_weights = self.ensemble_data.get(
            'model_weights',
            {name: 1.0 / len(self.base_models) for name in self.base_models.keys()}
        )
        
        # Load scaler from bundle first, fallback to file
        if 'scaler' in self.ensemble_data:
            self.scaler = self.ensemble_data['scaler']
            logger.info("Loaded scaler from bundle")
        else:
            # Fallback: load from separate file
            scaler_path = self.config.paths.models_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler from separate file")
            else:
                raise FileNotFoundError(
                    "Scaler not found in bundle or as separate file at "
                    f"{scaler_path}"
                )
        
        # Add these attributes for compatibility (DEFINE FIRST!)
        self.model_names = list(self.base_models.keys())
        self.n_models = len(self.base_models)
        self.audit = self.ensemble_data.get('audit', {})  # ← ADD THIS LINE

        # Now log using the attributes
        logger.info(f"✓ Ensemble loaded successfully:")
        logger.info(f"  - Models: {self.n_models} ({', '.join(self.model_names)})")
        logger.info(f"  - Strategy: {self.strategy}")
        logger.info(f"  - Threshold: {self.optimal_threshold:.4f}")
        logger.info(f"  - PSI (drift): {self.audit.get('psi', 'N/A')}")
         
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble.
        
        Args:
            X: Feature array (scaled)
        
        Returns:
            Array of predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.optimal_threshold).astype(int)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions using weighted soft voting.
        
        Args:
            X: Feature array (scaled)
        
        Returns:
            Array of probabilities for positive class
        """
        weighted_probs = []
        
        for model_name, model in self.base_models.items():
            # Get probabilities from this model
            proba = model.predict_proba(X)[:, 1]
            
            # Apply model weight
            weight = self.model_weights.get(model_name, 1.0 / len(self.base_models))
            weighted_probs.append(proba * weight)
        
        # Average weighted probabilities
        ensemble_proba = np.sum(weighted_probs, axis=0)
        
        return ensemble_proba
    
    def predict_with_confidence(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Feature array (scaled)
        
        Returns:
            Tuple of (predictions, probabilities, confidence_scores)
        """
        # Get probabilities
        probabilities = self.predict_proba(X)
        
        # Make predictions
        predictions = (probabilities >= self.optimal_threshold).astype(int)
        
        # Calculate confidence (distance from decision boundary)
        confidence = np.abs(probabilities - self.optimal_threshold)
        
        return predictions, probabilities, confidence
    
    def get_model_agreement(self, X: np.ndarray) -> np.ndarray:
        """
        Get agreement scores across base models.
        
        Args:
            X: Feature array (scaled)
        
        Returns:
            Array of agreement scores (0-3 for 3 models)
        """
        predictions = []
        
        for model in self.base_models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Sum predictions (agreement score)
        agreement = np.sum(predictions, axis=0)
        
        return agreement
    
    def identify_uncertain_predictions(
        self,
        X: np.ndarray,
        confidence_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Identify predictions that should be flagged for review.
        
        Args:
            X: Feature array (scaled)
            confidence_threshold: Threshold for flagging. If None, uses config.
        
        Returns:
            Boolean array indicating uncertain predictions
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.ensemble.confidence_threshold
        
        # Get confidence scores
        _, probabilities, _ = self.predict_with_confidence(X)
        
        # Calculate probability standard deviation across models
        model_probas = []
        for model in self.base_models.values():
            proba = model.predict_proba(X)[:, 1]
            model_probas.append(proba)
        
        proba_std = np.std(model_probas, axis=0)
        
        # Get agreement scores
        agreement = self.get_model_agreement(X)
        
        # Flag uncertain: high std OR low agreement
        uncertain = (proba_std > confidence_threshold) | (agreement == 1)
        
        return uncertain
    
    def evaluate_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_fn: Optional[int] = None,
        cost_fp: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate prediction cost.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_fn: Cost of false negative. If None, uses config.
            cost_fp: Cost of false positive. If None, uses config.
        
        Returns:
            Dictionary with cost breakdown
        """
        if cost_fn is None:
            cost_fn = self.config.ensemble.cost_fn
        if cost_fp is None:
            cost_fp = self.config.ensemble.cost_fp
        
        # Calculate confusion matrix elements
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        
        # Calculate costs
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'cost_fn': int(fn * cost_fn),
            'cost_fp': int(fp * cost_fp),
            'total_cost': int(total_cost)
        }
    
    def get_feature_importance(
        self,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Get ensemble feature importance.
        
        Args:
            feature_names: List of feature names
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        ensemble_importance = np.zeros(len(feature_names))
        
        for model_name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                weight = self.model_weights.get(
                    model_name,
                    1.0 / len(self.base_models)
                )
                ensemble_importance += model.feature_importances_ * weight
        
        # Create dictionary
        importance_dict = {
            name: float(importance)
            for name, importance in zip(feature_names, ensemble_importance)
        }
        
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def get_model_info(self) -> Dict:
        """
        Get information about the ensemble model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'strategy': self.strategy,
            'optimal_threshold': self.optimal_threshold,
            'base_models': list(self.base_models.keys()),
            'model_weights': self.model_weights,
            'cost_fn': self.ensemble_data.get('cost_fn'),
            'cost_fp': self.ensemble_data.get('cost_fp'),
            'performance': self.ensemble_data.get('performance', {})
        }


# Convenience function
def load_ensemble_model(model_path: Optional[str] = None) -> EnsembleModel:
    """
    Load ensemble model.
    
    Args:
        model_path: Optional path to model file
    
    Returns:
        EnsembleModel instance
    """
    return EnsembleModel(model_path)


if __name__ == "__main__":
    # Test ensemble model loading
    try:
        model = EnsembleModel()
        info = model.get_model_info()
        
        logger.info("\nEnsemble Model Info:")
        logger.info(f"  Strategy: {info['strategy']}")
        logger.info(f"  Threshold: {info['optimal_threshold']:.3f}")
        logger.info(f"  Base Models: {info['base_models']}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")