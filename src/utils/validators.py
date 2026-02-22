"""
Validation utilities for TBFusionAI.

Provides validation functions for:
- Audio files
- Clinical features
- Feature arrays
- File paths
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)


def validate_audio_format(
    file_path: Union[str, Path], allowed_formats: Optional[List[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate audio file format.

    Args:
        file_path: Path to audio file
        allowed_formats: List of allowed formats. If None, uses config.

    Returns:
        Tuple of (is_valid, error_message)
    """
    config = get_config()

    if allowed_formats is None:
        allowed_formats = config.api.allowed_audio_formats

    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    # Check if it's a file
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    # Check file extension
    file_ext = file_path.suffix.lower()
    if file_ext not in allowed_formats:
        return False, f"Unsupported format '{file_ext}'. Allowed: {allowed_formats}"

    # Check file size
    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "File is empty"

    if file_size > config.api.max_upload_size:
        max_mb = config.api.max_upload_size / (1024 * 1024)
        return False, f"File exceeds maximum size of {max_mb}MB"

    return True, None


def validate_clinical_features(
    features: Dict[str, Union[int, float, str]],
    required_features: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate clinical features.

    Args:
        features: Dictionary of clinical features
        required_features: List of required features. If None, uses config.

    Returns:
        Tuple of (is_valid, error_message)
    """
    config = get_config()

    if required_features is None:
        required_features = config.metadata.clinical_features

    # Check for required features
    missing_features = []
    for feature in required_features:
        if feature not in features:
            missing_features.append(feature)

    if missing_features:
        return False, f"Missing required features: {missing_features}"

    # Validate age
    if "age" in features:
        age = features["age"]
        if not isinstance(age, (int, float)):
            return False, "Age must be a number"
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120"

    # Validate sex
    if "sex" in features:
        sex = str(features["sex"]).lower()
        if sex not in ["male", "female", "m", "f"]:
            return False, "Sex must be 'Male' or 'Female'"

    # Validate cough duration
    if "reported_cough_dur" in features:
        duration = features["reported_cough_dur"]
        if not isinstance(duration, (int, float)):
            return False, "Cough duration must be a number"
        if duration < 0:
            return False, "Cough duration cannot be negative"

    # Validate binary features
    binary_features = config.metadata.binary_features
    for feature in binary_features:
        if feature in features:
            value = str(features[feature]).lower()
            if value not in ["yes", "no", "y", "n", "0", "1"]:
                return False, f"{feature} must be 'Yes' or 'No'"

    return True, None


def validate_feature_array(
    X: np.ndarray, expected_features: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate feature array shape and content.

    Args:
        X: Feature array
        expected_features: Expected number of features. If None, uses config.

    Returns:
        Tuple of (is_valid, error_message)
    """
    config = get_config()

    # Check if numpy array
    if not isinstance(X, np.ndarray):
        return False, "Features must be a numpy array"

    # Check dimensions
    if X.ndim != 2:
        return False, f"Features must be 2D array, got {X.ndim}D"

    # Check number of features
    if expected_features is None:
        try:
            import joblib

            metadata_path = config.paths.models_path / "training_metadata.joblib"
            metadata = joblib.load(metadata_path)
            expected_features = metadata["n_features"]
        except Exception:
            logger.warning("Could not load metadata for feature validation")
            expected_features = None

    if expected_features is not None:
        if X.shape[1] != expected_features:
            return False, f"Expected {expected_features} features, got {X.shape[1]}"

    # Check for NaN or infinite values
    if np.isnan(X).any():
        return False, "Features contain NaN values"

    if np.isinf(X).any():
        return False, "Features contain infinite values"

    return True, None


def validate_file_path(
    file_path: Union[str, Path], must_exist: bool = True, must_be_file: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate file path.

    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        must_be_file: Whether path must be a file (not directory)

    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(file_path)

    # Check existence
    if must_exist and not file_path.exists():
        return False, f"Path does not exist: {file_path}"

    # Check if file
    if must_be_file and file_path.exists() and not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    # Check parent directory exists (for new files)
    if not must_exist and not file_path.parent.exists():
        return False, f"Parent directory does not exist: {file_path.parent}"

    return True, None


def validate_probability(
    probability: float, min_val: float = 0.0, max_val: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate probability value.

    Args:
        probability: Probability value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(probability, (int, float)):
        return False, "Probability must be a number"

    if np.isnan(probability):
        return False, "Probability is NaN"

    if np.isinf(probability):
        return False, "Probability is infinite"

    if probability < min_val or probability > max_val:
        return False, f"Probability must be between {min_val} and {max_val}"

    return True, None


def validate_threshold(
    threshold: float, min_val: float = 0.0, max_val: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate threshold value.

    Args:
        threshold: Threshold value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Tuple of (is_valid, error_message)
    """
    return validate_probability(threshold, min_val, max_val)


def validate_batch_size(
    batch_size: int, min_size: int = 1, max_size: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate batch size.

    Args:
        batch_size: Batch size to validate
        min_size: Minimum allowed size
        max_size: Maximum allowed size. If None, no maximum.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(batch_size, int):
        return False, "Batch size must be an integer"

    if batch_size < min_size:
        return False, f"Batch size must be at least {min_size}"

    if max_size is not None and batch_size > max_size:
        return False, f"Batch size cannot exceed {max_size}"

    return True, None


def validate_model_output(
    predictions: np.ndarray, probabilities: np.ndarray
) -> Tuple[bool, Optional[str]]:
    """
    Validate model output arrays.

    Args:
        predictions: Binary predictions array
        probabilities: Probability predictions array

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check predictions
    if not isinstance(predictions, np.ndarray):
        return False, "Predictions must be a numpy array"

    if predictions.ndim != 1:
        return False, "Predictions must be 1D array"

    if not np.all(np.isin(predictions, [0, 1])):
        return False, "Predictions must be binary (0 or 1)"

    # Check probabilities
    if not isinstance(probabilities, np.ndarray):
        return False, "Probabilities must be a numpy array"

    if probabilities.ndim != 1:
        return False, "Probabilities must be 1D array"

    if len(predictions) != len(probabilities):
        return False, "Predictions and probabilities must have same length"

    # Validate probability values
    for prob in probabilities:
        is_valid, error = validate_probability(prob)
        if not is_valid:
            return False, f"Invalid probability: {error}"

    return True, None
