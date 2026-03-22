"""
Pytest configuration and fixtures - UPDATED FOR CURRENT STRUCTURE.

Provides common fixtures for testing:
- Sample data (Clinical, Audio, DataFrames)
- Mock Model generation (Ensemble, Scaler, Metadata)
- Environment Isolation (Monkeypatching paths)
"""

import io
import wave
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.config import get_config


@pytest.fixture(scope="session")
def test_config():
    """Get the base configuration object."""
    return get_config()


@pytest.fixture(scope="session")
def sample_clinical_features() -> Dict:
    """Sample clinical features for testing."""
    return {
        "age": 45,
        "sex": "Male",
        "reported_cough_dur": 21,
        "tb_prior": "No",
        "hemoptysis": "No",
        "weight_loss": "Yes",
        "fever": "Yes",
        "night_sweats": "No",
    }


@pytest.fixture(scope="session")
def sample_audio_features() -> np.ndarray:
    """Sample audio features (768-dimensional)."""
    return np.random.randn(768)


@pytest.fixture(scope="session")
def sample_full_features(sample_clinical_features, sample_audio_features) -> Dict:
    """Sample full feature set (clinical + audio)."""
    features = sample_clinical_features.copy()

    # Add audio features
    for i, value in enumerate(sample_audio_features):
        features[f"feat_{i}"] = float(value)

    # Add noise features
    for i in range(10):
        features[f"noise_{i}"] = 0.0

    return features


@pytest.fixture(scope="session")
def sample_audio_bytes() -> bytes:
    """Generates a valid 1-second WAV file in memory for testing."""
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    # Generate simple sine wave (A4 note)
    t = np.linspace(0, duration, num_samples)
    audio_data = (np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    buffer.seek(0)
    return buffer.read()


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """
    Generic temporary directory for utility tests.
    Fixes the 'fixture not found' errors in test_utils.py.
    """
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def temp_models_dir(tmp_path_factory):
    """Create a temporary directory for mock model artifacts."""
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def mock_model_artifacts(temp_models_dir):
    """
    Creates real .joblib files in a temp directory with the NEW schema.
    This ensures TBPredictor can actually 'load' something during tests.
    """
    # 1. Create Mock Base Models
    base_models = {
        "CatBoost": RandomForestClassifier(n_estimators=2, random_state=42),
        "Meta-XGBoost": RandomForestClassifier(n_estimators=2, random_state=42),
    }

    # Fit with dummy data (786 features: 768 audio + 8 clinical + 10 noise)
    X_dummy = np.random.randn(10, 786)
    y_dummy = np.random.randint(0, 2, 10)
    for m in base_models.values():
        m.fit(X_dummy, y_dummy)

    # 2. Create Scaler
    scaler = StandardScaler().fit(X_dummy)

    # 3. Save Ensemble (Using the updated 'models' and 'threshold' keys)
    ensemble_data = {
        "models": base_models,
        "threshold": 0.35,
        "scaler": scaler,
        "strategy": "cost_sensitive",
        "audit": {"drift_score": 0.01, "status": "healthy"},
        "integrity_hash": "test_hash_001",
    }
    joblib.dump(ensemble_data, temp_models_dir / "cost_sensitive_ensemble_model.joblib")

    # 4. Save Metadata
    metadata = {
        "feature_columns": [f"feat_{i}" for i in range(768)]
        + [
            "age",
            "sex",
            "reported_cough_dur",
            "tb_prior",
            "hemoptysis",
            "weight_loss",
            "fever",
            "night_sweats",
        ]
        + [f"noise_{i}" for i in range(10)],
        "n_features": 786,
        "model_version": "1.0.0-test",
    }
    joblib.dump(metadata, temp_models_dir / "training_metadata.joblib")

    return temp_models_dir


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, mock_model_artifacts, test_config):
    """
    CRITICAL: This fixture overrides the config paths globally during tests.
    It forces the app to load the MOCK models we just created above.
    """
    # Patch the path in the config object
    monkeypatch.setattr(test_config.paths, "models_path", mock_model_artifacts)

    # Patch the get_config function itself so any new calls return our patched config
    import src.config

    monkeypatch.setattr(src.config, "get_config", lambda: test_config)


@pytest.fixture(scope="function")
def api_client():
    """FastAPI test client initialized with the patched environment."""
    from src.api.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def sample_prediction_form_data(sample_clinical_features) -> Dict:
    """Standardized form data for /predict endpoint."""
    data = sample_clinical_features.copy()
    data.update(
        {
            "generate_spectrogram": "true",
            "validate_quality": "false",  # Default to false for unit tests
        }
    )
    return data


# """
# Pytest configuration and fixtures - UPDATED FOR CURRENT STRUCTURE.

# Provides common fixtures for testing:
# - Sample data
# - Mock models
# - Test configurations
# """

# import io
# from pathlib import Path
# from typing import Dict

# import numpy as np
# import pytest
# from fastapi.testclient import TestClient

# from src.config import get_config


# @pytest.fixture(scope="session")
# def test_config():
#     """Get test configuration."""
#     return get_config()


# @pytest.fixture(scope="session")
# def sample_clinical_features() -> Dict:
#     """Sample clinical features for testing."""
#     return {
#         "age": 45,
#         "sex": "Male",
#         "reported_cough_dur": 21,
#         "tb_prior": "No",
#         "hemoptysis": "No",
#         "weight_loss": "Yes",
#         "fever": "Yes",
#         "night_sweats": "No",
#     }


# @pytest.fixture(scope="session")
# def sample_audio_features() -> np.ndarray:
#     """Sample audio features (768-dimensional)."""
#     return np.random.randn(768)


# @pytest.fixture(scope="session")
# def sample_full_features(sample_clinical_features, sample_audio_features) -> Dict:
#     """Sample full feature set (clinical + audio)."""
#     features = sample_clinical_features.copy()

#     # Add audio features
#     for i, value in enumerate(sample_audio_features):
#         features[f"feat_{i}"] = float(value)

#     # Add noise features (if used in training)
#     for i in range(10):
#         features[f"noise_{i}"] = 0.0

#     return features


# @pytest.fixture(scope="session")
# def sample_audio_bytes() -> bytes:
#     """Sample audio file as bytes."""
#     # Create a simple WAV file header (minimal)
#     sample_rate = 16000
#     duration = 1.0
#     num_samples = int(sample_rate * duration)

#     # Generate simple sine wave
#     frequency = 440.0  # A4 note
#     t = np.linspace(0, duration, num_samples)
#     audio_data = np.sin(2 * np.pi * frequency * t)

#     # Convert to 16-bit PCM
#     audio_data = (audio_data * 32767).astype(np.int16)

#     # Create WAV file in memory
#     import struct
#     import wave

#     buffer = io.BytesIO()
#     with wave.open(buffer, "wb") as wav_file:
#         wav_file.setnchannels(1)
#         wav_file.setsampwidth(2)
#         wav_file.setframerate(sample_rate)
#         wav_file.writeframes(audio_data.tobytes())

#     buffer.seek(0)
#     return buffer.read()


# @pytest.fixture(scope="function")
# def api_client():
#     """FastAPI test client."""
#     from src.api.main import app

#     return TestClient(app)


# @pytest.fixture(scope="session")
# def temp_dir(tmp_path_factory):
#     """Create temporary directory for tests."""
#     return tmp_path_factory.mktemp("test_data")


# @pytest.fixture(scope="session")
# def mock_model_path(temp_dir):
#     """
#     Create mock model files matching CURRENT ensemble structure.

#     UPDATED: Uses 'models' and 'threshold' keys (not 'base_models' and 'optimal_threshold')
#     UPDATED: Includes 'audit' dictionary
#     UPDATED: Includes 'scaler' in bundle
#     """
#     import joblib
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.preprocessing import StandardScaler

#     models_dir = temp_dir / "trained_models"
#     models_dir.mkdir(exist_ok=True)

#     # Create mock base models
#     base_models = {
#         "CatBoost": RandomForestClassifier(n_estimators=10, random_state=42),
#         "Meta-XGBoost": RandomForestClassifier(n_estimators=10, random_state=42),
#         "Meta-LightGBM": RandomForestClassifier(n_estimators=10, random_state=42),
#     }

#     # Fit with dummy data
#     X_dummy = np.random.randn(100, 786)
#     y_dummy = np.random.randint(0, 2, 100)

#     for model in base_models.values():
#         model.fit(X_dummy, y_dummy)

#     # Create scaler
#     scaler = StandardScaler()
#     scaler.fit(X_dummy)

#     # Save ensemble model with CURRENT structure
#     ensemble_data = {
#         # UPDATED: Use 'models' not 'base_models'
#         "models": base_models,
#         # UPDATED: Use 'threshold' not 'optimal_threshold'
#         "threshold": 0.32,
#         # UPDATED: Add 'scaler' to bundle
#         "scaler": scaler,
#         "strategy": "cost",
#         # UPDATED: Add 'audit' dictionary
#         "audit": {"psi": 0.0234, "uncertain_cases": 145, "drift_score": 0.12},
#         "model_weights": {"CatBoost": 0.4, "Meta-XGBoost": 0.35, "Meta-LightGBM": 0.25},
#         "integrity_hash": "mock_hash_12345",
#     }

#     ensemble_path = models_dir / "cost_sensitive_ensemble_model.joblib"
#     joblib.dump(ensemble_data, ensemble_path)

#     # Save scaler separately (for backward compatibility)
#     scaler_path = models_dir / "scaler.joblib"
#     joblib.dump(scaler, scaler_path)

#     # Save metadata with CURRENT structure
#     metadata = {
#         "best_model": "CatBoost",
#         "top_3_models": ["CatBoost", "Meta-XGBoost", "Meta-LightGBM"],
#         "feature_columns": [f"feat_{i}" for i in range(768)]
#         + [
#             "age",
#             "sex",
#             "reported_cough_dur",
#             "tb_prior",
#             "hemoptysis",
#             "weight_loss",
#             "fever",
#             "night_sweats",
#         ]
#         + [f"noise_{i}" for i in range(10)],
#         "clinical_features": [
#             "age",
#             "sex",
#             "reported_cough_dur",
#             "tb_prior",
#             "hemoptysis",
#             "weight_loss",
#             "fever",
#             "night_sweats",
#         ],
#         "audio_features": [f"feat_{i}" for i in range(768)],
#         "noise_features": [f"noise_{i}" for i in range(10)],
#         "n_features": 786,
#         "binary_features": [
#             "tb_prior",
#             "hemoptysis",
#             "weight_loss",
#             "fever",
#             "night_sweats",
#         ],
#     }

#     metadata_path = models_dir / "training_metadata.joblib"
#     joblib.dump(metadata, metadata_path)

#     return models_dir


# @pytest.fixture
# def sample_dataframe():
#     """Sample pandas DataFrame for testing."""
#     import pandas as pd

#     data = {
#         "age": [45, 32, 58, 41],
#         "sex": ["Male", "Female", "Male", "Female"],
#         "reported_cough_dur": [21, 14, 30, 7],
#         "tb_prior": ["No", "No", "Yes", "No"],
#         "hemoptysis": ["No", "Yes", "No", "No"],
#         "weight_loss": ["Yes", "Yes", "No", "Yes"],
#         "fever": ["Yes", "No", "Yes", "Yes"],
#         "night_sweats": ["No", "Yes", "Yes", "No"],
#         "tb_status": [1, 0, 1, 0],
#     }

#     # Add audio features
#     for i in range(768):
#         data[f"feat_{i}"] = np.random.randn(4)

#     return pd.DataFrame(data)


# @pytest.fixture(scope="session")
# def sample_prediction_form_data(sample_clinical_features) -> Dict:
#     """
#     Sample form data for prediction endpoint.
#     Matches CURRENT API structure with validate_quality parameter.
#     """
#     return {
#         "age": sample_clinical_features["age"],
#         "sex": sample_clinical_features["sex"],
#         "reported_cough_dur": sample_clinical_features["reported_cough_dur"],
#         "tb_prior": sample_clinical_features["tb_prior"],
#         "hemoptysis": sample_clinical_features["hemoptysis"],
#         "weight_loss": sample_clinical_features["weight_loss"],
#         "fever": sample_clinical_features["fever"],
#         "night_sweats": sample_clinical_features["night_sweats"],
#         "generate_spectrogram": "true",
#         "validate_quality": "false",  # UPDATED: Added this parameter
#     }
