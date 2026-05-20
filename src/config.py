"""
Configuration module for TBFusionAI project.
Contains all configurable parameters with cost-sensitive techniques.

Note: PathConfig is a plain Python class (not BaseSettings) to prevent Pydantic
from wrapping Path objects in OptionInfo when used with Typer.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic_settings import BaseSettings


class PathConfig:
    """
    Path configurations — plain Python class, NOT BaseSettings.
    Env vars with PATH_* prefix are intentionally not read here;
    set APP_HOME instead.
    """

    def __init__(self):
        self.project_name = "TBFusionAI"
        self.project_root = Path(__file__).parent.parent.absolute()
        self.artifacts_path = self.project_root / "artifacts"
        self.dataset_path = self.artifacts_path / "dataset"
        self.models_path = self.artifacts_path / "trained_models"
        self.preprocessed_path = self.artifacts_path / "preprocessed_data"
        self.labeled_data_path = self.artifacts_path / "labeled_data"
        self.reports_path = self.artifacts_path / "reports"
        self.participants_path = self.artifacts_path / "participants"


class DataIngestionConfig(BaseSettings):
    """Configuration for data ingestion pipeline."""

    repo_id: str = "AHFIDAILabs/coda_tb_dataset"
    dataset_filename: str = "dataset.zip"
    raw_data_subdir: str = "raw_data"
    meta_data_subdir: str = "meta_data"
    longitudinal_data_dir: str = "longitudinal_data"
    solicited_data_dir: str = "solicited_data"

    model_config = {"env_prefix": "INGESTION_"}


class AudioExtractionConfig(BaseSettings):
    """Configuration for audio feature extraction."""

    model_name: str = "facebook/wav2vec2-base-960h"
    sample_rate: int = 16000
    max_retries: int = 3
    retry_delay: int = 10
    batch_size: int = 32
    feature_dim: int = 768

    model_config = {
        "env_prefix": "AUDIO_EXTRACT_",
        "protected_namespaces": ("settings_",),
    }


class AudioPreprocessingConfig(BaseSettings):
    """Configuration for audio preprocessing and segmentation."""

    lowcut: float = 300.0
    highcut: float = 3400.0
    filter_order: int = 5

    noise_prop_decrease: float = 0.5

    energy_threshold_ratio: float = 0.001
    min_segment_length_sec: float = 0.1
    silence_gap_sec: float = 0.15
    librosa_top_db: int = 35
    librosa_min_len_sec: float = 0.25

    rms_min: float = 0.005
    snr_min: float = 3.0
    zcr_max: float = 0.3
    flatness_max: float = 0.6
    duration_min: float = 0.3
    duration_max: float = 0.5

    n_mels: int = 128
    spectrogram_dpi: int = 100
    spectrogram_figsize: tuple = (3, 3)

    model_config = {"env_prefix": "AUDIO_PREPROCESS_"}


class MetadataConfig(BaseSettings):
    """Configuration for metadata matching and feature integration."""

    clinical_features: List[str] = [
        "sex",
        "age",
        "reported_cough_dur",
        "tb_prior",
        "hemoptysis",
        "weight_loss",
        "fever",
        "night_sweats",
    ]

    binary_features: List[str] = [
        "tb_prior",
        "hemoptysis",
        "weight_loss",
        "fever",
        "night_sweats",
    ]

    target_column: str = "tb_status"

    model_config = {"env_prefix": "METADATA_"}


class CTGANConfig(BaseSettings):
    """Configuration for CTGAN synthetic data generation — disabled (CPU too slow)."""

    use_ctgan: bool = False
    epochs: int = 100
    batch_size: int = 500
    log_frequency: bool = True
    verbose: bool = True
    cuda: bool = False

    model_config = {"env_prefix": "CTGAN_"}


class ModelTrainingConfig(BaseSettings):
    """Configuration for model training pipeline."""

    test_size: float = 0.2
    random_state: int = 42

    cv_folds: int = 5
    cv_shuffle: bool = True

    noise_fraction: float = 0.02
    num_noise_features: int = 10

    primary_metric: str = "recall"

    selection_weights: Dict[str, int] = {
        "false_negatives": 100,
        "recall": 30,
        "f1": 10,
        "accuracy": 5,
        "roc_auc": 5,
    }

    use_calibration: bool = True
    calibration_method: str = "isotonic"
    calibration_cv: int = 5

    use_meta_cost: bool = True
    meta_cost_iterations: int = 3

    randomized_search_iter: int = 8
    randomized_search_cv: int = 3

    logistic_max_iter: int = 1000
    logistic_class_weight: str = "balanced"
    logistic_param_grid: dict = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["lbfgs", "liblinear"],
    }

    rf_n_estimators: int = 100
    rf_class_weight: str = "balanced"
    rf_param_grid: dict = {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 6, 8, 10],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt", "log2"],
    }

    xgb_n_estimators: int = 150
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 4
    xgb_param_grid: dict = {
        "n_estimators": [100, 150],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    lgb_n_estimators: int = 150
    lgb_learning_rate: float = 0.1
    lgb_max_depth: int = 4
    lgb_param_grid: dict = {
        "n_estimators": [100, 150],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [25, 31, 40],
    }

    catboost_iterations: int = 200
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 5
    catboost_param_grid: dict = {
        "iterations": [150, 200, 250],
        "depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bylevel": [0.7, 0.8, 0.9],
    }

    mlp_activation: str = "relu"
    mlp_use_asymmetric_loss: bool = True
    mlp_fn_cost: int = 100
    mlp_fp_cost: int = 10
    mlp_param_grid: dict = {
        "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [500, 1000],
    }

    model_config = {"env_prefix": "TRAIN_"}


class EnsembleConfig(BaseSettings):
    """Configuration for ensemble strategies."""

    model_weights: Dict[str, float] = {
        "CatBoost": 0.4,
        "XGBoost": 0.35,
        "LightGBM": 0.25,
    }

    hard_voting_threshold: int = 2
    soft_voting_threshold: float = 0.5

    confidence_threshold: float = 0.12
    uncertainty_margin: float = 0.1

    cost_fn: int = 100
    cost_fp: int = 10
    threshold_search_start: float = 0.2
    threshold_search_end: float = 0.8
    threshold_search_step: float = 0.01

    use_fbeta_threshold: bool = True
    fbeta_values: List[float] = [2.0, 3.0]

    use_youden_threshold: bool = True

    threshold_selection: str = "auto"

    model_config = {"env_prefix": "ENSEMBLE_", "protected_namespaces": ("settings_",)}


class InferenceConfig(BaseSettings):
    """Configuration for model inference."""

    high_confidence_threshold: float = 0.15
    uncertainty_threshold: float = 0.05
    top_features_display: int = 5
    batch_inference_size: int = 1000

    use_reject_option: bool = True
    reject_on_uncertainty: bool = True
    reject_on_disagreement: bool = True
    reject_on_boundary: bool = True

    model_config = {"env_prefix": "INFERENCE_"}


class APIConfig(BaseSettings):
    """Configuration for FastAPI application."""

    app_title: str = "TBFusionAI API"
    app_description: str = "TB Detection using AI-powered Cough Analysis"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Wildcard origins require credentials=False per the CORS spec.
    # Replace "*" with an explicit origin list if cookies/auth headers are needed.
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = False
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]

    max_upload_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_audio_formats: List[str] = [".wav", ".mp3", ".ogg", ".webm"]

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4

    model_config = {"env_prefix": "API_"}


class Config:
    """Main configuration class that aggregates all sub-configs."""

    def __init__(self):
        self.paths = PathConfig()
        self.data_ingestion = DataIngestionConfig()
        self.audio_extraction = AudioExtractionConfig()
        self.audio_preprocessing = AudioPreprocessingConfig()
        self.metadata = MetadataConfig()
        self.ctgan = CTGANConfig()
        self.model_training = ModelTrainingConfig()
        self.ensemble = EnsembleConfig()
        self.inference = InferenceConfig()
        self.api = APIConfig()
        self._create_directories()

    def _create_directories(self) -> None:
        for directory in [
            self.paths.artifacts_path,
            self.paths.dataset_path,
            self.paths.models_path,
            self.paths.preprocessed_path,
            self.paths.labeled_data_path,
            self.paths.reports_path,
            self.paths.participants_path,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_config() -> Config:
    """Return the cached singleton Config instance."""
    return Config()
