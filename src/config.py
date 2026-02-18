"""
Configuration module for TBFusionAI project.
Contains all configurable parameters with new cost-sensitive techniques.

CRITICAL FIX: PathConfig is now a plain Python class (not BaseSettings)
This completely prevents Pydantic from wrapping Path objects in OptionInfo.
"""

from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class PathConfig:
    """
    Path configurations - Plain Python class (NO Pydantic BaseSettings).
    
    This prevents OptionInfo wrapping that causes path-related errors.
    """
    
    def __init__(self):
        """Initialize all paths as actual Path objects."""
        self.project_name = "TBFusionAI"
        self.project_root = Path(__file__).parent.parent.absolute()
        self.artifacts_path = self.project_root / "artifacts"
        self.dataset_path = self.artifacts_path / "dataset"
        self.models_path = self.artifacts_path / "trained_models"
        self.preprocessed_path = self.artifacts_path / "preprocessed_data"
        self.labeled_data_path = self.artifacts_path / "labeled_data"
        self.reports_path = self.artifacts_path / "reports"


class DataIngestionConfig(BaseSettings):
    """Configuration for data ingestion pipeline."""
    
    repo_id: str = "AHFIDAILabs/coda_tb_dataset"
    dataset_filename: str = "dataset.zip"
    raw_data_subdir: str = "raw_data"
    meta_data_subdir: str = "meta_data"
    longitudinal_data_dir: str = "longitudinal_data"
    solicited_data_dir: str = "solicited_data"
    
    model_config = {
        "env_prefix": "INGESTION_"
    }


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
        "protected_namespaces": ('settings_',)
    }


class AudioPreprocessingConfig(BaseSettings):
    """Configuration for audio preprocessing and segmentation."""
    
    # Bandpass filter parameters
    lowcut: float = 300.0
    highcut: float = 3400.0
    filter_order: int = 5
    
    # Noise reduction
    noise_prop_decrease: float = 0.5
    
    # Segmentation parameters
    energy_threshold_ratio: float = 0.001
    min_segment_length_sec: float = 0.1
    silence_gap_sec: float = 0.15
    librosa_top_db: int = 35
    librosa_min_len_sec: float = 0.25
    
    # Quality filtering thresholds
    rms_min: float = 0.005
    snr_min: float = 3.0
    zcr_max: float = 0.3
    flatness_max: float = 0.6
    duration_min: float = 0.3
    duration_max: float = 0.5
    
    # Mel spectrogram parameters
    n_mels: int = 128
    spectrogram_dpi: int = 100
    spectrogram_figsize: tuple = (3, 3)
    
    model_config = {
        "env_prefix": "AUDIO_PREPROCESS_"
    }


class MetadataConfig(BaseSettings):
    """Configuration for metadata matching and feature integration."""
    
    clinical_features: List[str] = [
        'sex', 'age', 'reported_cough_dur', 'tb_prior',
        'hemoptysis', 'weight_loss', 'fever', 'night_sweats'
    ]
    
    binary_features: List[str] = [
        'tb_prior', 'hemoptysis', 'weight_loss', 'fever', 'night_sweats'
    ]
    
    target_column: str = 'tb_status'
    
    model_config = {
        "env_prefix": "METADATA_"
    }


class CTGANConfig(BaseSettings):
    """Configuration for CTGAN synthetic data generation - DISABLED."""
    
    # CTGAN disabled - takes 10+ days on CPU
    # Using cost-sensitive learning instead
    use_ctgan: bool = False
    epochs: int = 100
    batch_size: int = 500
    log_frequency: bool = True
    verbose: bool = True
    cuda: bool = False
    
    model_config = {
        "env_prefix": "CTGAN_"
    }


class ModelTrainingConfig(BaseSettings):
    """Configuration for model training pipeline - ENHANCED."""
    
    # Data split
    test_size: float = 0.2
    random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    cv_shuffle: bool = True
    
    # Label noise simulation
    noise_fraction: float = 0.02
    num_noise_features: int = 10
    
    # Scoring metrics
    primary_metric: str = "recall"  # Minimize FN
    
    # Selection weights (emphasis on FN reduction)
    selection_weights: Dict[str, int] = {
        'false_negatives': 100,  # PRIMARY
        'recall': 30,            # SECONDARY
        'f1': 10,                # TERTIARY
        'accuracy': 5,           # Supporting
        'roc_auc': 5             # Supporting
    }
    
    # NEW: Probability Calibration
    use_calibration: bool = True
    calibration_method: str = "isotonic"  # 'isotonic' or 'sigmoid'
    calibration_cv: int = 5
    
    # NEW: Meta-Cost Ensemble
    use_meta_cost: bool = True
    meta_cost_iterations: int = 3  # Number of meta-cost models
    
    # Hyperparameter search
    randomized_search_iter: int = 8
    randomized_search_cv: int = 3
    
    # Logistic Regression
    logistic_max_iter: int = 1000
    logistic_class_weight: str = "balanced"
    logistic_param_grid: dict = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['lbfgs', 'liblinear']
    }
    
    # Random Forest
    rf_n_estimators: int = 100
    rf_class_weight: str = "balanced"
    rf_param_grid: dict = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 6, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # XGBoost
    xgb_n_estimators: int = 150
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 4
    xgb_param_grid: dict = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # LightGBM
    lgb_n_estimators: int = 150
    lgb_learning_rate: float = 0.1
    lgb_max_depth: int = 4
    lgb_param_grid: dict = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [25, 31, 40]
    }
    
    # CatBoost
    catboost_iterations: int = 200
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 5
    catboost_param_grid: dict = {
        'iterations': [150, 200, 250],
        'depth': [4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bylevel': [0.7, 0.8, 0.9]
    }
    
    # MLP - NEW: Asymmetric Loss Support
    mlp_activation: str = 'relu'
    mlp_use_asymmetric_loss: bool = True  # NEW: Use cost-sensitive loss
    mlp_fn_cost: int = 100  # Cost for false negatives
    mlp_fp_cost: int = 10   # Cost for false positives
    mlp_param_grid: dict = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500, 1000]
    }
    
    model_config = {
        "env_prefix": "TRAIN_"
    }


class EnsembleConfig(BaseSettings):
    """Configuration for ensemble strategies - ENHANCED."""
    
    # Model weights (based on FN performance)
    model_weights: Dict[str, float] = {
        'CatBoost': 0.4,   # Best performer
        'XGBoost': 0.35,   # Second best
        'LightGBM': 0.25   # Third
    }
    
    # Voting strategies
    hard_voting_threshold: int = 2  # 2 out of 3
    soft_voting_threshold: float = 0.5
    
    # Confidence thresholding - ENHANCED
    confidence_threshold: float = 0.12
    uncertainty_margin: float = 0.1  # NEW: Distance from decision boundary for reject option
    
    # Cost-sensitive parameters
    cost_fn: int = 100  # Cost of False Negative
    cost_fp: int = 10   # Cost of False Positive
    threshold_search_start: float = 0.2
    threshold_search_end: float = 0.8
    threshold_search_step: float = 0.01
    
    # NEW: F-beta Score Optimization
    use_fbeta_threshold: bool = True
    fbeta_values: List[float] = [2.0, 3.0]  # Test F2 and F3 scores
    
    # NEW: Youden's Index
    use_youden_threshold: bool = True
    
    # NEW: Threshold Selection Strategy
    threshold_selection: str = "auto"  # 'auto', 'cost', 'fbeta', 'youden', or 'ensemble'
    
    model_config = {
        "env_prefix": "ENSEMBLE_",
        "protected_namespaces": ('settings_',)
    }


class InferenceConfig(BaseSettings):
    """Configuration for model inference - ENHANCED."""
    
    high_confidence_threshold: float = 0.15
    uncertainty_threshold: float = 0.05
    top_features_display: int = 5
    batch_inference_size: int = 1000
    
    # NEW: Enhanced reject option
    use_reject_option: bool = True
    reject_on_uncertainty: bool = True
    reject_on_disagreement: bool = True
    reject_on_boundary: bool = True  # NEW: Reject if near decision boundary
    
    model_config = {
        "env_prefix": "INFERENCE_"
    }


class APIConfig(BaseSettings):
    """Configuration for FastAPI application."""
    
    app_title: str = "TBFusionAI API"
    app_description: str = "TB Detection using AI-powered Cough Analysis"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # File upload
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_audio_formats: List[str] = [".wav", ".mp3", ".ogg", ".webm"]
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    
    model_config = {
        "env_prefix": "API_"
    }


class Config:
    """Main configuration class that aggregates all configs."""
    
    def __init__(self):
        self.paths = PathConfig()  # Plain class, no OptionInfo wrapping!
        self.data_ingestion = DataIngestionConfig()
        self.audio_extraction = AudioExtractionConfig()
        self.audio_preprocessing = AudioPreprocessingConfig()
        self.metadata = MetadataConfig()
        self.ctgan = CTGANConfig()
        self.model_training = ModelTrainingConfig()
        self.ensemble = EnsembleConfig()
        self.inference = InferenceConfig()
        self.api = APIConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.paths.artifacts_path,
            self.paths.dataset_path,
            self.paths.models_path,
            self.paths.preprocessed_path,
            self.paths.labeled_data_path,
            self.paths.reports_path,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_config() -> Config:
    """
    Get cached configuration instance.
    
    Returns:
        Config: Cached configuration object
    """
    return Config()


# """
# Configuration module for TBFusionAI project - FIXED.
# Contains all configurable parameters with new cost-sensitive techniques.

# FIX: Changed PathConfig to use @property instead of Field(default_factory=...)
# This prevents Pydantic from wrapping Path objects in OptionInfo.
# """

# from pathlib import Path
# from typing import Dict, List, Optional
# from pydantic_settings import BaseSettings
# from functools import lru_cache


# class PathConfig:
#     """Path configurations for the project."""

#     def __init__(self):
#         self._project_root: Optional[Path] = None

#     @property
#     def project_root(self) -> Path:
#         if self._project_root is None:
#             self._project_root = Path(__file__).parent.parent.absolute()
#         return self._project_root

#     @property
#     def artifacts_path(self) -> Path:
#         return self.project_root / "artifacts"

#     @property
#     def dataset_path(self) -> Path:
#         return self.artifacts_path / "dataset"

#     @property
#     def models_path(self) -> Path:
#         return self.artifacts_path / "trained_models"

#     @property
#     def preprocessed_path(self) -> Path:
#         return self.artifacts_path / "preprocessed_data"

#     @property
#     def labeled_data_path(self) -> Path:
#         return self.artifacts_path / "labeled_data"

#     @property
#     def reports_path(self) -> Path:
#         return self.artifacts_path / "reports"


# class DataIngestionConfig(BaseSettings):
#     """Configuration for data ingestion pipeline."""
    
#     repo_id: str = "AHFIDAILabs/coda_tb_dataset"
#     dataset_filename: str = "dataset.zip"
#     raw_data_subdir: str = "raw_data"
#     meta_data_subdir: str = "meta_data"
#     longitudinal_data_dir: str = "longitudinal_data"
#     solicited_data_dir: str = "solicited_data"
    
#     model_config = {
#         "env_prefix": "INGESTION_"
#     }


# class AudioExtractionConfig(BaseSettings):
#     """Configuration for audio feature extraction."""
    
#     model_name: str = "facebook/wav2vec2-base-960h"
#     sample_rate: int = 16000
#     max_retries: int = 3
#     retry_delay: int = 10
#     batch_size: int = 32
#     feature_dim: int = 768
    
#     model_config = {
#         "env_prefix": "AUDIO_EXTRACT_",
#         "protected_namespaces": ('settings_',)
#     }


# class AudioPreprocessingConfig(BaseSettings):
#     """Configuration for audio preprocessing and segmentation."""
    
#     # Bandpass filter parameters
#     lowcut: float = 300.0
#     highcut: float = 3400.0
#     filter_order: int = 5
    
#     # Noise reduction
#     noise_prop_decrease: float = 0.5
    
#     # Segmentation parameters
#     energy_threshold_ratio: float = 0.001
#     min_segment_length_sec: float = 0.1
#     silence_gap_sec: float = 0.15
#     librosa_top_db: int = 35
#     librosa_min_len_sec: float = 0.25
    
#     # Quality filtering thresholds
#     rms_min: float = 0.005
#     snr_min: float = 3.0
#     zcr_max: float = 0.3
#     flatness_max: float = 0.6
#     duration_min: float = 0.3
#     duration_max: float = 0.5
    
#     # Mel spectrogram parameters
#     n_mels: int = 128
#     spectrogram_dpi: int = 100
#     spectrogram_figsize: tuple = (3, 3)
    
#     model_config = {
#         "env_prefix": "AUDIO_PREPROCESS_"
#     }


# class MetadataConfig(BaseSettings):
#     """Configuration for metadata matching and feature integration."""
    
#     clinical_features: List[str] = [
#         'sex', 'age', 'reported_cough_dur', 'tb_prior',
#         'hemoptysis', 'weight_loss', 'fever', 'night_sweats'
#     ]
    
#     binary_features: List[str] = [
#         'tb_prior', 'hemoptysis', 'weight_loss', 'fever', 'night_sweats'
#     ]
    
#     target_column: str = 'tb_status'
    
#     model_config = {
#         "env_prefix": "METADATA_"
#     }


# class CTGANConfig(BaseSettings):
#     """Configuration for CTGAN synthetic data generation - DISABLED."""
    
#     # CTGAN disabled - takes 10+ days on CPU
#     # Using cost-sensitive learning instead
#     use_ctgan: bool = False
#     epochs: int = 100
#     batch_size: int = 500
#     log_frequency: bool = True
#     verbose: bool = True
#     cuda: bool = False
    
#     model_config = {
#         "env_prefix": "CTGAN_"
#     }


# class ModelTrainingConfig(BaseSettings):
#     """Configuration for model training pipeline - ENHANCED."""
    
#     # Data split
#     test_size: float = 0.2
#     random_state: int = 42
    
#     # Cross-validation
#     cv_folds: int = 5
#     cv_shuffle: bool = True
    
#     # Label noise simulation
#     noise_fraction: float = 0.02
#     num_noise_features: int = 10
    
#     # Scoring metrics
#     primary_metric: str = "recall"  # Minimize FN
    
#     # Selection weights (emphasis on FN reduction)
#     selection_weights: Dict[str, int] = {
#         'false_negatives': 100,  # PRIMARY
#         'recall': 30,            # SECONDARY
#         'f1': 10,                # TERTIARY
#         'accuracy': 5,           # Supporting
#         'roc_auc': 5             # Supporting
#     }
    
#     # NEW: Probability Calibration
#     use_calibration: bool = True
#     calibration_method: str = "isotonic"  # 'isotonic' or 'sigmoid'
#     calibration_cv: int = 5
    
#     # NEW: Meta-Cost Ensemble
#     use_meta_cost: bool = True
#     meta_cost_iterations: int = 3  # Number of meta-cost models
    
#     # Hyperparameter search
#     randomized_search_iter: int = 8
#     randomized_search_cv: int = 3
    
#     # Logistic Regression
#     logistic_max_iter: int = 1000
#     logistic_class_weight: str = "balanced"
#     logistic_param_grid: dict = {
#         'C': [0.001, 0.01, 0.1, 1, 10],
#         'penalty': ['l1', 'l2'],
#         'solver': ['lbfgs', 'liblinear']
#     }
    
#     # Random Forest
#     rf_n_estimators: int = 100
#     rf_class_weight: str = "balanced"
#     rf_param_grid: dict = {
#         'n_estimators': [100, 150, 200],
#         'max_depth': [5, 6, 8, 10],
#         'min_samples_split': [5, 10],
#         'min_samples_leaf': [2, 4],
#         'max_features': ['sqrt', 'log2']
#     }
    
#     # XGBoost
#     xgb_n_estimators: int = 150
#     xgb_learning_rate: float = 0.1
#     xgb_max_depth: int = 4
#     xgb_param_grid: dict = {
#         'n_estimators': [100, 150],
#         'max_depth': [3, 4, 5],
#         'learning_rate': [0.05, 0.1],
#         'subsample': [0.8, 1.0],
#         'colsample_bytree': [0.8, 1.0]
#     }
    
#     # LightGBM
#     lgb_n_estimators: int = 150
#     lgb_learning_rate: float = 0.1
#     lgb_max_depth: int = 4
#     lgb_param_grid: dict = {
#         'n_estimators': [100, 150],
#         'max_depth': [3, 4, 5],
#         'learning_rate': [0.05, 0.1],
#         'num_leaves': [25, 31, 40]
#     }
    
#     # CatBoost
#     catboost_iterations: int = 200
#     catboost_learning_rate: float = 0.05
#     catboost_depth: int = 5
#     catboost_param_grid: dict = {
#         'iterations': [150, 200, 250],
#         'depth': [4, 5, 6],
#         'learning_rate': [0.03, 0.05, 0.1],
#         'l2_leaf_reg': [1, 3, 5],
#         'subsample': [0.7, 0.8, 0.9],
#         'colsample_bylevel': [0.7, 0.8, 0.9]
#     }
    
#     # MLP - NEW: Asymmetric Loss Support
#     mlp_activation: str = 'relu'
#     mlp_use_asymmetric_loss: bool = True  # NEW: Use cost-sensitive loss
#     mlp_fn_cost: int = 100  # Cost for false negatives
#     mlp_fp_cost: int = 10   # Cost for false positives
#     mlp_param_grid: dict = {
#         'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
#         'alpha': [0.0001, 0.001, 0.01],
#         'learning_rate_init': [0.001, 0.01],
#         'max_iter': [500, 1000]
#     }
    
#     model_config = {
#         "env_prefix": "TRAIN_"
#     }


# class EnsembleConfig(BaseSettings):
#     """Configuration for ensemble strategies - ENHANCED."""
    
#     # Model weights (based on FN performance)
#     model_weights: Dict[str, float] = {
#         'CatBoost': 0.4,   # Best performer
#         'XGBoost': 0.35,   # Second best
#         'LightGBM': 0.25   # Third
#     }
    
#     # Voting strategies
#     hard_voting_threshold: int = 2  # 2 out of 3
#     soft_voting_threshold: float = 0.5
    
#     # Confidence thresholding - ENHANCED
#     confidence_threshold: float = 0.12
#     uncertainty_margin: float = 0.1  # NEW: Distance from decision boundary for reject option
    
#     # Cost-sensitive parameters
#     cost_fn: int = 100  # Cost of False Negative
#     cost_fp: int = 10   # Cost of False Positive
#     threshold_search_start: float = 0.2
#     threshold_search_end: float = 0.8
#     threshold_search_step: float = 0.01
    
#     # NEW: F-beta Score Optimization
#     use_fbeta_threshold: bool = True
#     fbeta_values: List[float] = [2.0, 3.0]  # Test F2 and F3 scores
    
#     # NEW: Youden's Index
#     use_youden_threshold: bool = True
    
#     # NEW: Threshold Selection Strategy
#     threshold_selection: str = "auto"  # 'auto', 'cost', 'fbeta', 'youden', or 'ensemble'
    
#     model_config = {
#         "env_prefix": "ENSEMBLE_",
#         "protected_namespaces": ('settings_',)
#     }


# class InferenceConfig(BaseSettings):
#     """Configuration for model inference - ENHANCED."""
    
#     high_confidence_threshold: float = 0.15
#     uncertainty_threshold: float = 0.05
#     top_features_display: int = 5
#     batch_inference_size: int = 1000
    
#     # NEW: Enhanced reject option
#     use_reject_option: bool = True
#     reject_on_uncertainty: bool = True
#     reject_on_disagreement: bool = True
#     reject_on_boundary: bool = True  # NEW: Reject if near decision boundary
    
#     model_config = {
#         "env_prefix": "INFERENCE_"
#     }


# class APIConfig(BaseSettings):
#     """Configuration for FastAPI application."""
    
#     app_title: str = "TBFusionAI API"
#     app_description: str = "TB Detection using AI-powered Cough Analysis"
#     app_version: str = "1.0.0"
#     api_prefix: str = "/api/v1"
    
#     # CORS settings
#     cors_origins: List[str] = ["*"]
#     cors_credentials: bool = True
#     cors_methods: List[str] = ["*"]
#     cors_headers: List[str] = ["*"]
    
#     # File upload
#     max_upload_size: int = 10 * 1024 * 1024  # 10MB
#     allowed_audio_formats: List[str] = [".wav", ".mp3", ".ogg", ".webm"]
    
#     # Server
#     host: str = "0.0.0.0"
#     port: int = 8000
#     reload: bool = False
#     workers: int = 4
    
#     model_config = {
#         "env_prefix": "API_"
#     }


# class Config:
#     """Main configuration class that aggregates all configs."""
    
#     def __init__(self):
#         self.paths = PathConfig()
#         self.data_ingestion = DataIngestionConfig()
#         self.audio_extraction = AudioExtractionConfig()
#         self.audio_preprocessing = AudioPreprocessingConfig()
#         self.metadata = MetadataConfig()
#         self.ctgan = CTGANConfig()
#         self.model_training = ModelTrainingConfig()
#         self.ensemble = EnsembleConfig()
#         self.inference = InferenceConfig()
#         self.api = APIConfig()
        
#         # Create necessary directories
#         self._create_directories()
    
#     def _create_directories(self) -> None:
#         """Create necessary directories if they don't exist."""
#         directories = [
#             self.paths.artifacts_path,
#             self.paths.dataset_path,
#             self.paths.models_path,
#             self.paths.preprocessed_path,
#             self.paths.labeled_data_path,
#             self.paths.reports_path,
#         ]
        
#         for directory in directories:
#             directory.mkdir(parents=True, exist_ok=True)


# @lru_cache()
# def get_config() -> Config:
#     """
#     Get cached configuration instance.
    
#     Returns:
#         Config: Cached configuration object
#     """
#     return Config()

# """
# Configuration module for TBFusionAI project.
# Contains all configurable parameters with new cost-sensitive techniques.
# """

# from pathlib import Path
# from typing import Dict, List, Optional
# from pydantic import Field
# from pydantic_settings import BaseSettings
# from functools import lru_cache


# class PathConfig(BaseSettings):
#     """Path configurations for the project."""
    
#     project_name: str = "TBFusionAI"
#     project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.absolute())
#     artifacts_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "artifacts")
#     dataset_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "artifacts" / "dataset")
#     models_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "artifacts" / "trained_models")
#     preprocessed_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "artifacts" / "preprocessed_data")
#     labeled_data_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "artifacts" / "labeled_data")
#     reports_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "artifacts" / "reports")
    
#     model_config = {
#         "env_prefix": "PATH_"
#     }


# class DataIngestionConfig(BaseSettings):
#     """Configuration for data ingestion pipeline."""
    
#     repo_id: str = "AHFIDAILabs/coda_tb_dataset"
#     dataset_filename: str = "dataset.zip"
#     raw_data_subdir: str = "raw_data"
#     meta_data_subdir: str = "meta_data"
#     longitudinal_data_dir: str = "longitudinal_data"
#     solicited_data_dir: str = "solicited_data"
    
#     model_config = {
#         "env_prefix": "INGESTION_"
#     }


# class AudioExtractionConfig(BaseSettings):
#     """Configuration for audio feature extraction."""
    
#     model_name: str = "facebook/wav2vec2-base-960h"
#     sample_rate: int = 16000
#     max_retries: int = 3
#     retry_delay: int = 10
#     batch_size: int = 32
#     feature_dim: int = 768
    
#     model_config = {
#         "env_prefix": "AUDIO_EXTRACT_",
#         "protected_namespaces": ('settings_',)
#     }


# class AudioPreprocessingConfig(BaseSettings):
#     """Configuration for audio preprocessing and segmentation."""
    
#     # Bandpass filter parameters
#     lowcut: float = 300.0
#     highcut: float = 3400.0
#     filter_order: int = 5
    
#     # Noise reduction
#     noise_prop_decrease: float = 0.5
    
#     # Segmentation parameters
#     energy_threshold_ratio: float = 0.001
#     min_segment_length_sec: float = 0.1
#     silence_gap_sec: float = 0.15
#     librosa_top_db: int = 35
#     librosa_min_len_sec: float = 0.25
    
#     # Quality filtering thresholds
#     rms_min: float = 0.005
#     snr_min: float = 3.0
#     zcr_max: float = 0.3
#     flatness_max: float = 0.6
#     duration_min: float = 0.3
#     duration_max: float = 0.5
    
#     # Mel spectrogram parameters
#     n_mels: int = 128
#     spectrogram_dpi: int = 100
#     spectrogram_figsize: tuple = (3, 3)
    
#     model_config = {
#         "env_prefix": "AUDIO_PREPROCESS_"
#     }


# class MetadataConfig(BaseSettings):
#     """Configuration for metadata matching and feature integration."""
    
#     clinical_features: List[str] = [
#         'sex', 'age', 'reported_cough_dur', 'tb_prior',
#         'hemoptysis', 'weight_loss', 'fever', 'night_sweats'
#     ]
    
#     binary_features: List[str] = [
#         'tb_prior', 'hemoptysis', 'weight_loss', 'fever', 'night_sweats'
#     ]
    
#     target_column: str = 'tb_status'
    
#     model_config = {
#         "env_prefix": "METADATA_"
#     }


# class CTGANConfig(BaseSettings):
#     """Configuration for CTGAN synthetic data generation - DISABLED."""
    
#     # CTGAN disabled - takes 10+ days on CPU
#     # Using cost-sensitive learning instead
#     use_ctgan: bool = False
#     epochs: int = 100
#     batch_size: int = 500
#     log_frequency: bool = True
#     verbose: bool = True
#     cuda: bool = False
    
#     model_config = {
#         "env_prefix": "CTGAN_"
#     }


# class ModelTrainingConfig(BaseSettings):
#     """Configuration for model training pipeline - ENHANCED."""
    
#     # Data split
#     test_size: float = 0.2
#     random_state: int = 42
    
#     # Cross-validation
#     cv_folds: int = 5
#     cv_shuffle: bool = True
    
#     # Label noise simulation
#     noise_fraction: float = 0.02
#     num_noise_features: int = 10
    
#     # Scoring metrics
#     primary_metric: str = "recall"  # Minimize FN
    
#     # Selection weights (emphasis on FN reduction)
#     selection_weights: Dict[str, int] = {
#         'false_negatives': 100,  # PRIMARY
#         'recall': 30,            # SECONDARY
#         'f1': 10,                # TERTIARY
#         'accuracy': 5,           # Supporting
#         'roc_auc': 5             # Supporting
#     }
    
#     # NEW: Probability Calibration
#     use_calibration: bool = True
#     calibration_method: str = "isotonic"  # 'isotonic' or 'sigmoid'
#     calibration_cv: int = 5
    
#     # NEW: Meta-Cost Ensemble
#     use_meta_cost: bool = True
#     meta_cost_iterations: int = 3  # Number of meta-cost models
    
#     # Hyperparameter search
#     randomized_search_iter: int = 8
#     randomized_search_cv: int = 3
    
#     # Logistic Regression
#     logistic_max_iter: int = 1000
#     logistic_class_weight: str = "balanced"
#     logistic_param_grid: Dict = {
#         'C': [0.001, 0.01, 0.1, 1, 10],
#         'penalty': ['l1', 'l2'],
#         'solver': ['lbfgs', 'liblinear']
#     }
    
#     # Random Forest
#     rf_n_estimators: int = 100
#     rf_class_weight: str = "balanced"
#     rf_param_grid: Dict = {
#         'n_estimators': [100, 150, 200],
#         'max_depth': [5, 6, 8, 10],
#         'min_samples_split': [5, 10],
#         'min_samples_leaf': [2, 4],
#         'max_features': ['sqrt', 'log2']
#     }
    
#     # XGBoost
#     xgb_n_estimators: int = 150
#     xgb_learning_rate: float = 0.1
#     xgb_max_depth: int = 4
#     xgb_param_grid: Dict = {
#         'n_estimators': [100, 150],
#         'max_depth': [3, 4, 5],
#         'learning_rate': [0.05, 0.1],
#         'subsample': [0.8, 1.0],
#         'colsample_bytree': [0.8, 1.0]
#     }
    
#     # LightGBM
#     lgb_n_estimators: int = 150
#     lgb_learning_rate: float = 0.1
#     lgb_max_depth: int = 4
#     lgb_param_grid: Dict = {
#         'n_estimators': [100, 150],
#         'max_depth': [3, 4, 5],
#         'learning_rate': [0.05, 0.1],
#         'num_leaves': [25, 31, 40]
#     }
    
#     # CatBoost
#     catboost_iterations: int = 200
#     catboost_learning_rate: float = 0.05
#     catboost_depth: int = 5
#     catboost_param_grid: Dict = {
#         'iterations': [150, 200, 250],
#         'depth': [4, 5, 6],
#         'learning_rate': [0.03, 0.05, 0.1],
#         'l2_leaf_reg': [1, 3, 5],
#         'subsample': [0.7, 0.8, 0.9],
#         'colsample_bylevel': [0.7, 0.8, 0.9]
#     }
    
#     # MLP - NEW: Asymmetric Loss Support
#     mlp_activation: str = 'relu'
#     mlp_use_asymmetric_loss: bool = True  # NEW: Use cost-sensitive loss
#     mlp_fn_cost: int = 100  # Cost for false negatives
#     mlp_fp_cost: int = 10   # Cost for false positives
#     mlp_param_grid: Dict = {
#         'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
#         'alpha': [0.0001, 0.001, 0.01],
#         'learning_rate_init': [0.001, 0.01],
#         'max_iter': [500, 1000]
#     }
    
#     model_config = {
#         "env_prefix": "TRAIN_"
#     }


# class EnsembleConfig(BaseSettings):
#     """Configuration for ensemble strategies - ENHANCED."""
    
#     # Model weights (based on FN performance)
#     model_weights: Dict[str, float] = {
#         'CatBoost': 0.4,   # Best performer
#         'XGBoost': 0.35,   # Second best
#         'LightGBM': 0.25   # Third
#     }
    
#     # Voting strategies
#     hard_voting_threshold: int = 2  # 2 out of 3
#     soft_voting_threshold: float = 0.5
    
#     # Confidence thresholding - ENHANCED
#     confidence_threshold: float = 0.12
#     uncertainty_margin: float = 0.1  # NEW: Distance from decision boundary for reject option
    
#     # Cost-sensitive parameters
#     cost_fn: int = 100  # Cost of False Negative
#     cost_fp: int = 10   # Cost of False Positive
#     threshold_search_start: float = 0.2
#     threshold_search_end: float = 0.8
#     threshold_search_step: float = 0.01
    
#     # NEW: F-beta Score Optimization
#     use_fbeta_threshold: bool = True
#     fbeta_values: List[float] = [2.0, 3.0]  # Test F2 and F3 scores
    
#     # NEW: Youden's Index
#     use_youden_threshold: bool = True
    
#     # NEW: Threshold Selection Strategy
#     threshold_selection: str = "auto"  # 'auto', 'cost', 'fbeta', 'youden', or 'ensemble'
    
#     model_config = {
#         "env_prefix": "ENSEMBLE_",
#         "protected_namespaces": ('settings_',)
#     }


# class InferenceConfig(BaseSettings):
#     """Configuration for model inference - ENHANCED."""
    
#     high_confidence_threshold: float = 0.15
#     uncertainty_threshold: float = 0.05
#     top_features_display: int = 5
#     batch_inference_size: int = 1000
    
#     # NEW: Enhanced reject option
#     use_reject_option: bool = True
#     reject_on_uncertainty: bool = True
#     reject_on_disagreement: bool = True
#     reject_on_boundary: bool = True  # NEW: Reject if near decision boundary
    
#     model_config = {
#         "env_prefix": "INFERENCE_"
#     }


# class APIConfig(BaseSettings):
#     """Configuration for FastAPI application."""
    
#     app_title: str = "TBFusionAI API"
#     app_description: str = "TB Detection using AI-powered Cough Analysis"
#     app_version: str = "1.0.0"
#     api_prefix: str = "/api/v1"
    
#     # CORS settings
#     cors_origins: List[str] = ["*"]
#     cors_credentials: bool = True
#     cors_methods: List[str] = ["*"]
#     cors_headers: List[str] = ["*"]
    
#     # File upload
#     max_upload_size: int = 10 * 1024 * 1024  # 10MB
#     allowed_audio_formats: List[str] = [".wav", ".mp3", ".ogg", ".webm"]
    
#     # Server
#     host: str = "0.0.0.0"
#     port: int = 8000
#     reload: bool = False
#     workers: int = 4
    
#     model_config = {
#         "env_prefix": "API_"
#     }


# class Config:
#     """Main configuration class that aggregates all configs."""
    
#     def __init__(self):
#         self.paths = PathConfig()
#         self.data_ingestion = DataIngestionConfig()
#         self.audio_extraction = AudioExtractionConfig()
#         self.audio_preprocessing = AudioPreprocessingConfig()
#         self.metadata = MetadataConfig()
#         self.ctgan = CTGANConfig()
#         self.model_training = ModelTrainingConfig()
#         self.ensemble = EnsembleConfig()
#         self.inference = InferenceConfig()
#         self.api = APIConfig()
        
#         # Create necessary directories
#         self._create_directories()
    
#     def _create_directories(self) -> None:
#         """Create necessary directories if they don't exist."""
#         directories = [
#             self.paths.artifacts_path,
#             self.paths.dataset_path,
#             self.paths.models_path,
#             self.paths.preprocessed_path,
#             self.paths.labeled_data_path,
#             self.paths.reports_path,
#         ]
        
#         for directory in directories:
#             directory.mkdir(parents=True, exist_ok=True)


# @lru_cache()
# def get_config() -> Config:
#     """
#     Get cached configuration instance.
    
#     Returns:
#         Config: Cached configuration object
#     """
#     return Config()