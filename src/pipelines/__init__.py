"""
Pipeline modules for TBFusionAI project.

Contains all ML pipeline stages:
- Data ingestion
- Data processing (audio extraction, preprocessing, metadata matching)
- Model training
- Model evaluation
- Model inference
"""

from src.pipelines.data_ingestion import DataIngestionPipeline
from src.pipelines.data_processing import DataProcessingPipeline
from src.pipelines.model_training import ModelTrainingPipeline
from src.pipelines.model_evaluation import ModelEvaluationPipeline
from src.pipelines.model_inference import ModelInferencePipeline

__all__ = [
    "DataIngestionPipeline",
    "DataProcessingPipeline",
    "ModelTrainingPipeline",
    "ModelEvaluationPipeline",
    "ModelInferencePipeline"
] 