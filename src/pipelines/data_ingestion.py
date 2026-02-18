"""
Data Ingestion Pipeline for TBFusionAI - Updated with graceful Excel handling.

Handles downloading and extracting the CODA_TB_Dataset from Hugging Face,
processing metadata files, and preparing the raw data structure.
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestionPipeline:
    """
    Pipeline for ingesting TB dataset from Hugging Face.
    
    Attributes:
        config: Configuration object
        dataset_path: Path to dataset directory
        raw_data_path: Path to raw data directory
        meta_data_path: Path to metadata directory
    """
    
    def __init__(self):
        """Initialize the data ingestion pipeline."""
        self.config = get_config()
        self.dataset_path = self.config.paths.dataset_path
        self.raw_data_path = self.dataset_path / self.config.data_ingestion.raw_data_subdir
        self.meta_data_path = self.dataset_path / self.config.data_ingestion.meta_data_subdir
        
        logger.info("DataIngestionPipeline initialized")
    
    async def run(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete data ingestion pipeline.
        
        Returns:
            Dictionary containing all loaded metadata DataFrames
        
        Raises:
            Exception: If critical steps of the pipeline fail
        """
        logger.info("=" * 70)
        logger.info("STARTING DATA INGESTION PIPELINE")
        logger.info("=" * 70)
        
        try:
            # Step 1: Download and extract dataset
            await self._download_and_extract_dataset()
            
            # Step 2: Load and validate metadata
            metadata_dfs = await self._load_metadata_files()
            
            # Step 3: Verify audio files
            await self._verify_audio_files()
            
            logger.info("=" * 70)
            logger.info("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            
            return metadata_dfs
            
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            raise
    
    async def _download_and_extract_dataset(self) -> None:
        """
        Download dataset from Hugging Face and extract it.
        
        Raises:
            Exception: If download or extraction fails
        """
        logger.info("Step 1: Downloading and extracting dataset from Hugging Face")
        
        # Check if dataset already exists
        if self.dataset_path.exists() and len(list(self.dataset_path.iterdir())) > 0:
            logger.info(f"Dataset already exists at {self.dataset_path}. Skipping download.")
            return
        
        try:
            # Download ZIP file
            logger.info(f"Downloading {self.config.data_ingestion.dataset_filename}...")
            zip_file = hf_hub_download(
                repo_id=self.config.data_ingestion.repo_id,
                filename=self.config.data_ingestion.dataset_filename,
                repo_type="dataset"
            )
            logger.info(f"✓ Downloaded to: {zip_file}")
            
            # Extract files with flattening
            logger.info(f"Extracting to {self.config.paths.artifacts_path}...")
            self._extract_with_flattening(zip_file)
            
            logger.info("✓ Extraction complete")
            
        except Exception as e:
            logger.error(f"Failed to download/extract dataset: {str(e)}")
            raise
    
    def _extract_with_flattening(self, zip_file: str) -> None:
        """
        Extract ZIP file and flatten deeply nested audio folders.
        
        Args:
            zip_file: Path to the ZIP file
        """
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for member in zip_ref.infolist():
                # Skip empty paths and directories
                if not member.filename or member.filename.endswith('/'):
                    continue
                
                parts = member.filename.split('/')
                
                # Flatten longitudinal_data nested folders
                if 'longitudinal_data' in parts:
                    idx = parts.index('longitudinal_data')
                    # Keep: dataset/raw_data/longitudinal_data/filename
                    new_path = '/'.join(parts[:idx+1] + [parts[-1]])
                    member.filename = new_path
                
                # Flatten solicited_data nested folders
                elif 'solicited_data' in parts:
                    idx = parts.index('solicited_data')
                    # Keep: dataset/raw_data/solicited_data/filename
                    new_path = '/'.join(parts[:idx+1] + [parts[-1]])
                    member.filename = new_path
                
                # Keep meta_data structure as-is
                
                try:
                    zip_ref.extract(member, self.config.paths.artifacts_path)
                except Exception as e:
                    logger.warning(f"Could not extract {member.filename}: {e}")
    
    async def _load_metadata_files(self) -> Dict[str, pd.DataFrame]:
        """
        Find and load all metadata CSV files.
        
        Returns:
            Dictionary mapping metadata type to DataFrame
        
        Raises:
            FileNotFoundError: If critical metadata files are missing
        """
        logger.info("Step 2: Loading metadata files")
        
        metadata_files = {
            'clinical_add': self._find_file('*additional_variables_train.csv'),
            'clinical_meta': self._find_file('*Clinical_Meta_Info.csv'),
            'longitudinal_meta': self._find_file('*Longitudnal_Meta_Info.csv'),
            'solicited_meta': self._find_file('*Solicited_Meta_Info.csv'),
            'data_dict': self._find_file('*data dictionary*.xlsx')
        }
        
        metadata_dfs = {}
        critical_files = ['clinical_meta', 'longitudinal_meta']  # Excel is optional
        
        for key, filepath in metadata_files.items():
            if filepath is None:
                if key in critical_files:
                    logger.error(f"Critical metadata file not found for: {key}")
                    raise FileNotFoundError(f"Critical metadata file missing: {key}")
                else:
                    logger.warning(f"Optional metadata file not found for: {key}")
                continue
            
            try:
                if filepath.suffix == '.csv':
                    df = pd.read_csv(filepath)
                    metadata_dfs[key] = df
                    logger.info(f"✓ Loaded {key}: {df.shape}")
                    
                elif filepath.suffix == '.xlsx':
                    # Excel files are optional (like data dictionary)
                    try:
                        df = pd.read_excel(filepath)
                        metadata_dfs[key] = df
                        logger.info(f"✓ Loaded {key}: {df.shape}")
                    except ImportError as e:
                        logger.warning(f"Could not load Excel file {key}: {str(e)}")
                        logger.warning("Install openpyxl to read Excel files: pip install openpyxl")
                        logger.info(f"⏭ Skipping optional Excel file: {key}")
                    except Exception as e:
                        logger.warning(f"Failed to load Excel file {key}: {str(e)}")
                        logger.info(f"⏭ Skipping optional Excel file: {key}")
                else:
                    logger.warning(f"Unsupported file format: {filepath}")
                    continue
                
            except Exception as e:
                if key in critical_files:
                    logger.error(f"Failed to load critical file {key} from {filepath}: {e}")
                    raise
                else:
                    logger.warning(f"Failed to load optional file {key} from {filepath}: {e}")
        
        # Verify critical files were loaded
        for critical_file in critical_files:
            if critical_file not in metadata_dfs:
                raise FileNotFoundError(f"Critical metadata file not loaded: {critical_file}")
        
        return metadata_dfs
    
    def _find_file(self, pattern: str) -> Optional[Path]:
        """
        Find file matching pattern in dataset directory.
        
        Args:
            pattern: Glob pattern to match
        
        Returns:
            Path to first matching file or None
        """
        if self.dataset_path.exists():
            matches = list(self.dataset_path.rglob(pattern))
            if matches:
                return matches[0]
        return None
    
    async def _verify_audio_files(self) -> None:
        """
        Verify that audio files exist in expected directories.
        """
        logger.info("Step 3: Verifying audio files")
        
        audio_dirs = [
            (
                self.raw_data_path / self.config.data_ingestion.longitudinal_data_dir,
                "longitudinal_data"
            ),
            (
                self.raw_data_path / self.config.data_ingestion.solicited_data_dir,
                "solicited_data"
            )
        ]
        
        for audio_dir, name in audio_dirs:
            if not audio_dir.exists():
                logger.warning(f"Missing directory: {audio_dir}")
                continue
            
            # Count audio files
            audio_files = [
                f for f in audio_dir.iterdir() 
                if f.suffix.lower() in ['.wav', '.mp3', '.ogg']
            ]
            
            logger.info(f"✓ Found {len(audio_files)} audio files in {name}")
            
            if audio_files:
                # Show sample files
                samples = [f.name for f in audio_files[:3]]
                logger.info(f"  Sample: {samples}")
    
    def get_metadata_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics of loaded metadata.
        
        Returns:
            Dictionary with metadata summaries
        """
        summary = {}
        
        metadata_files = {
            'clinical_meta': self._find_file('*Clinical_Meta_Info.csv'),
            'longitudinal_meta': self._find_file('*Longitudnal_Meta_Info.csv'),
        }
        
        for key, filepath in metadata_files.items():
            if filepath and filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    summary[key] = {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'missing_values': df.isnull().sum().to_dict()
                    }
                except Exception as e:
                    logger.error(f"Failed to summarize {key}: {e}")
        
        return summary


# Standalone execution function
async def run_data_ingestion() -> Dict[str, pd.DataFrame]:
    """
    Standalone function to run data ingestion pipeline.
    
    Returns:
        Dictionary containing loaded metadata DataFrames
    """
    pipeline = DataIngestionPipeline()
    return await pipeline.run()


if __name__ == "__main__":
    import asyncio
    
    # Run pipeline
    metadata = asyncio.run(run_data_ingestion())
    
    logger.info("\nData ingestion completed successfully!")
    logger.info(f"Loaded {len(metadata)} metadata files")