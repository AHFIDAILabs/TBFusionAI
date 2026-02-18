"""
Data Processing Pipeline for TBFusionAI.

Handles:
1. Audio feature extraction using Wav2Vec2
2. Audio preprocessing and segmentation
3. Metadata matching and feature integration
4. Class imbalance handling using cost-sensitive learning (NO CTGAN)
"""

import gc
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from sklearn.utils import shuffle
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)


class DataProcessingPipeline:
    """
    Complete data processing pipeline for audio and metadata.
    """

    def __init__(self):
        """
        Initialize the data processing pipeline.
        """
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_path = self.config.paths.dataset_path
        self.preprocessed_path = self.config.paths.preprocessed_path
        self.labeled_data_path = self.config.paths.labeled_data_path

        self._create_output_directories()

        logger.info(f"DataProcessingPipeline initialized on device: {self.device}")

    def _create_output_directories(self) -> None:
        """
        Create all necessary output directories.
        """
        directories = [
            self.preprocessed_path / "01_denoised",
            self.preprocessed_path / "02_segments_hybrid",
            self.preprocessed_path / "03_segments_filtered",
            self.preprocessed_path / "04_final_audio",
            self.preprocessed_path / "05_spectrograms",
            self.preprocessed_path / "plots",
            self.labeled_data_path
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    async def run(self) -> pd.DataFrame:
        """
        Execute the complete data processing pipeline.

        Returns:
            Prepared dataset DataFrame
        """
        logger.info("=" * 70)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("=" * 70)

        try:
            embeddings_df, stats_df = await self._extract_audio_features()
            final_audio_count = await self._preprocess_and_segment_audio()
            merged_df = await self._match_metadata_and_integrate_features(embeddings_df)
            prepared_df = await self._handle_class_imbalance(merged_df)

            logger.info("=" * 70)
            logger.info("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

            return prepared_df

        except Exception as e:
            logger.error(f"Data processing pipeline failed: {str(e)}")
            raise

    async def _extract_audio_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract Wav2Vec2 embeddings from audio files.

        Returns:
            Tuple of embeddings DataFrame and statistics DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: EXTRACTING WAV2VEC2 FEATURES")
        logger.info("=" * 70)

        audio_dir = self.dataset_path / "raw_data" / "longitudinal_data"
        output_embeddings = self.preprocessed_path / "longitudinal_wav2vec2_embeddings.csv"
        output_stats = self.preprocessed_path / "longitudinal_audio_statistics.csv"

        if output_embeddings.exists() and output_stats.exists():
            logger.info("Features already extracted. Loading from disk...")
            return pd.read_csv(output_embeddings), pd.read_csv(output_stats)

        model, processor = await self._load_wav2vec2_model()

        audio_files = sorted([
            f for f in audio_dir.iterdir()
            if f.suffix.lower() == ".wav"
        ])

        embedding_data = []
        stats_data = []

        for audio_file in tqdm(audio_files, desc="Extracting features"):
            try:
                waveform, sr, original_sr = self._load_audio_file(audio_file)
                embedding = self._extract_embedding(waveform, sr, model, processor)

                embedding_data.append([audio_file.name] + embedding.tolist())

                stats_data.append([
                    audio_file.name,
                    waveform.shape[1] / sr,
                    original_sr,
                    sr,
                    waveform.shape[0],
                    waveform.shape[1],
                    "success"
                ])

            except Exception as e:
                logger.warning(f"Error processing {audio_file.name}: {e}")

        columns_embed = ["filename"] + [f"feat_{i}" for i in range(len(embedding_data[0]) - 1)]
        embeddings_df = pd.DataFrame(embedding_data, columns=columns_embed)

        stats_df = pd.DataFrame(
            stats_data,
            columns=[
                "filename",
                "duration_sec",
                "original_sr",
                "resampled_sr",
                "channels",
                "total_samples",
                "status"
            ]
        )

        embeddings_df.to_csv(output_embeddings, index=False)
        stats_df.to_csv(output_stats, index=False)

        return embeddings_df, stats_df

    async def _load_wav2vec2_model(self) -> Tuple[Wav2Vec2Model, Wav2Vec2Processor]:
        """
        Load Wav2Vec2 model and processor.

        Returns:
            Loaded model and processor
        """
        cache_dir = self.config.paths.artifacts_path / "model_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(self.config.audio_extraction.max_retries):
            try:
                processor = Wav2Vec2Processor.from_pretrained(
                    self.config.audio_extraction.model_name,
                    cache_dir=str(cache_dir)
                )
                model = Wav2Vec2Model.from_pretrained(
                    self.config.audio_extraction.model_name,
                    cache_dir=str(cache_dir)
                )
                model.eval()
                return model.to(self.device), processor
            except Exception:
                if attempt == self.config.audio_extraction.max_retries - 1:
                    raise

    def _load_audio_file(self, filepath: Path) -> Tuple[torch.Tensor, int, int]:
        """
        Load audio file with fallback methods.

        Returns:
            Waveform tensor, resampled rate, original rate
        """
        try:
            waveform, sr = torchaudio.load(str(filepath))
        except Exception:
            try:
                sr, waveform_np = wavfile.read(str(filepath))
                waveform = torch.FloatTensor(waveform_np).unsqueeze(0) / 32768.0
            except Exception:
                waveform_np, sr = librosa.load(str(filepath), sr=None)
                waveform = torch.FloatTensor(waveform_np).unsqueeze(0)

        original_sr = sr

        if sr != self.config.audio_extraction.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.audio_extraction.sample_rate)
            waveform = resampler(waveform)
            sr = self.config.audio_extraction.sample_rate

        return waveform, sr, original_sr

    def _extract_embedding(
        self,
        waveform: torch.Tensor,
        sr: int,
        model: Wav2Vec2Model,
        processor: Wav2Vec2Processor
    ) -> np.ndarray:
        """
        Extract Wav2Vec2 embedding from waveform.

        Returns:
            Embedding vector
        """
        inputs = processor(
            waveform.squeeze(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)[0]

        return embedding

    async def _preprocess_and_segment_audio(self) -> int:
        """
        Preprocess and segment audio files.

        Returns:
            Number of final audio segments
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: AUDIO PREPROCESSING AND SEGMENTATION")
        logger.info("=" * 70)

        raw_audio_dir = self.dataset_path / "raw_data" / "longitudinal_data"
        final_audio_dir = self.preprocessed_path / "04_final_audio"

        if list(final_audio_dir.glob("*.wav")):
            return len(list(final_audio_dir.glob("*.wav")))

        audio_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(raw_audio_dir)
            for f in files if f.lower().endswith(".wav")
        ]

        await self._stage_denoise(audio_files)
        await self._stage_segment()
        filtered_count = await self._stage_filter_quality()
        await self._stage_generate_spectrograms()

        return filtered_count

    async def _stage_denoise(self, audio_files: List[str]) -> int:
        """
        Apply denoising and bandpass filtering.

        Returns:
            Number of processed files
        """
        denoised_dir = self.preprocessed_path / "01_denoised"

        for file_path in tqdm(audio_files, desc="Denoising"):
            try:
                y, sr = librosa.load(file_path, sr=None)
                y_filtered = self._butter_bandpass_filter(y, sr)
                y_reduced = nr.reduce_noise(
                    y=y_filtered,
                    sr=sr,
                    prop_decrease=self.config.audio_preprocessing.noise_prop_decrease
                )
                sf.write(denoised_dir / os.path.basename(file_path), y_reduced, sr)
            except Exception as e:
                logger.warning(f"Failed to denoise {file_path}: {e}")

        return len(audio_files)

    def _butter_bandpass_filter(self, data: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.
        """
        nyq = 0.5 * sr
        low = self.config.audio_preprocessing.lowcut / nyq
        high = self.config.audio_preprocessing.highcut / nyq
        b, a = butter(self.config.audio_preprocessing.filter_order, [low, high], btype="band")
        return lfilter(b, a, data)

    async def _stage_segment(self) -> int:
        """
        Segment audio using hybrid approach.

        Returns:
            Number of generated segments
        """
        denoised_dir = self.preprocessed_path / "01_denoised"
        hybrid_dir = self.preprocessed_path / "02_segments_hybrid"

        for file in tqdm(denoised_dir.glob("*.wav"), desc="Segmenting"):
            y, sr = librosa.load(file, sr=None)
            segments = self._extract_segments_energy(y, sr) + self._extract_segments_librosa(y, sr)

            for i, (start, end) in enumerate(set(segments)):
                sf.write(hybrid_dir / f"{file.stem}_seg{i}.wav", y[start:end], sr)

        return len(list(hybrid_dir.glob("*.wav")))

    def _extract_segments_energy(self, y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """
        Extract segments using energy thresholding.
        """
        energy = np.square(y)
        threshold = self.config.audio_preprocessing.energy_threshold_ratio * np.max(energy)
        indices = np.where(energy > threshold)[0]

        if len(indices) == 0:
            return []

        segments = []
        start = indices[0]
        silence_gap = int(self.config.audio_preprocessing.silence_gap_sec * sr)

        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] > silence_gap:
                end = indices[i - 1]
                if (end - start) / sr >= self.config.audio_preprocessing.min_segment_length_sec:
                    segments.append((start, end))
                start = indices[i]

        end = indices[-1]
        if (end - start) / sr >= self.config.audio_preprocessing.min_segment_length_sec:
            segments.append((start, end))

        return segments

    def _extract_segments_librosa(self, y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """
        Extract segments using librosa silence splitting.
        """
        intervals = librosa.effects.split(y, top_db=self.config.audio_preprocessing.librosa_top_db)
        return [
            (start, end)
            for start, end in intervals
            if (end - start) / sr >= self.config.audio_preprocessing.librosa_min_len_sec
        ]

    async def _stage_filter_quality(self) -> int:
        """
        Filter audio segments by quality metrics.

        Returns:
            Number of accepted segments
        """
        hybrid_dir = self.preprocessed_path / "02_segments_hybrid"
        final_audio_dir = self.preprocessed_path / "04_final_audio"

        count = 0
        for file in tqdm(hybrid_dir.glob("*.wav"), desc="Quality filtering"):
            y, sr = librosa.load(file, sr=None)
            metrics = self._calculate_audio_metrics(y, sr)

            if metrics["rms"] > self.config.audio_preprocessing.rms_min and metrics["snr"] > self.config.audio_preprocessing.snr_min:
                shutil.copy(file, final_audio_dir / file.name)
                count += 1

        return count

    def _calculate_audio_metrics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculate audio quality metrics.
        """
        return {
            "duration": librosa.get_duration(y=y, sr=sr),
            "rms": librosa.feature.rms(y=y).mean(),
            "zcr": librosa.feature.zero_crossing_rate(y).mean(),
            "flatness": librosa.feature.spectral_flatness(y=y).mean(),
            "snr": self._estimate_snr(y),
        }

    def _estimate_snr(self, y: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio.
        """
        try:
            harmonic, _ = librosa.effects.hpss(y)
            return 10 * np.log10(np.mean(y ** 2) / (np.mean((y - harmonic) ** 2) + 1e-10))
        except Exception:
            return 0.0

    async def _stage_generate_spectrograms(self) -> None:
        """
        Generate mel spectrograms from final audio segments.
        """
        final_audio_dir = self.preprocessed_path / "04_final_audio"
        spectrograms_dir = self.preprocessed_path / "05_spectrograms"

        for file in tqdm(final_audio_dir.glob("*.wav"), desc="Generating spectrograms"):
            y, sr = librosa.load(file, sr=None)
            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.config.audio_preprocessing.n_mels
            )
            S_db = librosa.power_to_db(S, ref=np.max)

            plt.figure(figsize=self.config.audio_preprocessing.spectrogram_figsize)
            librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(
                spectrograms_dir / file.name.replace(".wav", ".png"),
                dpi=self.config.audio_preprocessing.spectrogram_dpi,
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close()

    async def _match_metadata_and_integrate_features(self, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match audio embeddings with clinical and longitudinal metadata.

        Returns:
            Merged DataFrame
        """
        clinical_meta = pd.read_csv(
            self.dataset_path / "meta_data" / "Clinical" / "CODA_TB_Clinical_Meta_Info.csv"
        )
        longitudinal_meta = pd.read_csv(
            self.dataset_path / "meta_data" / "Cough Metadata" / "CODA_TB_Longitudnal_Meta_Info.csv"
        )

        clinical_meta.columns = clinical_meta.columns.str.lower()
        longitudinal_meta.columns = longitudinal_meta.columns.str.lower()

        metadata_merged = pd.merge(longitudinal_meta, clinical_meta, on="participant", how="inner")

        def extract_base_filename(name: str) -> str:
            return str(name).replace(".wav", "").replace(".png", "").split("_")[0]

        embeddings_df["base_filename"] = embeddings_df["filename"].apply(extract_base_filename)
        metadata_merged["base_filename"] = metadata_merged["filename"].apply(extract_base_filename)

        df_merged = pd.merge(embeddings_df, metadata_merged, on="base_filename", how="inner")
        df_merged.to_csv(self.labeled_data_path / "wav2vec2_with_labels.csv", index=False)

        return df_merged

    async def _handle_class_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for cost-sensitive learning WITHOUT synthetic generation.

        Returns:
            Cleaned DataFrame ready for training
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: PREPARING DATA FOR COST-SENSITIVE LEARNING")
        logger.info("=" * 70)

        cols_to_drop = [
            "participant", "filename_x", "filename_y", "sound_prediction_score",
            "height", "weight", "tb_prior_pul", "tb_prior_extrapul",
            "tb_prior_unknown", "heart_rate", "temperature", "smoke_lweek"
        ]

        df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        if "sex" in df_clean.columns:
            df_clean["sex"] = (df_clean["sex"] == "Male").astype(int)

        for col in self.config.metadata.binary_features:
            if col in df_clean.columns:
                df_clean[col] = (df_clean[col] == "Yes").astype(int)

        logger.info("\n📊 Class Distribution Analysis:")
        class_counts = df_clean["tb_status"].value_counts()
        logger.info(f"\n{class_counts}")

        minority = (df_clean["tb_status"] == 1).sum()
        majority = (df_clean["tb_status"] == 0).sum()
        ratio = majority / max(minority, 1)

        logger.info(f"  TB Negative (0): {majority:,}")
        logger.info(f"  TB Positive (1): {minority:,}")
        logger.info(f"  Imbalance Ratio: {ratio:.2f}:1")

        logger.info("\n✓ Using original imbalanced data (NO synthetic generation)")
        logger.info("✓ Models expected to apply cost-sensitive learning")

        df_clean = shuffle(df_clean, random_state=self.config.model_training.random_state)
        df_clean.to_csv(self.labeled_data_path / "wav2vec2_balanced_ctgan.csv", index=False)

        gc.collect()
        return df_clean


# Standalone execution function
async def run_data_processing() -> pd.DataFrame:
    """
    Standalone function to run data processing pipeline.

    Returns:
        Prepared dataset DataFrame
    """
    pipeline = DataProcessingPipeline()
    return await pipeline.run()


if __name__ == "__main__":
    import asyncio

    prepared_data = asyncio.run(run_data_processing())

    logger.info("Data processing completed successfully")
    logger.info(f"Final dataset shape: {prepared_data.shape}")