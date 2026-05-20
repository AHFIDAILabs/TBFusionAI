"""
Preprocessor module for TBFusionAI.

Contains:
1. AudioPreprocessor: Audio file preprocessing and feature extraction
2. FeaturePreprocessor: Feature encoding and scaling
"""

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from PIL import Image
from scipy.signal import butter, lfilter
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)


class AudioQualityError(Exception):
    """Raised when audio quality is below acceptable threshold."""

    pass


class AudioPreprocessor:
    """Audio preprocessing and feature extraction with WebM support."""

    def __init__(self):
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Wav2Vec2Model] = None
        self.processor: Optional[Wav2Vec2Processor] = None
        logger.info(f"AudioPreprocessor initialized on device: {self.device}")

    def load_wav2vec2_model(self) -> None:
        """Load Wav2Vec2 model and processor."""
        if self.model is not None:
            return

        logger.info("Loading Wav2Vec2 model...")
        cache_dir = self.config.paths.artifacts_path / "model_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_name = self.config.audio_extraction.model_name

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_name, cache_dir=str(cache_dir), local_files_only=False
            )
            self.model = Wav2Vec2Model.from_pretrained(
                model_name, cache_dir=str(cache_dir), local_files_only=False
            )
            self.model.eval()
            self.model = self.model.to(self.device)
            logger.info("Wav2Vec2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {str(e)}")
            raise

    def _convert_webm_to_wav(self, webm_bytes: bytes) -> bytes:
        """Convert WebM audio to WAV format using ffmpeg."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(webm_bytes)
                temp_webm_path = tmp.name

            temp_wav_path = temp_webm_path.replace(".webm", ".wav")

            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        temp_webm_path,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-y",
                        temp_wav_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                with open(temp_wav_path, "rb") as f:
                    wav_bytes = f.read()
                logger.info("Converted WebM to WAV using ffmpeg")
                return wav_bytes
            finally:
                Path(temp_webm_path).unlink(missing_ok=True)
                Path(temp_wav_path).unlink(missing_ok=True)

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed: {e.stderr}")
            raise ValueError(
                "Failed to convert WebM audio. Please try uploading a WAV file instead."
            )
        except Exception as e:
            logger.error(f"WebM conversion error: {str(e)}")
            raise ValueError(f"Audio conversion failed: {str(e)}")

    def load_audio(
        self, audio_input: Union[str, Path, bytes, io.BytesIO]
    ) -> Tuple[torch.Tensor, int]:
        """Load audio from various input types with WebM support."""
        try:
            if isinstance(audio_input, (str, Path)):
                with open(audio_input, "rb") as f:
                    audio_bytes = f.read()
            elif isinstance(audio_input, bytes):
                audio_bytes = audio_input
            elif isinstance(audio_input, io.BytesIO):
                audio_input.seek(0)
                audio_bytes = audio_input.read()
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            is_webm = (
                audio_bytes[:4] == b"\x1a\x45\xdf\xa3"
                or b"webm" in audio_bytes[:100].lower()
                or b"matroska" in audio_bytes[:100].lower()
            )

            if is_webm:
                logger.info("WebM format detected, converting to WAV...")
                audio_bytes = self._convert_webm_to_wav(audio_bytes)

            buffer = io.BytesIO(audio_bytes)
            buffer.seek(0)
            waveform, sr = torchaudio.load(buffer)

            if sr != self.config.audio_extraction.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sr, self.config.audio_extraction.sample_rate
                )
                waveform = resampler(waveform)
                sr = self.config.audio_extraction.sample_rate

            logger.info(f"Audio loaded: shape={waveform.shape}, sr={sr}Hz")
            return waveform, sr

        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise ValueError(
                f"Failed to load audio from {type(audio_input).__name__}: {str(e)}"
            )

    def normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.abs(waveform).max()
        if max_val > 0:
            return waveform / max_val
        return waveform

    def apply_bandpass_filter(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        nyq = 0.5 * sr
        low = self.config.audio_preprocessing.lowcut / nyq
        high = self.config.audio_preprocessing.highcut / nyq
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, low + 0.001, 0.999)
        b, a = butter(
            self.config.audio_preprocessing.filter_order, [low, high], btype="band"
        )
        return lfilter(b, a, waveform)

    def reduce_noise(
        self, waveform: np.ndarray, sr: int, noise_duration: float = 0.5
    ) -> np.ndarray:
        """Apply spectral noise reduction."""
        try:
            noise_sample_count = int(sr * noise_duration)
            if len(waveform) <= noise_sample_count:
                return waveform

            noise_profile = waveform[:noise_sample_count]
            D = librosa.stft(waveform)
            D_noise = librosa.stft(noise_profile)
            noise_mag = np.mean(np.abs(D_noise), axis=1, keepdims=True)
            magnitude = np.abs(D)
            phase = np.angle(D)
            reduced_mag = np.maximum(magnitude - noise_mag, 0.1 * magnitude)
            D_reduced = reduced_mag * np.exp(1j * phase)
            return librosa.istft(D_reduced, length=len(waveform))
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}. Using original audio.")
            return waveform

    def preprocess_audio(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO],
        apply_filters: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """Complete audio preprocessing pipeline."""
        waveform, sr = self.load_audio(audio_input)
        # Mix down to mono before converting to numpy
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        y = waveform.squeeze().numpy()
        y = self.normalize_audio(y)
        if apply_filters:
            y = self.apply_bandpass_filter(y, sr)
            y = self.reduce_noise(y, sr)
            y = self.normalize_audio(y)
        return y, sr

    def calculate_audio_metrics(
        self, audio_input: Union[str, Path, bytes, io.BytesIO]
    ) -> Dict[str, float]:
        """Calculate comprehensive audio quality metrics."""
        waveform, sr = self.load_audio(audio_input)
        # Mix down to mono before converting to numpy
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        y = waveform.squeeze().numpy()

        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        flatness = librosa.feature.spectral_flatness(y=y).mean()

        try:
            harmonic, _ = librosa.effects.hpss(y)
            signal_power = np.mean(harmonic**2)
            noise = y - harmonic
            noise_power = np.mean(noise**2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        except Exception:
            snr = 0.0

        clipping_threshold = 0.99
        clipping_ratio = float(np.sum(np.abs(y) > clipping_threshold) / len(y))

        silence_threshold = 0.01
        silence_ratio = float(np.sum(np.abs(y) < silence_threshold) / len(y))

        return {
            "duration": float(duration),
            "rms": float(rms),
            "zcr": float(zcr),
            "flatness": float(flatness),
            "snr": float(snr),
            "clipping_ratio": clipping_ratio,
            "silence_ratio": silence_ratio,
        }

    def validate_audio_quality(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO],
        min_snr: float = -5.0,
        max_clipping_ratio: float = 0.05,
        max_silence_ratio: float = 0.5,
        min_duration: float = 0.3,
        max_duration: float = 30.0,
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Validate audio quality meets minimum requirements."""
        metrics = self.calculate_audio_metrics(audio_input)

        if metrics["duration"] < min_duration:
            return (
                False,
                f"Audio too short ({metrics['duration']:.1f}s < {min_duration}s)",
                metrics,
            )
        if metrics["duration"] > max_duration:
            return (
                False,
                f"Audio too long ({metrics['duration']:.1f}s > {max_duration}s)",
                metrics,
            )
        if metrics["snr"] < min_snr:
            return (
                False,
                f"Poor audio quality (SNR {metrics['snr']:.1f}dB < {min_snr}dB). Record in a quieter environment.",
                metrics,
            )
        if metrics["clipping_ratio"] > max_clipping_ratio:
            return (
                False,
                f"Audio clipping detected ({metrics['clipping_ratio']*100:.1f}%). Reduce microphone volume.",
                metrics,
            )
        if metrics["silence_ratio"] > max_silence_ratio:
            return (
                False,
                f"Too much silence ({metrics['silence_ratio']*100:.1f}%). Speak louder or closer to microphone.",
                metrics,
            )

        return True, None, metrics

    def extract_features(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO],
        validate_quality: bool = True,
    ) -> np.ndarray:
        """Extract Wav2Vec2 features from audio."""
        if validate_quality:
            is_valid, error_msg, metrics = self.validate_audio_quality(audio_input)
            if not is_valid:
                logger.warning(f"Audio quality validation failed: {error_msg}")
                raise AudioQualityError(error_msg)
            logger.info(
                f"Audio quality validated: SNR={metrics['snr']:.1f}dB, "
                f"Duration={metrics['duration']:.1f}s"
            )

        if self.model is None or self.processor is None:
            self.load_wav2vec2_model()

        if self.processor is None or self.model is None:
            raise RuntimeError("Failed to load Wav2Vec2 model")

        y, sr = self.preprocess_audio(audio_input, apply_filters=True)
        waveform = torch.from_numpy(y).float()

        inputs = self.processor(
            waveform.squeeze(), sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)[0]

        logger.info(f"Features extracted: {embedding.shape}")
        return embedding

    def generate_spectrogram(
        self, audio_input: Union[str, Path, bytes, io.BytesIO]
    ) -> Image.Image:
        """Generate Mel spectrogram image."""
        y, sr = self.preprocess_audio(audio_input, apply_filters=True)

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.config.audio_preprocessing.n_mels
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(
            figsize=self.config.audio_preprocessing.spectrogram_figsize
        )
        librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma", ax=ax
        )
        ax.axis("off")
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=self.config.audio_preprocessing.spectrogram_dpi,
        )
        plt.close(fig)

        buf.seek(0)
        image = Image.open(buf)
        image.load()  # force pixel read before buf can be GC'd
        return image


class FeaturePreprocessor:
    """Feature encoding and preprocessing."""

    def __init__(self):
        self.config = get_config()
        logger.info("FeaturePreprocessor initialized")

    def encode_features(
        self, features: Dict[str, Union[int, float, str]]
    ) -> Dict[str, Union[int, float]]:
        """Encode categorical and binary features."""
        encoded: Dict[str, Union[int, float]] = {}

        for key, value in features.items():
            if isinstance(value, str):
                if key == "sex":
                    encoded[key] = 1 if value.lower() == "male" else 0
                elif key in self.config.metadata.binary_features:
                    encoded[key] = 1 if value.lower() == "yes" else 0
                else:
                    continue
            else:
                encoded[key] = value

        return encoded

    def validate_features(
        self, features: Dict[str, Union[int, float, str]]
    ) -> Tuple[bool, Optional[str]]:
        """Validate required features are present."""
        for feature in self.config.metadata.clinical_features:
            if feature not in features:
                return False, f"Missing required feature: {feature}"

        if not any(k.startswith("feat_") for k in features):
            return False, "Missing audio features. Audio must be processed first."

        return True, None

    def prepare_feature_dict(
        self,
        clinical_features: Dict[str, Union[int, float, str]],
        audio_features: np.ndarray,
    ) -> Dict[str, Union[int, float]]:
        """Combine clinical and audio features."""
        encoded_clinical = self.encode_features(clinical_features)
        for i, value in enumerate(audio_features):
            encoded_clinical[f"feat_{i}"] = float(value)
        return encoded_clinical


if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    logger.info("AudioPreprocessor ready for testing")
