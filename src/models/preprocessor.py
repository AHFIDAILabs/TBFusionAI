"""
Preprocessor module for TBFusionAI - FIXED: WebM audio support.

Contains:
1. AudioPreprocessor: Audio file preprocessing and feature extraction
2. FeaturePreprocessor: Feature encoding and scaling

CRITICAL FIX: Added WebM → WAV conversion using pydub/ffmpeg
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
import soundfile as sf
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
    """
    Audio preprocessing and feature extraction with WebM support.
    """

    def __init__(self):
        """Initialize the audio preprocessor."""
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
            logger.info(f"✓ Wav2Vec2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {str(e)}")
            raise

    def _convert_webm_to_wav(self, webm_bytes: bytes) -> bytes:
        """
        Convert WebM audio to WAV format using ffmpeg.

        Args:
            webm_bytes: WebM audio data as bytes

        Returns:
            WAV audio data as bytes

        CRITICAL FIX: This enables WebM recording support!
        """
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
                temp_webm.write(webm_bytes)
                temp_webm_path = temp_webm.name

            temp_wav_path = temp_webm_path.replace(".webm", ".wav")

            try:
                # Convert using ffmpeg
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        temp_webm_path,
                        "-ar",
                        "16000",  # Resample to 16kHz
                        "-ac",
                        "1",  # Mono
                        "-y",  # Overwrite
                        temp_wav_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                # Read converted WAV
                with open(temp_wav_path, "rb") as f:
                    wav_bytes = f.read()

                logger.info("✓ Converted WebM to WAV using ffmpeg")
                return wav_bytes

            finally:
                # Cleanup temp files
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
        """
        Load audio from various input types with WebM support.

        Args:
            audio_input: File path, bytes, or BytesIO object

        Returns:
            Tuple of (waveform tensor, sample rate)

        CRITICAL FIX: Auto-converts WebM to WAV before loading
        """
        try:
            # Convert input to bytes for format detection
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

            # Detect WebM format (magic bytes check)
            is_webm = (
                audio_bytes[:4] == b"\x1a\x45\xdf\xa3"  # EBML header
                or b"webm" in audio_bytes[:100].lower()
                or b"matroska" in audio_bytes[:100].lower()
            )

            if is_webm:
                logger.info("🔄 WebM format detected, converting to WAV...")
                audio_bytes = self._convert_webm_to_wav(audio_bytes)

            # Load audio using torchaudio
            buffer = io.BytesIO(audio_bytes)
            buffer.seek(0)
            waveform, sr = torchaudio.load(buffer)

            # Resample to 16kHz if needed
            if sr != self.config.audio_extraction.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sr, self.config.audio_extraction.sample_rate
                )
                waveform = resampler(waveform)
                sr = self.config.audio_extraction.sample_rate

            logger.info(f"✓ Audio loaded: shape={waveform.shape}, sr={sr}Hz")
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
            reduced_waveform = librosa.istft(D_reduced, length=len(waveform))
            return reduced_waveform
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
        y = waveform.squeeze().numpy()

        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        flatness = librosa.feature.spectral_flatness(y=y).mean()

        try:
            harmonic, percussive = librosa.effects.hpss(y)
            signal_power = np.mean(harmonic**2)
            noise = y - harmonic
            noise_power = np.mean(noise**2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        except Exception:
            snr = 0.0

        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        clipping_ratio = clipped_samples / len(y)

        silence_threshold = 0.01
        silent_frames = np.sum(np.abs(y) < silence_threshold)
        silence_ratio = silent_frames / len(y)

        return {
            "duration": float(duration),
            "rms": float(rms),
            "zcr": float(zcr),
            "flatness": float(flatness),
            "snr": float(snr),
            "clipping_ratio": float(clipping_ratio),
            "silence_ratio": float(silence_ratio),
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
                f"Poor audio quality (SNR {metrics['snr']:.1f}dB < {min_snr}dB). Please record in a quieter environment.",
                metrics,
            )
        if metrics["clipping_ratio"] > max_clipping_ratio:
            return (
                False,
                f"Audio clipping detected ({metrics['clipping_ratio']*100:.1f}% > {max_clipping_ratio*100:.1f}%). Please reduce microphone volume.",
                metrics,
            )
        if metrics["silence_ratio"] > max_silence_ratio:
            return (
                False,
                f"Too much silence ({metrics['silence_ratio']*100:.1f}% > {max_silence_ratio*100:.1f}%). Please speak louder or closer to microphone.",
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
                f"✓ Audio quality validated: SNR={metrics['snr']:.1f}dB, Duration={metrics['duration']:.1f}s"
            )

        # Ensure model is loaded
        if self.model is None or self.processor is None:
            self.load_wav2vec2_model()

        # Type guard to satisfy Pylance
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
            last_hidden_state = outputs.last_hidden_state
            embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)[0]

        logger.info(f"✓ Features extracted: {embedding.shape}")
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

        # Copy and convert features
        for key, value in features.items():
            if isinstance(value, str):
                # Handle string values
                if key == "sex":
                    encoded[key] = 1 if value.lower() == "male" else 0
                elif key in self.config.metadata.binary_features:
                    encoded[key] = 1 if value.lower() == "yes" else 0
                else:
                    # Skip non-numeric string features (shouldn't happen)
                    continue
            else:
                # Already numeric
                encoded[key] = value

        return encoded

    def validate_features(
        self, features: Dict[str, Union[int, float, str]]
    ) -> Tuple[bool, Optional[str]]:
        """Validate required features are present."""
        required_clinical = self.config.metadata.clinical_features

        for feature in required_clinical:
            if feature not in features:
                return False, f"Missing required feature: {feature}"

        audio_features = [k for k in features.keys() if k.startswith("feat_")]
        if len(audio_features) == 0:
            return False, "Missing audio features. Audio must be processed first."

        return True, None

    def prepare_feature_dict(
        self,
        clinical_features: Dict[str, Union[int, float, str]],
        audio_features: np.ndarray,
    ) -> Dict[str, Union[int, float]]:
        """Combine clinical and audio features."""
        # Encode clinical features (returns Dict[str, Union[int, float]])
        encoded_clinical = self.encode_features(clinical_features)

        # Add audio features (all float)
        for i, value in enumerate(audio_features):
            encoded_clinical[f"feat_{i}"] = float(value)

        return encoded_clinical


# Convenience functions
def extract_audio_features(
    audio_input: Union[str, Path, bytes, io.BytesIO], validate_quality: bool = True
) -> np.ndarray:
    """Quick function to extract audio features."""
    preprocessor = AudioPreprocessor()
    return preprocessor.extract_features(audio_input, validate_quality=validate_quality)


def generate_spectrogram(
    audio_input: Union[str, Path, bytes, io.BytesIO],
) -> Image.Image:
    """Quick function to generate spectrogram."""
    preprocessor = AudioPreprocessor()
    return preprocessor.generate_spectrogram(audio_input)


def validate_audio_quality(
    audio_input: Union[str, Path, bytes, io.BytesIO],
) -> Tuple[bool, Optional[str], Dict[str, float]]:
    """Quick function to validate audio quality."""
    preprocessor = AudioPreprocessor()
    return preprocessor.validate_audio_quality(audio_input)


if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    logger.info("AudioPreprocessor ready for testing")


# """
# Preprocessor module for TBFusionAI.

# Contains:
# 1. AudioPreprocessor: Audio file preprocessing and feature extraction
# 2. FeaturePreprocessor: Feature encoding and scaling

# ENHANCEMENTS:
# - Fixed BytesIO handling with seek(0)
# - Automatic bandpass filtering
# - Spectral noise reduction
# - Audio normalization
# - Quality rejection (SNR threshold)
# - Clipping detection
# - Silence ratio calculation
# """

# import io
# from pathlib import Path
# from typing import Dict, Optional, Tuple, Union

# import librosa
# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# import soundfile as sf
# import torch
# import torchaudio
# from PIL import Image
# from scipy.signal import butter, lfilter
# from transformers import Wav2Vec2Model, Wav2Vec2Processor

# from src.config import get_config
# from src.logger import get_logger

# logger = get_logger(__name__)


# class AudioQualityError(Exception):
#     """Raised when audio quality is below acceptable threshold."""

#     pass


# class AudioPreprocessor:
#     """
#     Audio preprocessing and feature extraction.

#     Handles:
#     - Audio loading and resampling
#     - Noise reduction and filtering
#     - Audio normalization
#     - Quality validation
#     - Wav2Vec2 feature extraction
#     - Spectrogram generation
#     """

#     def __init__(self):
#         """Initialize the audio preprocessor."""
#         self.config = get_config()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Load Wav2Vec2 model
#         self.model: Optional[Wav2Vec2Model] = None
#         self.processor: Optional[Wav2Vec2Processor] = None

#         logger.info(f"AudioPreprocessor initialized on device: {self.device}")

#     def load_wav2vec2_model(self) -> None:
#         """Load Wav2Vec2 model and processor."""
#         if self.model is not None:
#             return  # Already loaded

#         logger.info("Loading Wav2Vec2 model...")

#         cache_dir = self.config.paths.artifacts_path / "model_cache"
#         cache_dir.mkdir(parents=True, exist_ok=True)

#         model_name = self.config.audio_extraction.model_name

#         try:
#             self.processor = Wav2Vec2Processor.from_pretrained(
#                 model_name, cache_dir=str(cache_dir), local_files_only=False
#             )

#             self.model = Wav2Vec2Model.from_pretrained(
#                 model_name, cache_dir=str(cache_dir), local_files_only=False
#             )

#             self.model.eval()
#             self.model = self.model.to(self.device)

#             logger.info(f"✓ Wav2Vec2 model loaded successfully")

#         except Exception as e:
#             logger.error(f"Failed to load Wav2Vec2 model: {str(e)}")
#             raise

#     def load_audio(
#         self, audio_input: Union[str, Path, bytes, io.BytesIO]
#     ) -> Tuple[torch.Tensor, int]:
#         """
#         Load audio from various input types.

#         Args:
#             audio_input: File path, bytes, or BytesIO object

#         Returns:
#             Tuple of (waveform tensor, sample rate)

#         FIXED: Added seek(0) for BytesIO handling
#         """
#         try:
#             # Handle different input types
#             if isinstance(audio_input, (str, Path)):
#                 waveform, sr = torchaudio.load(str(audio_input))

#             elif isinstance(audio_input, bytes):
#                 buffer = io.BytesIO(audio_input)
#                 buffer.seek(0)  # ← CRITICAL FIX
#                 waveform, sr = torchaudio.load(buffer)

#             elif isinstance(audio_input, io.BytesIO):
#                 audio_input.seek(0)  # ← CRITICAL FIX
#                 waveform, sr = torchaudio.load(audio_input)

#             else:
#                 raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

#             # Resample to 16kHz if needed
#             if sr != self.config.audio_extraction.sample_rate:
#                 resampler = torchaudio.transforms.Resample(
#                     sr, self.config.audio_extraction.sample_rate
#                 )
#                 waveform = resampler(waveform)
#                 sr = self.config.audio_extraction.sample_rate

#             return waveform, sr

#         except Exception as e:
#             logger.error(
#                 f"Failed to load audio from {type(audio_input).__name__}: {str(e)}"
#             )
#             raise ValueError(
#                 f"Failed to load audio from {type(audio_input).__name__}: {str(e)}"
#             )

#     def normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
#         """
#         Normalize audio to [-1, 1] range.

#         Args:
#             waveform: Audio waveform

#         Returns:
#             Normalized waveform
#         """
#         max_val = np.abs(waveform).max()
#         if max_val > 0:
#             return waveform / max_val
#         return waveform

#     def apply_bandpass_filter(self, waveform: np.ndarray, sr: int) -> np.ndarray:
#         """
#         Apply Butterworth bandpass filter to remove frequencies outside speech range.

#         Args:
#             waveform: Audio waveform
#             sr: Sample rate

#         Returns:
#             Filtered waveform
#         """
#         nyq = 0.5 * sr
#         low = self.config.audio_preprocessing.lowcut / nyq
#         high = self.config.audio_preprocessing.highcut / nyq

#         # Clamp to valid range
#         low = np.clip(low, 0.001, 0.999)
#         high = np.clip(high, low + 0.001, 0.999)

#         b, a = butter(
#             self.config.audio_preprocessing.filter_order, [low, high], btype="band"
#         )

#         return lfilter(b, a, waveform)

#     def reduce_noise(
#         self, waveform: np.ndarray, sr: int, noise_duration: float = 0.5
#     ) -> np.ndarray:
#         """
#         Apply spectral noise reduction using noise profile from beginning.

#         Args:
#             waveform: Audio waveform
#             sr: Sample rate
#             noise_duration: Duration of noise profile in seconds

#         Returns:
#             Noise-reduced waveform
#         """
#         try:
#             # Use first noise_duration seconds as noise profile
#             noise_sample_count = int(sr * noise_duration)

#             if len(waveform) <= noise_sample_count:
#                 # Audio too short, skip noise reduction
#                 return waveform

#             # Simple spectral subtraction
#             # Get noise profile from beginning
#             noise_profile = waveform[:noise_sample_count]

#             # Compute STFT
#             D = librosa.stft(waveform)
#             D_noise = librosa.stft(noise_profile)

#             # Estimate noise magnitude
#             noise_mag = np.mean(np.abs(D_noise), axis=1, keepdims=True)

#             # Spectral subtraction
#             magnitude = np.abs(D)
#             phase = np.angle(D)

#             # Subtract noise with floor
#             reduced_mag = np.maximum(magnitude - noise_mag, 0.1 * magnitude)

#             # Reconstruct
#             D_reduced = reduced_mag * np.exp(1j * phase)
#             reduced_waveform = librosa.istft(D_reduced, length=len(waveform))

#             return reduced_waveform

#         except Exception as e:
#             logger.warning(f"Noise reduction failed: {e}. Using original audio.")
#             return waveform

#     def preprocess_audio(
#         self,
#         audio_input: Union[str, Path, bytes, io.BytesIO],
#         apply_filters: bool = True,
#     ) -> Tuple[np.ndarray, int]:
#         """
#         Complete audio preprocessing pipeline.

#         Args:
#             audio_input: Audio file path, bytes, or BytesIO
#             apply_filters: Whether to apply bandpass filter and noise reduction

#         Returns:
#             Tuple of (preprocessed waveform, sample rate)
#         """
#         # Load audio
#         waveform, sr = self.load_audio(audio_input)
#         y = waveform.squeeze().numpy()

#         # Normalize
#         y = self.normalize_audio(y)

#         if apply_filters:
#             # Apply bandpass filter
#             y = self.apply_bandpass_filter(y, sr)

#             # Apply noise reduction
#             y = self.reduce_noise(y, sr)

#             # Normalize again after filtering
#             y = self.normalize_audio(y)

#         return y, sr

#     def calculate_audio_metrics(
#         self, audio_input: Union[str, Path, bytes, io.BytesIO]
#     ) -> Dict[str, float]:
#         """
#         Calculate comprehensive audio quality metrics.

#         Args:
#             audio_input: Audio file path, bytes, or BytesIO

#         Returns:
#             Dictionary of audio metrics including:
#             - duration: Audio length in seconds
#             - rms: Root mean square energy
#             - zcr: Zero crossing rate
#             - flatness: Spectral flatness
#             - snr: Signal-to-noise ratio estimate
#             - clipping_ratio: Ratio of clipped samples
#             - silence_ratio: Ratio of silent frames
#         """
#         waveform, sr = self.load_audio(audio_input)
#         y = waveform.squeeze().numpy()

#         # Basic metrics
#         duration = librosa.get_duration(y=y, sr=sr)
#         rms = librosa.feature.rms(y=y).mean()
#         zcr = librosa.feature.zero_crossing_rate(y).mean()
#         flatness = librosa.feature.spectral_flatness(y=y).mean()

#         # Estimate SNR using harmonic-percussive separation
#         try:
#             harmonic, percussive = librosa.effects.hpss(y)
#             signal_power = np.mean(harmonic**2)
#             noise = y - harmonic
#             noise_power = np.mean(noise**2)
#             snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
#         except Exception:
#             snr = 0.0

#         # Clipping detection
#         clipping_threshold = 0.99
#         clipped_samples = np.sum(np.abs(y) > clipping_threshold)
#         clipping_ratio = clipped_samples / len(y)

#         # Silence detection
#         silence_threshold = 0.01
#         silent_frames = np.sum(np.abs(y) < silence_threshold)
#         silence_ratio = silent_frames / len(y)

#         return {
#             "duration": float(duration),
#             "rms": float(rms),
#             "zcr": float(zcr),
#             "flatness": float(flatness),
#             "snr": float(snr),
#             "clipping_ratio": float(clipping_ratio),
#             "silence_ratio": float(silence_ratio),
#         }

#     def validate_audio_quality(
#         self,
#         audio_input: Union[str, Path, bytes, io.BytesIO],
#         min_snr: float = -5.0,
#         max_clipping_ratio: float = 0.05,
#         max_silence_ratio: float = 0.5,
#         min_duration: float = 0.3,
#         max_duration: float = 30.0,
#     ) -> Tuple[bool, Optional[str], Dict[str, float]]:
#         """
#         Validate audio quality meets minimum requirements.

#         Args:
#             audio_input: Audio file path, bytes, or BytesIO
#             min_snr: Minimum acceptable SNR in dB
#             max_clipping_ratio: Maximum acceptable clipping ratio
#             max_silence_ratio: Maximum acceptable silence ratio
#             min_duration: Minimum audio duration in seconds
#             max_duration: Maximum audio duration in seconds

#         Returns:
#             Tuple of (is_valid, error_message, metrics)
#         """
#         metrics = self.calculate_audio_metrics(audio_input)

#         # Check duration
#         if metrics["duration"] < min_duration:
#             return (
#                 False,
#                 f"Audio too short ({metrics['duration']:.1f}s < {min_duration}s)",
#                 metrics,
#             )

#         if metrics["duration"] > max_duration:
#             return (
#                 False,
#                 f"Audio too long ({metrics['duration']:.1f}s > {max_duration}s)",
#                 metrics,
#             )

#         # Check SNR
#         if metrics["snr"] < min_snr:
#             return (
#                 False,
#                 f"Poor audio quality (SNR {metrics['snr']:.1f}dB < {min_snr}dB). Please record in a quieter environment.",
#                 metrics,
#             )

#         # Check clipping
#         if metrics["clipping_ratio"] > max_clipping_ratio:
#             return (
#                 False,
#                 f"Audio clipping detected ({metrics['clipping_ratio']*100:.1f}% > {max_clipping_ratio*100:.1f}%). Please reduce microphone volume.",
#                 metrics,
#             )

#         # Check silence
#         if metrics["silence_ratio"] > max_silence_ratio:
#             return (
#                 False,
#                 f"Too much silence ({metrics['silence_ratio']*100:.1f}% > {max_silence_ratio*100:.1f}%). Please speak louder or closer to microphone.",
#                 metrics,
#             )

#         return True, None, metrics

#     def extract_features(
#         self,
#         audio_input: Union[str, Path, bytes, io.BytesIO],
#         validate_quality: bool = True,
#     ) -> np.ndarray:
#         """
#         Extract Wav2Vec2 features from audio with automatic preprocessing.

#         Args:
#             audio_input: Audio file path, bytes, or BytesIO
#             validate_quality: Whether to validate audio quality before extraction

#         Returns:
#             Feature embedding as numpy array (768-dimensional)

#         Raises:
#             AudioQualityError: If audio quality is below threshold
#         """
#         # Validate quality if requested
#         if validate_quality:
#             is_valid, error_msg, metrics = self.validate_audio_quality(audio_input)
#             if not is_valid:
#                 logger.warning(f"Audio quality validation failed: {error_msg}")
#                 raise AudioQualityError(error_msg)
#             logger.info(
#                 f"✓ Audio quality validated: SNR={metrics['snr']:.1f}dB, Duration={metrics['duration']:.1f}s"
#             )

#         # Ensure model is loaded
#         if self.model is None:
#             self.load_wav2vec2_model()

#         # Preprocess audio (normalize, filter, denoise)
#         y, sr = self.preprocess_audio(audio_input, apply_filters=True)

#         # Convert to tensor
#         waveform = torch.from_numpy(y).float()

#         # Preprocess for Wav2Vec2
#         inputs = self.processor(
#             waveform.squeeze(), sampling_rate=sr, return_tensors="pt", padding=True
#         )

#         # Move to device
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         # Extract features
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             last_hidden_state = outputs.last_hidden_state
#             embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

#         if embedding.ndim == 1:
#             embedding = embedding.reshape(1, -1)[0]

#         logger.info(f"✓ Features extracted: {embedding.shape}")

#         return embedding

#     def generate_spectrogram(
#         self, audio_input: Union[str, Path, bytes, io.BytesIO]
#     ) -> Image.Image:
#         """
#         Generate Mel spectrogram image.

#         Args:
#             audio_input: Audio file path, bytes, or BytesIO

#         Returns:
#             PIL Image of spectrogram
#         """
#         # Load and preprocess audio
#         y, sr = self.preprocess_audio(audio_input, apply_filters=True)

#         # Compute Mel spectrogram
#         S = librosa.feature.melspectrogram(
#             y=y, sr=sr, n_mels=self.config.audio_preprocessing.n_mels
#         )
#         S_db = librosa.power_to_db(S, ref=np.max)

#         # Plot
#         fig, ax = plt.subplots(
#             figsize=self.config.audio_preprocessing.spectrogram_figsize
#         )

#         librosa.display.specshow(
#             S_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma", ax=ax
#         )

#         ax.axis("off")
#         plt.tight_layout(pad=0)

#         # Convert to PIL Image
#         buf = io.BytesIO()
#         plt.savefig(
#             buf,
#             format="png",
#             bbox_inches="tight",
#             pad_inches=0,
#             dpi=self.config.audio_preprocessing.spectrogram_dpi,
#         )
#         plt.close(fig)

#         buf.seek(0)
#         image = Image.open(buf)

#         return image


# class FeaturePreprocessor:
#     """
#     Feature encoding and preprocessing.

#     Handles:
#     - Categorical feature encoding
#     - Binary feature encoding
#     - Feature validation
#     """

#     def __init__(self):
#         """Initialize the feature preprocessor."""
#         self.config = get_config()
#         logger.info("FeaturePreprocessor initialized")

#     def encode_features(
#         self, features: Dict[str, Union[int, float, str]]
#     ) -> Dict[str, Union[int, float]]:
#         """
#         Encode categorical and binary features.

#         Args:
#             features: Raw feature dictionary

#         Returns:
#             Encoded feature dictionary
#         """
#         encoded = features.copy()

#         # Encode sex (categorical)
#         if "sex" in encoded:
#             if isinstance(encoded["sex"], str):
#                 encoded["sex"] = 1 if encoded["sex"].lower() == "male" else 0

#         # Encode binary features (yes/no)
#         binary_features = self.config.metadata.binary_features
#         for feature in binary_features:
#             if feature in encoded:
#                 if isinstance(encoded[feature], str):
#                     encoded[feature] = 1 if encoded[feature].lower() == "yes" else 0

#         return encoded

#     def validate_features(
#         self, features: Dict[str, Union[int, float, str]]
#     ) -> Tuple[bool, Optional[str]]:
#         """
#         Validate required features are present.

#         Args:
#             features: Feature dictionary

#         Returns:
#             Tuple of (is_valid, error_message)
#         """
#         required_clinical = self.config.metadata.clinical_features

#         # Check for required clinical features
#         for feature in required_clinical:
#             if feature not in features:
#                 return False, f"Missing required feature: {feature}"

#         # Check for audio features
#         audio_features = [k for k in features.keys() if k.startswith("feat_")]
#         if len(audio_features) == 0:
#             return False, "Missing audio features. Audio must be processed first."

#         return True, None

#     def prepare_feature_dict(
#         self,
#         clinical_features: Dict[str, Union[int, float, str]],
#         audio_features: np.ndarray,
#     ) -> Dict[str, Union[int, float]]:
#         """
#         Combine clinical and audio features into single dictionary.

#         Args:
#             clinical_features: Clinical feature dictionary
#             audio_features: Audio feature array (768-dimensional)

#         Returns:
#             Combined feature dictionary
#         """
#         # Encode clinical features
#         encoded_clinical = self.encode_features(clinical_features)

#         # Add audio features
#         for i, value in enumerate(audio_features):
#             encoded_clinical[f"feat_{i}"] = float(value)

#         return encoded_clinical


# # Convenience functions
# def extract_audio_features(
#     audio_input: Union[str, Path, bytes, io.BytesIO], validate_quality: bool = True
# ) -> np.ndarray:
#     """
#     Quick function to extract audio features with quality validation.

#     Args:
#         audio_input: Audio file path, bytes, or BytesIO
#         validate_quality: Whether to validate audio quality

#     Returns:
#         Feature embedding array

#     Raises:
#         AudioQualityError: If audio quality is below threshold
#     """
#     preprocessor = AudioPreprocessor()
#     return preprocessor.extract_features(audio_input, validate_quality=validate_quality)


# def generate_spectrogram(
#     audio_input: Union[str, Path, bytes, io.BytesIO],
# ) -> Image.Image:
#     """
#     Quick function to generate spectrogram.

#     Args:
#         audio_input: Audio file path, bytes, or BytesIO

#     Returns:
#         PIL Image of spectrogram
#     """
#     preprocessor = AudioPreprocessor()
#     return preprocessor.generate_spectrogram(audio_input)


# def validate_audio_quality(
#     audio_input: Union[str, Path, bytes, io.BytesIO],
# ) -> Tuple[bool, Optional[str], Dict[str, float]]:
#     """
#     Quick function to validate audio quality.

#     Args:
#         audio_input: Audio file path, bytes, or BytesIO

#     Returns:
#         Tuple of (is_valid, error_message, metrics)
#     """
#     preprocessor = AudioPreprocessor()
#     return preprocessor.validate_audio_quality(audio_input)


# if __name__ == "__main__":
#     # Test audio preprocessing
#     preprocessor = AudioPreprocessor()

#     logger.info("AudioPreprocessor ready for testing")
#     logger.info("Use extract_features() or generate_spectrogram() methods")


# """
# Preprocessor module for TBFusionAI.

# Contains:
# 1. AudioPreprocessor: Audio file preprocessing and feature extraction
# 2. FeaturePreprocessor: Feature encoding and scaling
# """

# import io
# from pathlib import Path
# from typing import Dict, Optional, Tuple, Union

# import librosa
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# import soundfile as sf
# import torch
# import torchaudio
# from PIL import Image
# from scipy.signal import butter, lfilter
# from transformers import Wav2Vec2Model, Wav2Vec2Processor

# from src.config import get_config
# from src.logger import get_logger

# logger = get_logger(__name__)


# class AudioPreprocessor:
#     """
#     Audio preprocessing and feature extraction.

#     Handles:
#     - Audio loading and resampling
#     - Noise reduction and filtering
#     - Wav2Vec2 feature extraction
#     - Spectrogram generation
#     """

#     def __init__(self):
#         """Initialize the audio preprocessor."""
#         self.config = get_config()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Load Wav2Vec2 model
#         self.model: Optional[Wav2Vec2Model] = None
#         self.processor: Optional[Wav2Vec2Processor] = None

#         logger.info(f"AudioPreprocessor initialized on device: {self.device}")

#     def load_wav2vec2_model(self) -> None:
#         """Load Wav2Vec2 model and processor."""
#         if self.model is not None:
#             return  # Already loaded

#         logger.info("Loading Wav2Vec2 model...")

#         cache_dir = self.config.paths.artifacts_path / "model_cache"
#         cache_dir.mkdir(parents=True, exist_ok=True)

#         model_name = self.config.audio_extraction.model_name

#         try:
#             self.processor = Wav2Vec2Processor.from_pretrained(
#                 model_name,
#                 cache_dir=str(cache_dir),
#                 local_files_only=False
#             )

#             self.model = Wav2Vec2Model.from_pretrained(
#                 model_name,
#                 cache_dir=str(cache_dir),
#                 local_files_only=False
#             )

#             self.model.eval()
#             self.model = self.model.to(self.device)

#             logger.info(f"✓ Wav2Vec2 model loaded successfully")

#         except Exception as e:
#             logger.error(f"Failed to load Wav2Vec2 model: {str(e)}")
#             raise

#     def load_audio(
#         self,
#         audio_input: Union[str, Path, bytes, io.BytesIO]
#     ) -> Tuple[torch.Tensor, int]:
#         """
#         Load audio from various input types.

#         Args:
#             audio_input: File path, bytes, or BytesIO object

#         Returns:
#             Tuple of (waveform tensor, sample rate)
#         """
#         try:
#             # Handle different input types
#             if isinstance(audio_input, (str, Path)):
#                 waveform, sr = torchaudio.load(str(audio_input))
#             elif isinstance(audio_input, bytes):
#                 buffer = io.BytesIO(audio_input)
#                 buffer.seek(0)
#                 waveform, sr = torchaudio.load(buffer)
#             elif isinstance(audio_input, io.BytesIO):
#                 audio_input.seek(0)
#                 waveform, sr = torchaudio.load(audio_input)
#             else:
#                 raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

#             # Resample to 16kHz if needed
#             if sr != self.config.audio_extraction.sample_rate:
#                 resampler = torchaudio.transforms.Resample(
#                     sr,
#                     self.config.audio_extraction.sample_rate
#                 )
#                 waveform = resampler(waveform)
#                 sr = self.config.audio_extraction.sample_rate

#             return waveform, sr

#         except Exception as e:
#             logger.error(f"Failed to load audio: {str(e)}")
#             raise

#     def extract_features(
#         self,
#         audio_input: Union[str, Path, bytes, io.BytesIO]
#     ) -> np.ndarray:
#         """
#         Extract Wav2Vec2 features from audio.

#         Args:
#             audio_input: Audio file path, bytes, or BytesIO

#         Returns:
#             Feature embedding as numpy array (768-dimensional)
#         """
#         # Ensure model is loaded
#         if self.model is None:
#             self.load_wav2vec2_model()

#         # Load audio
#         waveform, sr = self.load_audio(audio_input)

#         # Preprocess audio
#         inputs = self.processor(
#             waveform.squeeze(),
#             sampling_rate=sr,
#             return_tensors="pt",
#             padding=True
#         )

#         # Move to device
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         # Extract features
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             last_hidden_state = outputs.last_hidden_state
#             embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

#         if embedding.ndim == 1:
#             embedding = embedding.reshape(1, -1)[0]

#         return embedding

#     def apply_bandpass_filter(
#         self,
#         waveform: np.ndarray,
#         sr: int
#     ) -> np.ndarray:
#         """
#         Apply Butterworth bandpass filter.

#         Args:
#             waveform: Audio waveform
#             sr: Sample rate

#         Returns:
#             Filtered waveform
#         """
#         nyq = 0.5 * sr
#         low = self.config.audio_preprocessing.lowcut / nyq
#         high = self.config.audio_preprocessing.highcut / nyq

#         b, a = butter(
#             self.config.audio_preprocessing.filter_order,
#             [low, high],
#             btype='band'
#         )

#         return lfilter(b, a, waveform)

#     def generate_spectrogram(
#         self,
#         audio_input: Union[str, Path, bytes, io.BytesIO]
#     ) -> Image.Image:
#         """
#         Generate Mel spectrogram image.

#         Args:
#             audio_input: Audio file path, bytes, or BytesIO

#         Returns:
#             PIL Image of spectrogram
#         """
#         # Load audio
#         waveform, sr = self.load_audio(audio_input)
#         y = waveform.squeeze().numpy()

#         # Compute Mel spectrogram
#         S = librosa.feature.melspectrogram(
#             y=y,
#             sr=sr,
#             n_mels=self.config.audio_preprocessing.n_mels
#         )
#         S_db = librosa.power_to_db(S, ref=np.max)

#         # Plot
#         fig, ax = plt.subplots(
#             figsize=self.config.audio_preprocessing.spectrogram_figsize
#         )

#         librosa.display.specshow(
#             S_db,
#             sr=sr,
#             x_axis='time',
#             y_axis='mel',
#             cmap='magma',
#             ax=ax
#         )

#         ax.axis('off')
#         plt.tight_layout(pad=0)

#         # Convert to PIL Image
#         buf = io.BytesIO()
#         plt.savefig(
#             buf,
#             format='png',
#             bbox_inches='tight',
#             pad_inches=0,
#             dpi=self.config.audio_preprocessing.spectrogram_dpi
#         )
#         plt.close(fig)

#         buf.seek(0)
#         image = Image.open(buf)

#         return image

#     def calculate_audio_metrics(
#         self,
#         audio_input: Union[str, Path, bytes, io.BytesIO]
#     ) -> Dict[str, float]:
#         """
#         Calculate audio quality metrics.

#         Args:
#             audio_input: Audio file path, bytes, or BytesIO

#         Returns:
#             Dictionary of audio metrics
#         """
#         waveform, sr = self.load_audio(audio_input)
#         y = waveform.squeeze().numpy()

#         duration = librosa.get_duration(y=y, sr=sr)
#         rms = librosa.feature.rms(y=y).mean()
#         zcr = librosa.feature.zero_crossing_rate(y).mean()
#         flatness = librosa.feature.spectral_flatness(y=y).mean()

#         # Estimate SNR
#         try:
#             harmonic, _ = librosa.effects.hpss(y)
#             signal_power = np.mean(y ** 2)
#             noise = y - harmonic
#             noise_power = np.mean(noise ** 2)
#             snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
#         except Exception:
#             snr = 0.0

#         return {
#             'duration': float(duration),
#             'rms': float(rms),
#             'zcr': float(zcr),
#             'flatness': float(flatness),
#             'snr': float(snr)
#         }


# class FeaturePreprocessor:
#     """
#     Feature encoding and preprocessing.

#     Handles:
#     - Categorical feature encoding
#     - Binary feature encoding
#     - Feature validation
#     """

#     def __init__(self):
#         """Initialize the feature preprocessor."""
#         self.config = get_config()
#         logger.info("FeaturePreprocessor initialized")

#     def encode_features(
#         self,
#         features: Dict[str, Union[int, float, str]]
#     ) -> Dict[str, Union[int, float]]:
#         """
#         Encode categorical and binary features.

#         Args:
#             features: Raw feature dictionary

#         Returns:
#             Encoded feature dictionary
#         """
#         encoded = features.copy()

#         # Encode sex
#         if 'sex' in encoded:
#             if isinstance(encoded['sex'], str):
#                 encoded['sex'] = 1 if encoded['sex'].lower() == 'male' else 0

#         # Encode binary features
#         binary_features = self.config.metadata.binary_features
#         for feature in binary_features:
#             if feature in encoded:
#                 if isinstance(encoded[feature], str):
#                     encoded[feature] = 1 if encoded[feature].lower() == 'yes' else 0

#         return encoded

#     def validate_features(
#         self,
#         features: Dict[str, Union[int, float, str]]
#     ) -> Tuple[bool, Optional[str]]:
#         """
#         Validate required features are present.

#         Args:
#             features: Feature dictionary

#         Returns:
#             Tuple of (is_valid, error_message)
#         """
#         required_clinical = self.config.metadata.clinical_features

#         # Check for required clinical features
#         for feature in required_clinical:
#             if feature not in features:
#                 return False, f"Missing required feature: {feature}"

#         # Check for audio features
#         audio_features = [k for k in features.keys() if k.startswith('feat_')]
#         if len(audio_features) == 0:
#             return False, "Missing audio features. Audio must be processed first."

#         return True, None

#     def prepare_feature_dict(
#         self,
#         clinical_features: Dict[str, Union[int, float, str]],
#         audio_features: np.ndarray
#     ) -> Dict[str, Union[int, float]]:
#         """
#         Combine clinical and audio features into single dictionary.

#         Args:
#             clinical_features: Clinical feature dictionary
#             audio_features: Audio feature array (768-dimensional)

#         Returns:
#             Combined feature dictionary
#         """
#         # Encode clinical features
#         encoded_clinical = self.encode_features(clinical_features)

#         # Add audio features
#         for i, value in enumerate(audio_features):
#             encoded_clinical[f'feat_{i}'] = float(value)

#         return encoded_clinical


# # Convenience functions
# def extract_audio_features(
#     audio_input: Union[str, Path, bytes, io.BytesIO]
# ) -> np.ndarray:
#     """
#     Quick function to extract audio features.

#     Args:
#         audio_input: Audio file path, bytes, or BytesIO

#     Returns:
#         Feature embedding array
#     """
#     preprocessor = AudioPreprocessor()
#     return preprocessor.extract_features(audio_input)


# def generate_spectrogram(
#     audio_input: Union[str, Path, bytes, io.BytesIO]
# ) -> Image.Image:
#     """
#     Quick function to generate spectrogram.

#     Args:
#         audio_input: Audio file path, bytes, or BytesIO

#     Returns:
#         PIL Image of spectrogram
#     """
#     preprocessor = AudioPreprocessor()
#     return preprocessor.generate_spectrogram(audio_input)


# if __name__ == "__main__":
#     # Test audio preprocessing
#     preprocessor = AudioPreprocessor()

#     logger.info("AudioPreprocessor ready for testing")
#     logger.info("Use extract_features() or generate_spectrogram() methods")
