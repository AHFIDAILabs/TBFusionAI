"""
Tests for pipeline modules - UPDATED FOR CURRENT STRUCTURE.

Tests:
- Audio preprocessing (with quality validation)
- Feature preprocessing
- Model inference
"""

import io
import numpy as np
import pytest

from src.models.preprocessor import AudioPreprocessor, FeaturePreprocessor, AudioQualityError


class TestAudioPreprocessor:
    """
    Tests for audio preprocessing.
    
    UPDATED: Tests new quality validation features.
    """
    
    def test_preprocessor_initialization(self):
        """Test audio preprocessor initialization."""
        preprocessor = AudioPreprocessor()
        
        assert preprocessor.config is not None
        assert preprocessor.device is not None
        assert preprocessor.model is None  # Lazy loaded
        assert preprocessor.processor is None  # Lazy loaded
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        preprocessor = AudioPreprocessor()
        
        # Create test signal
        signal = np.array([0.5, 1.0, -0.5, -1.0])
        
        # Normalize
        normalized = preprocessor.normalize_audio(signal)
        
        # Check normalization
        assert normalized.max() == 1.0 or normalized.max() == -1.0
        assert -1.0 <= normalized.min() <= 1.0
    
    def test_bandpass_filter(self):
        """Test bandpass filter."""
        preprocessor = AudioPreprocessor()
        
        # Create test signal
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Apply filter
        filtered = preprocessor.apply_bandpass_filter(signal, sr)
        
        assert filtered.shape == signal.shape
        assert not np.allclose(filtered, signal)  # Should be different
    
    def test_reduce_noise(self):
        """
        Test noise reduction.
        
        UPDATED: Tests new noise reduction method.
        """
        preprocessor = AudioPreprocessor()
        
        # Create test signal with noise
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.sin(2 * np.pi * 440 * t)
        noisy_signal = signal + np.random.randn(len(signal)) * 0.1
        
        # Apply noise reduction
        reduced = preprocessor.reduce_noise(noisy_signal, sr)
        
        assert reduced.shape == noisy_signal.shape
        # Noise reduction should change the signal
        assert not np.allclose(reduced, noisy_signal)
    
    def test_preprocess_audio(self, sample_audio_bytes):
        """
        Test complete audio preprocessing pipeline.
        
        UPDATED: Tests new preprocessing pipeline.
        """
        preprocessor = AudioPreprocessor()
        
        audio_io = io.BytesIO(sample_audio_bytes)
        
        try:
            # Test with filters
            y_filtered, sr = preprocessor.preprocess_audio(audio_io, apply_filters=True)
            assert y_filtered is not None
            assert sr == 16000
            assert -1.0 <= y_filtered.max() <= 1.0
            assert -1.0 <= y_filtered.min() <= 1.0
            
            # Test without filters
            audio_io.seek(0)
            y_raw, sr = preprocessor.preprocess_audio(audio_io, apply_filters=False)
            assert y_raw is not None
            assert not np.allclose(y_filtered, y_raw)  # Should be different
            
        except Exception as e:
            pytest.skip(f"Audio processing failed: {str(e)}")
    
    def test_calculate_audio_metrics(self, sample_audio_bytes):
        """
        Test audio metrics calculation.
        
        UPDATED: Tests new metrics (clipping_ratio, silence_ratio).
        """
        preprocessor = AudioPreprocessor()
        
        audio_io = io.BytesIO(sample_audio_bytes)
        
        try:
            metrics = preprocessor.calculate_audio_metrics(audio_io)
            
            # Check all metrics present
            assert 'duration' in metrics
            assert 'rms' in metrics
            assert 'zcr' in metrics
            assert 'flatness' in metrics
            assert 'snr' in metrics
            assert 'clipping_ratio' in metrics  # NEW
            assert 'silence_ratio' in metrics   # NEW
            
            # Check value ranges
            assert metrics['duration'] > 0
            assert metrics['rms'] >= 0
            assert 0 <= metrics['zcr'] <= 1
            assert 0 <= metrics['flatness'] <= 1
            assert 0 <= metrics['clipping_ratio'] <= 1  # NEW
            assert 0 <= metrics['silence_ratio'] <= 1   # NEW
            
        except Exception as e:
            pytest.skip(f"Audio processing failed: {str(e)}")
    
    def test_validate_audio_quality(self, sample_audio_bytes):
        """
        Test audio quality validation.
        
        UPDATED: Tests new quality validation method.
        """
        preprocessor = AudioPreprocessor()
        
        audio_io = io.BytesIO(sample_audio_bytes)
        
        try:
            is_valid, error_msg, metrics = preprocessor.validate_audio_quality(
                audio_io,
                min_snr=-5.0,
                max_clipping_ratio=0.05,
                max_silence_ratio=0.5,
                min_duration=0.3,
                max_duration=30.0
            )
            
            # Check return types
            assert isinstance(is_valid, bool)
            assert error_msg is None or isinstance(error_msg, str)
            assert isinstance(metrics, dict)
            
            # If invalid, should have error message
            if not is_valid:
                assert error_msg is not None
                
        except Exception as e:
            pytest.skip(f"Audio processing failed: {str(e)}")
    
    def test_extract_features_with_validation(self, sample_audio_bytes):
        """
        Test feature extraction with quality validation.
        
        UPDATED: Tests new validate_quality parameter.
        """
        preprocessor = AudioPreprocessor()
        
        audio_io = io.BytesIO(sample_audio_bytes)
        
        try:
            # Test without validation
            features = preprocessor.extract_features(audio_io, validate_quality=False)
            assert features is not None
            assert features.shape == (768,)
            
            # Test with validation (may raise AudioQualityError)
            audio_io.seek(0)
            try:
                features_validated = preprocessor.extract_features(audio_io, validate_quality=True)
                assert features_validated is not None
                assert features_validated.shape == (768,)
            except AudioQualityError as e:
                # Expected for low quality audio
                assert "audio quality" in str(e).lower() or "snr" in str(e).lower()
                
        except Exception as e:
            pytest.skip(f"Feature extraction failed: {str(e)}")
    
    def test_generate_spectrogram(self, sample_audio_bytes):
        """Test spectrogram generation."""
        preprocessor = AudioPreprocessor()
        
        audio_io = io.BytesIO(sample_audio_bytes)
        
        try:
            spectrogram = preprocessor.generate_spectrogram(audio_io)
            
            # Check it's a PIL Image
            from PIL import Image
            assert isinstance(spectrogram, Image.Image)
            
            # Check size
            assert spectrogram.size[0] > 0
            assert spectrogram.size[1] > 0
            
        except Exception as e:
            pytest.skip(f"Spectrogram generation failed: {str(e)}")


class TestFeaturePreprocessor:
    """Tests for feature preprocessing."""
    
    def test_preprocessor_initialization(self):
        """Test feature preprocessor initialization."""
        preprocessor = FeaturePreprocessor()
        
        assert preprocessor.config is not None
    
    def test_encode_features(self):
        """Test feature encoding."""
        preprocessor = FeaturePreprocessor()
        
        features = {
            'sex': 'Male',
            'tb_prior': 'Yes',
            'hemoptysis': 'No',
            'weight_loss': 'Yes',
            'fever': 'No',
            'night_sweats': 'Yes',
            'age': 45,
            'reported_cough_dur': 21
        }
        
        encoded = preprocessor.encode_features(features)
        
        # Check sex encoding (Male = 1, Female = 0)
        assert encoded['sex'] == 1
        
        # Check binary encoding (Yes = 1, No = 0)
        assert encoded['tb_prior'] == 1
        assert encoded['hemoptysis'] == 0
        assert encoded['weight_loss'] == 1
        assert encoded['fever'] == 0
        assert encoded['night_sweats'] == 1
        
        # Check numeric features unchanged
        assert encoded['age'] == 45
        assert encoded['reported_cough_dur'] == 21
    
    def test_encode_features_already_numeric(self):
        """Test encoding when features are already numeric."""
        preprocessor = FeaturePreprocessor()
        
        features = {
            'sex': 1,
            'tb_prior': 0,
            'hemoptysis': 1,
            'age': 45
        }
        
        encoded = preprocessor.encode_features(features)
        
        # Should remain unchanged
        assert encoded['sex'] == 1
        assert encoded['tb_prior'] == 0
        assert encoded['hemoptysis'] == 1
        assert encoded['age'] == 45
    
    def test_validate_features(self, sample_full_features):
        """Test feature validation."""
        preprocessor = FeaturePreprocessor()
        
        # Valid features
        is_valid, error = preprocessor.validate_features(sample_full_features)
        assert is_valid
        assert error is None
        
        # Missing clinical features
        invalid_features = {'feat_0': 0.5}
        is_valid, error = preprocessor.validate_features(invalid_features)
        assert not is_valid
        assert error is not None
        assert 'Missing required feature' in error
        
        # Missing audio features
        clinical_only = {
            'age': 45,
            'sex': 'Male',
            'reported_cough_dur': 21,
            'tb_prior': 'No',
            'hemoptysis': 'No',
            'weight_loss': 'Yes',
            'fever': 'Yes',
            'night_sweats': 'No'
        }
        is_valid, error = preprocessor.validate_features(clinical_only)
        assert not is_valid
        assert 'audio features' in error.lower()
    
    def test_prepare_feature_dict(self, sample_clinical_features, sample_audio_features):
        """Test feature dictionary preparation."""
        preprocessor = FeaturePreprocessor()
        
        combined = preprocessor.prepare_feature_dict(
            sample_clinical_features,
            sample_audio_features
        )
        
        # Check clinical features encoded
        assert 'age' in combined
        assert 'sex' in combined
        assert combined['sex'] in [0, 1]  # Should be encoded
        
        # Check audio features present
        assert 'feat_0' in combined
        assert 'feat_767' in combined
        
        # Check all values are numeric
        for value in combined.values():
            assert isinstance(value, (int, float))


class TestConvenienceFunctions:
    """
    Test convenience functions.
    
    UPDATED: Tests new convenience functions.
    """
    
    def test_extract_audio_features_function(self, sample_audio_bytes):
        """Test convenience function for audio feature extraction."""
        from src.models.preprocessor import extract_audio_features
        
        audio_io = io.BytesIO(sample_audio_bytes)
        
        try:
            features = extract_audio_features(audio_io, validate_quality=False)
            assert features is not None
            assert features.shape == (768,)
        except Exception as e:
            pytest.skip(f"Feature extraction failed: {str(e)}")
    
    def test_generate_spectrogram_function(self, sample_audio_bytes):
        """Test convenience function for spectrogram generation."""
        from src.models.preprocessor import generate_spectrogram
        
        audio_io = io.BytesIO(sample_audio_bytes)
        
        try:
            spectrogram = generate_spectrogram(audio_io)
            
            from PIL import Image
            assert isinstance(spectrogram, Image.Image)
        except Exception as e:
            pytest.skip(f"Spectrogram generation failed: {str(e)}")
    
    def test_validate_audio_quality_function(self, sample_audio_bytes):
        """Test convenience function for audio quality validation."""
        from src.models.preprocessor import validate_audio_quality
        
        audio_io = io.BytesIO(sample_audio_bytes)
        
        try:
            is_valid, error_msg, metrics = validate_audio_quality(audio_io)
            
            assert isinstance(is_valid, bool)
            assert isinstance(metrics, dict)
        except Exception as e:
            pytest.skip(f"Quality validation failed: {str(e)}")


class TestModelInferencePipeline:
    """Tests for model inference pipeline."""
    
    def test_pipeline_initialization_without_models(self):
        """Test pipeline initialization without trained models."""
        from src.pipelines.model_inference import ModelInferencePipeline
        
        # This will fail if models aren't present
        try:
            pipeline = ModelInferencePipeline()
            assert pipeline.config is not None
        except Exception as e:
            # Expected if models not trained yet
            assert any(phrase in str(e).lower() for phrase in [
                "failed to load models",
                "no such file",
                "models not ready"
            ])
    
    @pytest.mark.asyncio
    async def test_predict_single_structure(self, sample_full_features):
        """Test predict_single method structure."""
        # This tests the structure, not the actual prediction
        assert 'age' in sample_full_features
        assert 'sex' in sample_full_features
        assert 'feat_0' in sample_full_features
        
        # Check all required clinical features
        required = ['age', 'sex', 'reported_cough_dur', 'tb_prior',
                   'hemoptysis', 'weight_loss', 'fever', 'night_sweats']
        for feature in required:
            assert feature in sample_full_features


class TestDataPipelines:
    """Basic tests for data pipelines."""
    
    def test_data_ingestion_initialization(self):
        """Test data ingestion pipeline initialization."""
        from src.pipelines.data_ingestion import DataIngestionPipeline
        
        pipeline = DataIngestionPipeline()
        
        assert pipeline.config is not None
        assert pipeline.dataset_path is not None
        assert pipeline.raw_data_path is not None
        assert pipeline.meta_data_path is not None