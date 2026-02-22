"""
Tests for utility modules.

Tests:
- Validators
- Helpers
- Common functions
"""

from pathlib import Path

import numpy as np
import pytest

from src.utils.helpers import (
    batch_iterator,
    calculate_metrics,
    calculate_percentage,
    create_directory,
    format_duration,
    get_file_size,
    safe_divide,
)
from src.utils.validators import (
    validate_audio_format,
    validate_batch_size,
    validate_clinical_features,
    validate_feature_array,
    validate_file_path,
    validate_probability,
    validate_threshold,
)


class TestValidators:
    """Tests for validation functions."""

    def test_validate_clinical_features_valid(self, sample_clinical_features):
        """Test validation with valid features."""
        is_valid, error = validate_clinical_features(sample_clinical_features)

        assert is_valid
        assert error is None

    def test_validate_clinical_features_missing(self):
        """Test validation with missing features."""
        incomplete_features = {"age": 45}

        is_valid, error = validate_clinical_features(incomplete_features)

        assert not is_valid
        assert error is not None
        assert "Missing required features" in error

    def test_validate_clinical_features_invalid_age(self):
        """Test validation with invalid age."""
        invalid_features = {
            "age": 150,  # Invalid
            "sex": "Male",
            "reported_cough_dur": 21,
            "tb_prior": "No",
            "hemoptysis": "No",
            "weight_loss": "Yes",
            "fever": "Yes",
            "night_sweats": "No",
        }

        is_valid, error = validate_clinical_features(invalid_features)

        assert not is_valid
        assert "Age" in error

    def test_validate_clinical_features_invalid_sex(self):
        """Test validation with invalid sex."""
        invalid_features = {
            "age": 45,
            "sex": "Invalid",  # Invalid
            "reported_cough_dur": 21,
            "tb_prior": "No",
            "hemoptysis": "No",
            "weight_loss": "Yes",
            "fever": "Yes",
            "night_sweats": "No",
        }

        is_valid, error = validate_clinical_features(invalid_features)

        assert not is_valid
        assert "Sex" in error

    def test_validate_feature_array_valid(self):
        """Test validation with valid feature array."""
        X = np.random.randn(10, 786)

        is_valid, error = validate_feature_array(X, expected_features=786)

        assert is_valid
        assert error is None

    def test_validate_feature_array_wrong_shape(self):
        """Test validation with wrong shape."""
        X = np.random.randn(10)  # 1D instead of 2D

        is_valid, error = validate_feature_array(X)

        assert not is_valid
        assert "2D array" in error

    def test_validate_feature_array_nan(self):
        """Test validation with NaN values."""
        X = np.random.randn(10, 786)
        X[0, 0] = np.nan

        is_valid, error = validate_feature_array(X, expected_features=786)

        assert not is_valid
        assert "NaN" in error

    def test_validate_feature_array_inf(self):
        """Test validation with infinite values."""
        X = np.random.randn(10, 786)
        X[0, 0] = np.inf

        is_valid, error = validate_feature_array(X, expected_features=786)

        assert not is_valid
        assert "infinite" in error

    def test_validate_probability_valid(self):
        """Test validation with valid probability."""
        is_valid, error = validate_probability(0.75)

        assert is_valid
        assert error is None

    def test_validate_probability_out_of_range(self):
        """Test validation with out-of-range probability."""
        is_valid, error = validate_probability(1.5)

        assert not is_valid
        assert "between" in error

    def test_validate_probability_nan(self):
        """Test validation with NaN probability."""
        is_valid, error = validate_probability(np.nan)

        assert not is_valid
        assert "NaN" in error

    def test_validate_threshold_valid(self):
        """Test validation with valid threshold."""
        is_valid, error = validate_threshold(0.45)

        assert is_valid
        assert error is None

    def test_validate_batch_size_valid(self):
        """Test validation with valid batch size."""
        is_valid, error = validate_batch_size(32)

        assert is_valid
        assert error is None

    def test_validate_batch_size_too_small(self):
        """Test validation with too small batch size."""
        is_valid, error = validate_batch_size(0)

        assert not is_valid
        assert "at least" in error

    def test_validate_file_path_existing(self, temp_dir):
        """Test validation with existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        is_valid, error = validate_file_path(test_file, must_exist=True)

        assert is_valid
        assert error is None

    def test_validate_file_path_nonexistent(self, temp_dir):
        """Test validation with non-existent file."""
        test_file = temp_dir / "nonexistent.txt"

        is_valid, error = validate_file_path(test_file, must_exist=True)

        assert not is_valid
        assert "does not exist" in error


class TestHelpers:
    """Tests for helper functions."""

    def test_create_directory(self, temp_dir):
        """Test directory creation."""
        test_dir = temp_dir / "new_directory"

        created_dir = create_directory(test_dir)

        assert created_dir.exists()
        assert created_dir.is_dir()

    def test_create_directory_exists(self, temp_dir):
        """Test directory creation when already exists."""
        test_dir = temp_dir / "existing_directory"
        test_dir.mkdir()

        created_dir = create_directory(test_dir)

        assert created_dir.exists()
        assert created_dir.is_dir()

    def test_get_file_size(self, temp_dir):
        """Test file size calculation."""
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!" * 100
        test_file.write_text(test_content)

        size_bytes = get_file_size(test_file, unit="B")
        size_kb = get_file_size(test_file, unit="KB")

        assert size_bytes > 0
        assert size_kb == size_bytes / 1024

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        formatted = format_duration(45.5)

        assert "s" in formatted
        assert "45" in formatted

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        formatted = format_duration(125.0)

        assert "m" in formatted

    def test_format_duration_hours(self):
        """Test duration formatting for hours."""
        formatted = format_duration(7200.0)

        assert "h" in formatted

    def test_safe_divide_normal(self):
        """Test safe division with normal values."""
        result = safe_divide(10, 2)

        assert result == 5.0

    def test_safe_divide_by_zero(self):
        """Test safe division by zero."""
        result = safe_divide(10, 0)

        assert result == 0.0

    def test_safe_divide_by_zero_custom_default(self):
        """Test safe division by zero with custom default."""
        result = safe_divide(10, 0, default=-1.0)

        assert result == -1.0

    def test_calculate_percentage(self):
        """Test percentage calculation."""
        percentage = calculate_percentage(25, 100)

        assert percentage == 25.0

    def test_calculate_percentage_zero_total(self):
        """Test percentage with zero total."""
        percentage = calculate_percentage(10, 0)

        assert percentage == 0.0

    def test_calculate_metrics(self):
        """Test metrics calculation from confusion matrix."""
        metrics = calculate_metrics(tn=80, fp=10, fn=5, tp=90)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Check value ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_batch_iterator(self):
        """Test batch iterator."""
        data = list(range(100))
        batch_size = 10

        batches = list(batch_iterator(data, batch_size))

        assert len(batches) == 10
        assert len(batches[0]) == batch_size
        assert batches[0] == list(range(10))
        assert batches[-1] == list(range(90, 100))
