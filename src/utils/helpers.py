"""
Helper utilities for TBFusionAI.

Provides utility functions for:
- File operations
- Data formatting
- Common calculations
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def create_directory(
    directory: Union[str, Path], exist_ok: bool = True, parents: bool = True
) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Directory path
        exist_ok: Whether to ignore if directory exists
        parents: Whether to create parent directories

    Returns:
        Path: Created directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=parents, exist_ok=exist_ok)
    logger.debug(f"Directory created/verified: {directory}")
    return directory


def get_file_size(file_path: Union[str, Path], unit: str = "MB") -> float:
    """
    Get file size in specified unit.

    Args:
        file_path: Path to file
        unit: Unit for size (B, KB, MB, GB)

    Returns:
        float: File size in specified unit
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = file_path.stat().st_size

    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}

    if unit not in units:
        raise ValueError(f"Invalid unit: {unit}. Use: {list(units.keys())}")

    return size_bytes / units[unit]


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_checksum(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate file checksum.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        str: Hexadecimal checksum string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get hash function
    if algorithm == "md5":
        hash_func = hashlib.md5()
    elif algorithm == "sha1":
        hash_func = hashlib.sha1()
    elif algorithm == "sha256":
        hash_func = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Calculate checksum
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def save_json(
    data: Union[Dict, List], file_path: Union[str, Path], indent: int = 2
) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)

    logger.debug(f"Saved JSON to: {file_path}")


def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
    """
    Load data from JSON file.

    Args:
        file_path: Input file path

    Returns:
        Loaded data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from: {file_path}")
    return data


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Converted object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def ensure_list(value: Any) -> List:
    """
    Ensure value is a list.

    Args:
        value: Value to convert

    Returns:
        List containing the value(s)
    """
    if isinstance(value, list):
        return value
    elif isinstance(value, (tuple, set)):
        return list(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    else:
        return [value]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails

    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def calculate_percentage(part: float, total: float, decimal_places: int = 2) -> float:
    """
    Calculate percentage.

    Args:
        part: Part value
        total: Total value
        decimal_places: Number of decimal places

    Returns:
        Percentage value
    """
    percentage = safe_divide(part, total, 0.0) * 100
    return round(percentage, decimal_places)


def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """
    Merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (overwrites dict1)
        deep: Whether to perform deep merge

    Returns:
        Merged dictionary
    """
    if not deep:
        return {**dict1, **dict2}

    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value

    return result


def flatten_dict(d: Dict, parent_key: str = "", separator: str = "_") -> Dict:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        separator: Separator between keys

    Returns:
        Flattened dictionary
    """
    items = []

    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))

    return dict(items)


def batch_iterator(data: List, batch_size: int):
    """
    Create batches from list.

    Args:
        data: List of data
        batch_size: Size of each batch

    Yields:
        Batches of data
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def format_confusion_matrix(tn: int, fp: int, fn: int, tp: int) -> str:
    """
    Format confusion matrix as string.

    Args:
        tn: True negatives
        fp: False positives
        fn: False negatives
        tp: True positives

    Returns:
        Formatted confusion matrix string
    """
    total = tn + fp + fn + tp

    return f"""
Confusion Matrix:
                Predicted Negative  Predicted Positive
Actual Negative        {tn:6d}              {fp:6d}
Actual Positive        {fn:6d}              {tp:6d}

Total Samples: {total}
Accuracy: {calculate_percentage(tn + tp, total)}%
"""


def calculate_metrics(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    """
    Calculate classification metrics from confusion matrix.

    Args:
        tn: True negatives
        fp: False positives
        fn: False negatives
        tp: True positives

    Returns:
        Dictionary of metrics
    """
    total = tn + fp + fn + tp

    accuracy = safe_divide(tn + tp, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1_score, 4),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def log_metrics(metrics: Dict[str, Union[int, float]]) -> None:
    """
    Log metrics in formatted way.

    Args:
        metrics: Dictionary of metrics
    """
    logger.info("\n" + "=" * 50)
    logger.info("METRICS")
    logger.info("=" * 50)

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key:20s}: {value:.4f}")
        else:
            logger.info(f"  {key:20s}: {value}")

    logger.info("=" * 50)
