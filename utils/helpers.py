"""
Helper Utilities
================

General-purpose helper functions used across services and the UI layer.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import EXPORT_TIMESTAMP_FORMAT


def ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """
    Sanitize a filename by removing or replacing unsafe characters.

    Strips path components so only the base name is returned.
    """
    base = Path(name).name
    unsafe_chars = '<>:"/\\|?*'
    sanitized = "".join(c if c not in unsafe_chars else "_" for c in base)
    sanitized = sanitized.strip().strip(".")
    return sanitized or "unnamed_file"


def format_file_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. '1.5 MB')."""
    if size_bytes < 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def generate_stored_filename(original_filename: str) -> str:
    """
    Build a unique on-disk filename using a timestamp prefix.

    Example: 20250603_143022_sales_data.xlsx
    """
    safe_name = safe_filename(original_filename)
    timestamp = datetime.now().strftime(EXPORT_TIMESTAMP_FORMAT)
    return f"{timestamp}_{safe_name}"


def preview_dataframe(df: pd.DataFrame, n_rows: int = 20) -> pd.DataFrame:
    """Return the first n rows for display in the UI."""
    return df.head(n_rows).copy()


def compute_dataset_health(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute inspection metrics for dataset health (no cleaning applied).

    Returns:
        dict with missing_values, duplicate_rows, and data_types sections.
    """
    missing_per_column = df.isna().sum()
    missing_display = {
        str(col): int(count) for col, count in missing_per_column.items()
    }
    total_missing = int(missing_per_column.sum())

    duplicate_count = int(df.duplicated().sum())

    dtype_display = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "missing_values": {
            "total": total_missing,
            "by_column": missing_display,
        },
        "duplicate_rows": duplicate_count,
        "data_types": dtype_display,
    }
