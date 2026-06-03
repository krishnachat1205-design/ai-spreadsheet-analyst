"""
Cleaning Service
================

Data quality and normalization operations that transform raw uploads
into analysis-ready DataFrames.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from core.config import NULL_PLACEHOLDERS

logger = logging.getLogger(__name__)

MissingValueStrategy = Literal["drop", "mean", "median", "mode", "smart"]

# Minimum share of non-null parsed values required to coerce a column type.
_TYPE_CONVERSION_THRESHOLD = 0.6


@dataclass
class CleaningResult:
    """Outcome of a single cleaning operation."""

    dataframe: pd.DataFrame
    affected_rows: int = 0
    details: str = ""


class CleaningServiceError(Exception):
    """Raised when a cleaning operation cannot be completed."""


def _replace_null_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Treat configured placeholder strings as missing values."""
    result = df.copy()
    object_cols = result.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        result[col] = result[col].replace(list(NULL_PLACEHOLDERS), np.nan)
    return result


def _is_empty_value(value) -> bool:
    """Return True when a cell should be treated as empty."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _standardize_single_column_name(name: str) -> str:
    """
    Normalize a column name to snake_case.

    Examples:
        Customer Name   -> customer_name
        Customer_Name   -> customer_name
        customer-name   -> customer_name
    """
    normalized = str(name).strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or "column"


def _count_empty_columns(df: pd.DataFrame) -> int:
    """Count columns where every value is null or blank."""
    count = 0
    for col in df.columns:
        if df[col].apply(_is_empty_value).all():
            count += 1
    return count


def generate_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Build a summary report of current dataset quality.

    Returns:
        dict with rows, columns, duplicates, missing_values, empty_columns.
    """
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "empty_columns": _count_empty_columns(df),
    }


def remove_duplicates(df: pd.DataFrame) -> CleaningResult:
    """Remove duplicate rows and report how many were dropped."""
    working = _replace_null_placeholders(df)
    before_count = len(working)
    cleaned = working.drop_duplicates().reset_index(drop=True)
    removed = before_count - len(cleaned)

    return CleaningResult(
        dataframe=cleaned,
        affected_rows=removed,
        details=f"Removed {removed} duplicate row(s).",
    )


def handle_missing_values(
    df: pd.DataFrame,
    strategy: MissingValueStrategy = "smart",
) -> CleaningResult:
    """
    Fill or drop missing values using the selected strategy.

    Strategies:
        drop   — remove rows with any missing value
        mean   — fill numeric columns with column mean
        median — fill numeric columns with column median
        mode   — fill columns with most frequent value
        smart  — numeric→median, text→mode, dates→forward fill
    """
    valid_strategies = ("drop", "mean", "median", "mode", "smart")
    if strategy not in valid_strategies:
        raise CleaningServiceError(
            f"Invalid strategy '{strategy}'. Choose from: {', '.join(valid_strategies)}."
        )

    working = _replace_null_placeholders(df.copy())
    affected = 0

    if strategy == "drop":
        before = len(working)
        cleaned = working.dropna().reset_index(drop=True)
        affected = before - len(cleaned)
        details = f"Dropped {affected} row(s) containing missing values."
        return CleaningResult(dataframe=cleaned, affected_rows=affected, details=details)

    if strategy in ("mean", "median"):
        cleaned = working.copy()
        numeric_cols = cleaned.select_dtypes(include="number").columns
        for col in numeric_cols:
            missing_mask = cleaned[col].isna()
            fill_count = int(missing_mask.sum())
            if fill_count == 0:
                continue
            fill_value = cleaned[col].mean() if strategy == "mean" else cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(fill_value)
            affected += fill_count
        details = f"Filled {affected} missing numeric cell(s) using {strategy}."
        return CleaningResult(dataframe=cleaned, affected_rows=affected, details=details)

    if strategy == "mode":
        cleaned = working.copy()
        for col in cleaned.columns:
            missing_mask = cleaned[col].isna()
            fill_count = int(missing_mask.sum())
            if fill_count == 0:
                continue
            modes = cleaned[col].mode(dropna=True)
            if modes.empty:
                continue
            cleaned[col] = cleaned[col].fillna(modes.iloc[0])
            affected += fill_count
        details = f"Filled {affected} missing cell(s) using mode."
        return CleaningResult(dataframe=cleaned, affected_rows=affected, details=details)

    # smart strategy
    cleaned = working.copy()
    for col in cleaned.columns:
        series = cleaned[col]
        missing_mask = series.isna()
        fill_count = int(missing_mask.sum())
        if fill_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(series):
            cleaned[col] = series.fillna(series.median())
            affected += fill_count
        elif pd.api.types.is_datetime64_any_dtype(series):
            cleaned[col] = series.ffill()
            affected += fill_count
        else:
            modes = series.mode(dropna=True)
            if not modes.empty:
                cleaned[col] = series.fillna(modes.iloc[0])
                affected += fill_count

    details = (
        "Applied smart imputation: numeric→median, text→mode, dates→forward fill. "
        f"Filled {affected} cell(s)."
    )
    return CleaningResult(dataframe=cleaned, affected_rows=affected, details=details)


def standardize_column_names(df: pd.DataFrame) -> CleaningResult:
    """Convert column names to lowercase snake_case."""
    working = df.copy()
    original_names = list(working.columns.astype(str))
    new_names = [_standardize_single_column_name(name) for name in original_names]

    # Resolve collisions by appending numeric suffixes.
    seen: dict[str, int] = {}
    resolved_names: list[str] = []
    renamed_count = 0

    for original, candidate in zip(original_names, new_names):
        if candidate in seen:
            seen[candidate] += 1
            final_name = f"{candidate}_{seen[candidate]}"
        else:
            seen[candidate] = 0
            final_name = candidate
        if original != final_name:
            renamed_count += 1
        resolved_names.append(final_name)

    working.columns = resolved_names
    return CleaningResult(
        dataframe=working,
        affected_rows=renamed_count,
        details=f"Standardized {renamed_count} column name(s) to snake_case.",
    )


def _try_convert_numeric(series: pd.Series) -> pd.Series | None:
    """Attempt numeric conversion when enough values parse successfully."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    converted = pd.to_numeric(series, errors="coerce")
    original_non_null = series.notna().sum()
    if original_non_null == 0:
        return None
    success_rate = converted.notna().sum() / original_non_null
    if success_rate >= _TYPE_CONVERSION_THRESHOLD:
        return converted
    return None


def _try_convert_datetime(series: pd.Series) -> pd.Series | None:
    """Attempt datetime conversion when enough values parse successfully."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    converted = pd.to_datetime(series, errors="coerce")
    original_non_null = series.notna().sum()
    if original_non_null == 0:
        return None
    success_rate = converted.notna().sum() / original_non_null
    if success_rate >= _TYPE_CONVERSION_THRESHOLD:
        return converted
    return None


def detect_and_fix_data_types(df: pd.DataFrame) -> CleaningResult:
    """
    Detect numeric, date, and text columns and coerce types where safe.

    Object columns are tested for numeric then datetime conversion.
    """
    working = df.copy()
    converted_columns: list[str] = []

    for col in working.columns:
        series = working[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        numeric_series = _try_convert_numeric(series)
        if numeric_series is not None:
            working[col] = numeric_series
            converted_columns.append(f"{col}→numeric")
            continue

        datetime_series = _try_convert_datetime(series)
        if datetime_series is not None:
            working[col] = datetime_series
            converted_columns.append(f"{col}→datetime")
            continue

        # Leave as text/object when conversion is not confident.
        if working[col].dtype == "object":
            working[col] = working[col].astype("string")

    details = (
        f"Converted {len(converted_columns)} column(s): {', '.join(converted_columns)}."
        if converted_columns
        else "No column types required conversion."
    )
    return CleaningResult(
        dataframe=working,
        affected_rows=len(converted_columns),
        details=details,
    )


def trim_whitespace(df: pd.DataFrame) -> CleaningResult:
    """Strip leading and trailing whitespace from text columns."""
    working = df.copy()
    affected_cells = 0

    text_cols = working.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        as_string = working[col].astype("string")
        stripped = as_string.str.strip()
        changed = as_string.notna() & (as_string != stripped)
        affected_cells += int(changed.sum())
        working[col] = stripped

    return CleaningResult(
        dataframe=working,
        affected_rows=affected_cells,
        details=f"Trimmed whitespace in {affected_cells} cell(s).",
    )


def remove_empty_rows(df: pd.DataFrame) -> CleaningResult:
    """Delete rows where every value is null or blank."""
    working = _replace_null_placeholders(df)
    before = len(working)
    non_empty_mask = ~working.apply(lambda row: all(_is_empty_value(v) for v in row), axis=1)
    cleaned = working.loc[non_empty_mask].reset_index(drop=True)
    removed = before - len(cleaned)

    return CleaningResult(
        dataframe=cleaned,
        affected_rows=removed,
        details=f"Removed {removed} completely empty row(s).",
    )


def remove_empty_columns(df: pd.DataFrame) -> CleaningResult:
    """Delete columns where every value is null or blank."""
    working = _replace_null_placeholders(df)
    empty_columns = [
        col for col in working.columns if working[col].apply(_is_empty_value).all()
    ]
    cleaned = working.drop(columns=empty_columns)
    removed = len(empty_columns)

    return CleaningResult(
        dataframe=cleaned,
        affected_rows=removed,
        details=f"Removed {removed} completely empty column(s).",
    )


def clean_everything(df: pd.DataFrame) -> CleaningResult:
    """
    Run the full cleaning pipeline in a sensible order.

    Order: names → whitespace → empty rows/columns → duplicates → types → missing values.
    """
    pipeline = (
        standardize_column_names,
        trim_whitespace,
        remove_empty_rows,
        remove_empty_columns,
        remove_duplicates,
        detect_and_fix_data_types,
        lambda frame: handle_missing_values(frame, strategy="smart"),
    )

    current = df.copy()
    total_affected = 0
    step_summaries: list[str] = []

    for step in pipeline:
        result = step(current)
        current = result.dataframe
        total_affected += result.affected_rows
        step_summaries.append(result.details)

    details = "Full clean completed. " + " | ".join(step_summaries)
    logger.info("Clean everything finished with %s total affected units.", total_affected)

    return CleaningResult(
        dataframe=current,
        affected_rows=total_affected,
        details=details,
    )
