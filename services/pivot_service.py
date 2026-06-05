"""
Pivot Intelligence Service
============================

Generates pivot tables from pandas DataFrames with automatic
field recommendations and aggregation support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PivotResult:
    """Container for pivot table generation results."""

    pivot_table: pd.DataFrame
    rows: list[str]
    columns: list[str] | None
    values: list[str]
    agg_function: str
    shape: tuple[int, int]
    total_cells: int
    message: str


class PivotServiceError(Exception):
    """Raised when pivot table generation fails."""


# Valid aggregation functions
VALID_AGG_FUNCTIONS = {"sum", "mean", "count", "min", "max"}


def _detect_field_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns as categorical or numeric for pivot recommendations."""
    categorical: list[str] = []
    numeric: list[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)

    return {"categorical": categorical, "numeric": numeric}


def get_recommended_pivot_fields(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze DataFrame and recommend fields for pivot table construction.

    Returns dict with:
        - row_candidates: list of categorical columns suitable for rows
        - col_candidates: list of categorical columns suitable for columns
        - value_candidates: list of numeric columns suitable for values
        - suggested_combinations: list of recommended (row, value, agg) tuples
    """
    if df.empty or len(df.columns) < 2:
        return {
            "row_candidates": [],
            "col_candidates": [],
            "value_candidates": [],
            "suggested_combinations": [],
            "message": "Dataset has insufficient columns for pivot analysis.",
        }

    field_types = _detect_field_types(df)
    categorical = field_types["categorical"]
    numeric = field_types["numeric"]

    # Filter out high-cardinality categorical fields (>100 unique values)
    # and very low-cardinality fields (<2 unique values)
    row_candidates = []
    col_candidates = []
    for col in categorical:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 100:
            row_candidates.append(col)
        if 2 <= unique_count <= 20:
            col_candidates.append(col)

    # Numeric fields with at least some variation
    value_candidates = []
    for col in numeric:
        if df[col].notna().sum() > 0 and df[col].nunique() > 1:
            value_candidates.append(col)

    # Generate suggested combinations
    suggested = []
    if row_candidates and value_candidates:
        for row in row_candidates[:3]:
            for val in value_candidates[:3]:
                suggested.append({
                    "rows": [row],
                    "columns": col_candidates[0] if col_candidates else None,
                    "values": [val],
                    "agg": "sum",
                    "reason": f"Summarize {val} by {row}",
                })

    return {
        "row_candidates": row_candidates,
        "col_candidates": col_candidates,
        "value_candidates": value_candidates,
        "suggested_combinations": suggested,
        "message": f"Found {len(row_candidates)} row candidates, {len(col_candidates)} column candidates, {len(value_candidates)} value candidates.",
    }


def generate_pivot_table(
    df: pd.DataFrame,
    rows: list[str],
    values: list[str],
    agg_function: str,
    columns: list[str] | None = None,
) -> PivotResult:
    """
    Generate a pivot table from the given DataFrame.

    Args:
        df: Source DataFrame.
        rows: List of column names to use as row indices.
        values: List of column names to aggregate.
        agg_function: Aggregation function ('sum', 'mean', 'count', 'min', 'max').
        columns: Optional list of column names for column indices.

    Returns:
        PivotResult containing the pivot table and metadata.
    """
    if df.empty:
        raise PivotServiceError("Cannot generate pivot table from empty DataFrame.")

    if not rows:
        raise PivotServiceError("At least one row field is required.")

    if not values:
        raise PivotServiceError("At least one value field is required.")

    agg_function = agg_function.lower().strip()
    if agg_function not in VALID_AGG_FUNCTIONS:
        raise PivotServiceError(
            f"Invalid aggregation function '{agg_function}'. "
            f"Valid options: {', '.join(sorted(VALID_AGG_FUNCTIONS))}"
        )

    # Validate fields exist
    all_fields = rows + values + (columns or [])
    missing = [f for f in all_fields if f not in df.columns]
    if missing:
        raise PivotServiceError(f"Field(s) not found in DataFrame: {', '.join(missing)}")

    # Validate value fields are numeric for sum/mean/min/max
    if agg_function in {"sum", "mean", "min", "max"}:
        non_numeric = [v for v in values if not pd.api.types.is_numeric_dtype(df[v])]
        if non_numeric:
            raise PivotServiceError(
                f"Aggregation '{agg_function}' requires numeric values. "
                f"Non-numeric field(s): {', '.join(non_numeric)}"
            )

    try:
        # Build pivot table using pandas pivot_table
        pivot_df = pd.pivot_table(
            df,
            index=rows,
            columns=columns if columns else None,
            values=values,
            aggfunc=agg_function,
            fill_value=0 if agg_function == "count" else 0,
            margins=False,
        )

        # Flatten multi-level columns if present
        if isinstance(pivot_df.columns, pd.MultiIndex):
            pivot_df.columns = [
                " | ".join(str(c) for c in col if str(c) != "") if isinstance(col, tuple) else str(col)
                for col in pivot_df.columns.values
            ]
        else:
            pivot_df.columns = [str(c) for c in pivot_df.columns]

        # Reset index to make rows regular columns for display
        pivot_df = pivot_df.reset_index()

        shape = (len(pivot_df), len(pivot_df.columns))
        total_cells = shape[0] * shape[1]

        message = (
            f"Pivot table generated: {shape[0]:,} rows × {shape[1]:,} columns. "
            f"Aggregated {agg_function} of {', '.join(values)} by {', '.join(rows)}"
            + (f" across {', '.join(columns)}" if columns else "")
            + "."
        )

        return PivotResult(
            pivot_table=pivot_df,
            rows=rows,
            columns=columns,
            values=values,
            agg_function=agg_function,
            shape=shape,
            total_cells=total_cells,
            message=message,
        )

    except Exception as exc:
        logger.exception("Pivot table generation failed")
        raise PivotServiceError(f"Failed to generate pivot table: {exc}") from exc