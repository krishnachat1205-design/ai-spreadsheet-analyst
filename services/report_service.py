"""
Report Service
==============

Generates a structured Analyst Report combining data quality, KPIs,
insights, recommendations, and cleaning history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from services.audit_service import get_action_history
from services.insight_service import InsightEngine, Insight, Recommendation
from services.kpi_service import KPIEngine, KPIResult

logger = logging.getLogger(__name__)


@dataclass
class AnalystReport:
    """Structured analyst report container."""

    dataset_overview: dict[str, Any]
    data_quality_summary: dict[str, Any]
    kpi_summary: list[KPIResult]
    key_insights: list[Insight]
    recommendations: list[Recommendation]
    cleaning_actions: list[dict[str, Any]]
    analyst_notes: str


def generate_analyst_report(df: pd.DataFrame) -> AnalystReport:
    """Build a comprehensive analyst report from the current DataFrame."""
    # Dataset overview
    overview = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_names": [str(c) for c in df.columns],
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    # Data quality summary
    total_missing = int(df.isna().sum().sum())
    total_cells = df.shape[0] * df.shape[1]
    dupes = int(df.duplicated().sum())
    empty_cols = int(df.isna().all().sum())

    quality = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": total_missing,
        "missing_percentage": round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0,
        "duplicate_rows": dupes,
        "duplicate_percentage": round((dupes / len(df) * 100), 2) if len(df) > 0 else 0,
        "empty_columns": empty_cols,
        "completeness_score": round(max(0, 100 - (total_missing / total_cells * 100)), 1) if total_cells > 0 else 100,
    }

    # KPIs
    kpi_engine = KPIEngine(df)
    kpis = kpi_engine.generate_all()

    # Insights & Recommendations
    insight_engine = InsightEngine(df)
    insights = insight_engine.generate_insights()
    recommendations = insight_engine.generate_recommendations()

    # Cleaning actions from audit log
    cleaning_actions = [
        entry
        for entry in get_action_history()
        if entry.get("action")
        in {
            "Standardize Column Names",
            "Remove Duplicates",
            "Fix Missing Values",
            "Fix Data Types",
            "Trim Whitespace",
            "Remove Empty Rows",
            "Remove Empty Columns",
            "Clean Everything",
            "File Upload",
        }
    ]

    # Analyst notes
    notes = _generate_analyst_notes(overview, quality, kpis, insights, recommendations)

    return AnalystReport(
        dataset_overview=overview,
        data_quality_summary=quality,
        kpi_summary=kpis,
        key_insights=insights,
        recommendations=recommendations,
        cleaning_actions=cleaning_actions,
        analyst_notes=notes,
    )


def _generate_analyst_notes(
    overview: dict,
    quality: dict,
    kpis: list[KPIResult],
    insights: list[Insight],
    recommendations: list[Recommendation],
) -> str:
    """Generate a narrative summary of the analysis."""
    notes_parts = []

    notes_parts.append(
        f"Dataset contains {overview['row_count']:,} rows and {overview['column_count']} columns. "
        f"Memory footprint: {overview['memory_usage_mb']} MB."
    )

    notes_parts.append(
        f"Data quality score: {quality['completeness_score']}/100. "
        f"Missing values: {quality['missing_values']:,} ({quality['missing_percentage']}%). "
        f"Duplicate rows: {quality['duplicate_rows']:,} ({quality['duplicate_percentage']}%)."
    )

    if kpis:
        kpi_names = [k.name for k in kpis[:3]]
        notes_parts.append(f"Key metrics detected: {', '.join(kpi_names)}.")
    else:
        notes_parts.append("No standard business metrics (revenue, orders, customers) were detected.")

    critical_insights = [i for i in insights if i.severity == "critical"]
    if critical_insights:
        notes_parts.append(
            f"CRITICAL: {len(critical_insights)} urgent issue(s) require immediate attention: "
            f"{critical_insights[0].observation}"
        )

    high_rec = [r for r in recommendations if r.priority == "high"]
    if high_rec:
        notes_parts.append(f"Top recommendation: {high_rec[0].title} — {high_rec[0].description}")

    return " ".join(notes_parts)