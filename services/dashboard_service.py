"""
Dashboard Service
=================

Production-ready Plotly-based analytics and visualization generation.
Optimized for large datasets with sampling, limits, and lazy loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# =============================================================================
# Performance Constants
# =============================================================================

MAX_HISTOGRAMS = 5
MAX_BOXPLOTS = 5
MAX_SCATTER_PLOTS = 3
MAX_PIE_CHARTS = 5
MAX_BAR_CHARTS = 5
MAX_SCATTER_SAMPLE = 3000
MAX_HISTOGRAM_SAMPLE = 5000
MAX_CORR_SAMPLE = 10000
LARGE_DATASET_THRESHOLD = 20000
VERY_LARGE_DATASET_THRESHOLD = 50000


# =============================================================================
# Dataset Tier Helpers
# =============================================================================

def get_dataset_size_tier(df: pd.DataFrame) -> str:
    """Classify dataset size for adaptive visualization behavior."""
    n = len(df)
    if n > VERY_LARGE_DATASET_THRESHOLD:
        return "very_large"
    if n > LARGE_DATASET_THRESHOLD:
        return "large"
    return "normal"


def _adaptive_sample_size(df: pd.DataFrame, base_limit: int) -> int:
    """Reduce sample sizes aggressively for large datasets."""
    tier = get_dataset_size_tier(df)
    if tier == "very_large":
        return min(base_limit // 5, 2000)
    if tier == "large":
        return min(base_limit // 2, 5000)
    return base_limit


def _limit_numeric_cols_for_corr(df: pd.DataFrame, numeric_cols: list[str], max_cols: int = 15) -> list[str]:
    """Return the most variable numeric columns when too many exist."""
    if len(numeric_cols) <= max_cols:
        return numeric_cols
    try:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        return variances.head(max_cols).index.tolist()
    except Exception:
        return numeric_cols[:max_cols]


# =============================================================================

@dataclass
class ChartResult:
    """Container for a generated chart and its metadata."""

    title: str
    figure: go.Figure
    chart_type: str
    columns_used: list[str]
    description: str
    applicable: bool = True


class DashboardServiceError(Exception):
    """Raised when chart generation fails."""


def _is_large_dataset(df: pd.DataFrame) -> bool:
    """Check if dataset exceeds the large dataset threshold."""
    return len(df) > LARGE_DATASET_THRESHOLD


def _sample_for_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Sample dataframe for correlation if it exceeds threshold."""
    limit = _adaptive_sample_size(df, MAX_CORR_SAMPLE)
    if len(df) > limit:
        return df[numeric_cols].sample(limit, random_state=42)
    return df[numeric_cols]


def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Automatically detect and classify columns by type.

    Returns dict with keys: numeric, categorical, date, boolean, all.
    """
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    date_cols: list[str] = []
    boolean_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        # Date detection
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue

        # Try parsing as date if object/string
        if series.dtype == "object" or pd.api.types.is_string_dtype(series):
            try:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().sum() / len(series) >= 0.5:
                    date_cols.append(col)
                    continue
            except Exception:
                pass

        # Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = series.dropna().nunique()
            if unique_vals == 2 and set(series.dropna().unique()).issubset({0, 1}):
                boolean_cols.append(col)
            else:
                numeric_cols.append(col)
            continue

        # Boolean detection
        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(col)
            continue

        # Categorical (everything else)
        categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols,
        "boolean": boolean_cols,
        "all": list(df.columns),
    }


def _safe_figure(title: str = "Chart") -> go.Figure:
    """Return an empty figure with a no-data message."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[
            dict(
                text="No data available for this chart",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _apply_common_layout(fig: go.Figure, title: str, height: int = 450) -> go.Figure:
    """Apply consistent theming to all figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=height,
        template="plotly_white",
        margin=dict(l=50, r=50, t=60, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


# =============================================================================
# A. KPI Visualizations
# =============================================================================


def generate_numeric_summary_charts(df: pd.DataFrame) -> list[ChartResult]:
    """
    Generate bar charts and line charts for numeric column summaries.
    Returns mean values as bar charts and trends if date columns exist.
    """
    results: list[ChartResult] = []
    col_types = detect_column_types(df)
    numeric_cols = col_types["numeric"]
    date_cols = col_types["date"]

    if not numeric_cols:
        results.append(
            ChartResult(
                title="Numeric Summary",
                figure=_safe_figure("Numeric Summary — No numeric columns"),
                chart_type="bar",
                columns_used=[],
                description="No numeric columns detected in dataset.",
                applicable=False,
            )
        )
        return results

    # Bar chart: mean of each numeric column
    try:
        means = df[numeric_cols].mean().dropna().sort_values(ascending=True)
        if not means.empty:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=means.values,
                        y=means.index.astype(str),
                        orientation="h",
                        marker_color="rgba(55, 126, 184, 0.8)",
                        text=[f"{v:,.2f}" for v in means.values],
                        textposition="outside",
                    )
                ]
            )
            fig = _apply_common_layout(fig, "Average Values by Numeric Column")
            results.append(
                ChartResult(
                    title="Average Values by Numeric Column",
                    figure=fig,
                    chart_type="bar",
                    columns_used=numeric_cols,
                    description="Horizontal bar chart showing mean values for each numeric column.",
                )
            )
    except Exception as exc:
        logger.warning("Numeric summary bar chart failed: %s", exc)

    # Line chart: trend over time if date + numeric exist
    if date_cols and numeric_cols:
        try:
            date_col = date_cols[0]
            numeric_col = numeric_cols[0]
            df_temp = df[[date_col, numeric_col]].copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors="coerce")
            df_temp = df_temp.dropna()

            if len(df_temp) > 1:
                df_temp = df_temp.sort_values(by=date_col)
                # Aggregate by month for readability
                df_temp["period"] = df_temp[date_col].dt.to_period("M").astype(str)
                monthly = df_temp.groupby("period")[numeric_col].sum().reset_index()

                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=monthly["period"],
                            y=monthly[numeric_col],
                            mode="lines+markers",
                            line=dict(color="rgba(228, 26, 28, 0.8)", width=2),
                            marker=dict(size=8),
                            name=numeric_col,
                        )
                    ]
                )
                fig = _apply_common_layout(fig, f"{numeric_col} Trend Over Time")
                fig.update_xaxes(tickangle=45)
                results.append(
                    ChartResult(
                        title=f"{numeric_col} Trend Over Time",
                        figure=fig,
                        chart_type="line",
                        columns_used=[date_col, numeric_col],
                        description=f"Line chart of {numeric_col} aggregated monthly by {date_col}.",
                    )
                )
        except Exception as exc:
            logger.warning("Trend line chart failed: %s", exc)

    return results


# =============================================================================
# B. Distribution Analysis
# =============================================================================


def generate_distribution_charts(df: pd.DataFrame) -> list[ChartResult]:
    """
    Generate histograms and box plots for numeric columns.
    Limited to MAX_HISTOGRAMS histograms and MAX_BOXPLOTS box plots.
    """
    results: list[ChartResult] = []
    col_types = detect_column_types(df)
    numeric_cols = col_types["numeric"]

    if not numeric_cols:
        results.append(
            ChartResult(
                title="Distribution Analysis",
                figure=_safe_figure("Distribution — No numeric columns"),
                chart_type="histogram",
                columns_used=[],
                description="No numeric columns available for distribution analysis.",
                applicable=False,
            )
        )
        return results

    tier = get_dataset_size_tier(df)

    # Adaptive limits
    max_histograms = 1 if tier == "very_large" else (2 if tier == "large" else MAX_HISTOGRAMS)
    max_boxplots = 1 if tier == "very_large" else (2 if tier == "large" else MAX_BOXPLOTS)
    max_hist_sample = _adaptive_sample_size(df, MAX_HISTOGRAM_SAMPLE)

    # Histograms for up to max_histograms numeric columns
    for col in numeric_cols[:max_histograms]:
        try:
            series = df[col].dropna()
            if series.empty:
                continue

            # Sample for histogram if dataset is large
            if len(series) > max_hist_sample:
                series = series.sample(max_hist_sample, random_state=42)

            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=series,
                        nbinsx=min(50, max(10, int(series.nunique() / 2))),
                        marker_color="rgba(77, 175, 74, 0.7)",
                        marker_line_color="rgba(77, 175, 74, 1.0)",
                        marker_line_width=1,
                        name=col,
                    )
                ]
            )
            fig = _apply_common_layout(fig, f"Distribution of {col}")
            fig.update_xaxes(title_text=col)
            fig.update_yaxes(title_text="Frequency")
            results.append(
                ChartResult(
                    title=f"Distribution of {col}",
                    figure=fig,
                    chart_type="histogram",
                    columns_used=[col],
                    description=f"Histogram showing frequency distribution of {col}.",
                )
            )
        except Exception as exc:
            logger.warning("Histogram for %s failed: %s", col, exc)

    # Box plot for up to max_boxplots numeric columns
    try:
        subset_cols = numeric_cols[:max_boxplots]
        subset_df = df[subset_cols].dropna(how="all")
        if not subset_df.empty and len(subset_cols) > 0:
            fig = go.Figure()
            for col in subset_cols:
                fig.add_trace(
                    go.Box(
                        y=subset_df[col].dropna(),
                        name=str(col)[:20],
                        boxpoints="outliers",
                        marker=dict(size=4),
                    )
                )
            fig = _apply_common_layout(fig, "Box Plot Comparison")
            fig.update_yaxes(title_text="Value")
            results.append(
                ChartResult(
                    title="Box Plot Comparison",
                    figure=fig,
                    chart_type="box",
                    columns_used=subset_cols,
                    description="Box plots comparing distributions across numeric columns.",
                )
            )
    except Exception as exc:
        logger.warning("Box plot failed: %s", exc)

    return results


# =============================================================================
# C. Category Analysis
# =============================================================================


def generate_categorical_summary_charts(df: pd.DataFrame) -> list[ChartResult]:
    """
    Generate pie charts and bar charts for categorical columns.
    Limited to MAX_PIE_CHARTS pie charts and MAX_BAR_CHARTS bar charts.
    """
    results: list[ChartResult] = []
    col_types = detect_column_types(df)
    categorical_cols = col_types["categorical"]

    if not categorical_cols:
        results.append(
            ChartResult(
                title="Category Analysis",
                figure=_safe_figure("Category Analysis — No categorical columns"),
                chart_type="pie",
                columns_used=[],
                description="No categorical columns detected.",
                applicable=False,
            )
        )
        return results

    tier = get_dataset_size_tier(df)
    max_pie = 2 if tier == "very_large" else (3 if tier == "large" else MAX_PIE_CHARTS)
    max_bar = 2 if tier == "very_large" else (3 if tier == "large" else MAX_BAR_CHARTS)

    # Pie charts for up to max_pie categorical columns
    for col in categorical_cols[:max_pie]:
        try:
            counts = df[col].value_counts().head(10)
            if counts.empty:
                continue

            colors = px.colors.qualitative.Set3[: len(counts)]
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=counts.index.astype(str),
                        values=counts.values,
                        hole=0.35,
                        marker_colors=colors,
                        textinfo="label+percent",
                        textposition="outside",
                        pull=[0.02] * len(counts),
                    )
                ]
            )
            fig = _apply_common_layout(fig, f"Top Categories in {col}")
            fig.update_layout(showlegend=False, height=450)
            results.append(
                ChartResult(
                    title=f"Top Categories in {col}",
                    figure=fig,
                    chart_type="pie",
                    columns_used=[col],
                    description=f"Pie chart showing top 10 categories in {col}.",
                )
            )
        except Exception as exc:
            logger.warning("Pie chart for %s failed: %s", col, exc)

    # Bar charts for up to max_bar categorical columns
    for col in categorical_cols[:max_bar]:
        try:
            counts = df[col].value_counts().head(15)
            if counts.empty:
                continue

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=counts.index.astype(str),
                        y=counts.values,
                        marker_color="rgba(152, 78, 163, 0.8)",
                        text=counts.values,
                        textposition="outside",
                    )
                ]
            )
            fig = _apply_common_layout(fig, f"Value Counts: {col}")
            fig.update_xaxes(title_text=col, tickangle=45)
            fig.update_yaxes(title_text="Count")
            results.append(
                ChartResult(
                    title=f"Value Counts: {col}",
                    figure=fig,
                    chart_type="bar",
                    columns_used=[col],
                    description=f"Bar chart of top 15 value counts in {col}.",
                )
            )
        except Exception as exc:
            logger.warning("Bar chart for %s failed: %s", col, exc)

    return results


# =============================================================================
# D. Relationship Analysis
# =============================================================================


def generate_correlation_heatmap(df: pd.DataFrame) -> ChartResult:
    """
    Generate a correlation heatmap for all numeric columns.
    Uses sampling if dataset exceeds MAX_CORR_SAMPLE rows.
    """
    col_types = detect_column_types(df)
    numeric_cols = col_types["numeric"]

    if len(numeric_cols) < 2:
        return ChartResult(
            title="Correlation Heatmap",
            figure=_safe_figure("Correlation — Need 2+ numeric columns"),
            chart_type="heatmap",
            columns_used=numeric_cols,
            description="At least 2 numeric columns required for correlation analysis.",
            applicable=False,
        )

    tier = get_dataset_size_tier(df)

    # For very large datasets with many numeric columns, limit to the most variable ones
    if tier == "very_large" and len(numeric_cols) > 15:
        numeric_cols = _limit_numeric_cols_for_corr(df, numeric_cols, max_cols=15)

    try:
        corr_df = _sample_for_correlation(df, numeric_cols)
        corr_matrix = corr_df.corr().round(2)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.astype(str),
                y=corr_matrix.index.astype(str),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr_matrix.values,
                texttemplate="%{text:.2f}",
                textfont=dict(size=10),
                hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )
        fig = _apply_common_layout(fig, "Correlation Matrix", height=500)
        fig.update_layout(showlegend=False)

        return ChartResult(
            title="Correlation Matrix",
            figure=fig,
            chart_type="heatmap",
            columns_used=numeric_cols,
            description="Pearson correlation heatmap across all numeric columns.",
        )
    except Exception as exc:
        logger.warning("Correlation heatmap failed: %s", exc)
        return ChartResult(
            title="Correlation Matrix",
            figure=_safe_figure("Correlation — Generation Error"),
            chart_type="heatmap",
            columns_used=numeric_cols,
            description=f"Failed to generate: {exc}",
            applicable=False,
        )


def generate_scatter_plots(df: pd.DataFrame) -> list[ChartResult]:
    """
    Generate scatter plots for pairs of numeric columns.
    Limited to MAX_SCATTER_PLOTS plots with MAX_SCATTER_SAMPLE rows.
    """
    results: list[ChartResult] = []
    col_types = detect_column_types(df)
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]

    tier = get_dataset_size_tier(df)

    # Very large datasets: skip scatter entirely to avoid browser / compute freeze
    if tier == "very_large":
        return [
            ChartResult(
                title="Scatter Plots",
                figure=_safe_figure("Scatter Plots — Disabled for Very Large Datasets"),
                chart_type="scatter",
                columns_used=numeric_cols,
                description="Scatter plots are disabled for datasets over 50,000 rows to maintain performance.",
                applicable=False,
            )
        ]

    if len(numeric_cols) < 2:
        return [
            ChartResult(
                title="Scatter Plots",
                figure=_safe_figure("Scatter — Need 2+ numeric columns"),
                chart_type="scatter",
                columns_used=numeric_cols,
                description="At least 2 numeric columns required for scatter plots.",
                applicable=False,
            )
        ]

    # Adaptive limits
    max_plots = 1 if tier == "large" else MAX_SCATTER_PLOTS
    max_sample = 1000 if tier == "large" else MAX_SCATTER_SAMPLE

    # Generate scatter for up to max_plots pairs
    pairs = []
    for i in range(min(len(numeric_cols), 4)):
        for j in range(i + 1, min(len(numeric_cols), 4)):
            pairs.append((numeric_cols[i], numeric_cols[j]))
            if len(pairs) >= max_plots:
                break
        if len(pairs) >= max_plots:
            break

    color_col = categorical_cols[0] if categorical_cols else None

    for x_col, y_col in pairs:
        try:
            cols = [x_col, y_col]
            if color_col and color_col in df.columns:
                cols.append(color_col)

            # Sample FIRST, then dropna — much faster for large datasets
            plot_df = df[cols].sample(min(max_sample, len(df)), random_state=42).dropna()

            if len(plot_df) < 2:
                continue

            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col else None,
                opacity=0.6,
                title=f"{y_col} vs {x_col}",
                template="plotly_white",
                height=450,
            )
            fig = _apply_common_layout(fig, f"{y_col} vs {x_col}")
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))

            results.append(
                ChartResult(
                    title=f"{y_col} vs {x_col}",
                    figure=fig,
                    chart_type="scatter",
                    columns_used=[x_col, y_col] + ([color_col] if color_col else []),
                    description=f"Scatter plot exploring relationship between {x_col} and {y_col}.",
                )
            )
        except Exception as exc:
            logger.warning("Scatter plot %s vs %s failed: %s", x_col, y_col, exc)

    return results


# =============================================================================
# E. Top-N Analysis
# =============================================================================


def generate_top_n_charts(df: pd.DataFrame, n: int = 10) -> list[ChartResult]:
    """
    Generate top-N bar charts for categorical vs numeric combinations.
    """
    results: list[ChartResult] = []
    col_types = detect_column_types(df)
    categorical_cols = col_types["categorical"]
    numeric_cols = col_types["numeric"]

    if not categorical_cols or not numeric_cols:
        results.append(
            ChartResult(
                title="Top-N Analysis",
                figure=_safe_figure("Top-N — Need categorical + numeric columns"),
                chart_type="bar",
                columns_used=[],
                description="Requires at least one categorical and one numeric column.",
                applicable=False,
            )
        )
        return results

    # Top categories by first numeric column
    cat_col = categorical_cols[0]
    num_col = numeric_cols[0]

    try:
        grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(n)
        if not grouped.empty:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=grouped.index.astype(str),
                        y=grouped.values,
                        marker_color="rgba(255, 127, 0, 0.8)",
                        text=[f"{v:,.0f}" for v in grouped.values],
                        textposition="outside",
                    )
                ]
            )
            fig = _apply_common_layout(fig, f"Top {n} {cat_col} by {num_col}")
            fig.update_xaxes(title_text=cat_col, tickangle=45)
            fig.update_yaxes(title_text=num_col)
            results.append(
                ChartResult(
                    title=f"Top {n} {cat_col} by {num_col}",
                    figure=fig,
                    chart_type="bar",
                    columns_used=[cat_col, num_col],
                    description=f"Top {n} categories ranked by total {num_col}.",
                )
            )
    except Exception as exc:
        logger.warning("Top-N chart failed: %s", exc)

    # Horizontal top-N for second numeric if available
    if len(numeric_cols) > 1 and len(categorical_cols) > 0:
        try:
            num_col_2 = numeric_cols[1]
            grouped_h = (
                df.groupby(cat_col)[num_col_2].sum().sort_values(ascending=True).tail(n)
            )
            if not grouped_h.empty:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=grouped_h.values,
                            y=grouped_h.index.astype(str),
                            orientation="h",
                            marker_color="rgba(55, 126, 184, 0.8)",
                            text=[f"{v:,.0f}" for v in grouped_h.values],
                            textposition="outside",
                        )
                    ]
                )
                fig = _apply_common_layout(fig, f"Top {n} {cat_col} by {num_col_2}")
                fig.update_xaxes(title_text=num_col_2)
                results.append(
                    ChartResult(
                        title=f"Top {n} {cat_col} by {num_col_2}",
                        figure=fig,
                        chart_type="bar",
                        columns_used=[cat_col, num_col_2],
                        description=f"Horizontal bar chart of top {n} categories by {num_col_2}.",
                    )
                )
        except Exception as exc:
            logger.warning("Horizontal top-N chart failed: %s", exc)

    return results


# =============================================================================
# F. Chart Recommendation
# =============================================================================


def recommend_best_charts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Analyze dataset structure and recommend the most useful chart types.

    Returns list of recommendation dicts with keys: chart_type, reason, priority, columns.
    """
    recommendations: list[dict[str, Any]] = []
    col_types = detect_column_types(df)
    numeric = col_types["numeric"]
    categorical = col_types["categorical"]
    date = col_types["date"]
    total_cols = len(df.columns)

    # Always recommend distribution if numeric exists
    if numeric:
        recommendations.append(
            {
                "chart_type": "histogram",
                "reason": f"{len(numeric)} numeric column(s) detected. Distribution analysis reveals data spread and outliers.",
                "priority": "high",
                "columns": numeric[:3],
            }
        )

    # Correlation if 2+ numeric
    if len(numeric) >= 2:
        recommendations.append(
            {
                "chart_type": "heatmap",
                "reason": f"{len(numeric)} numeric columns support correlation analysis to find relationships.",
                "priority": "high",
                "columns": numeric,
            }
        )

    # Categorical analysis
    if categorical:
        recommendations.append(
            {
                "chart_type": "pie",
                "reason": f"{len(categorical)} categorical column(s) detected. Category breakdown shows composition.",
                "priority": "medium",
                "columns": categorical[:2],
            }
        )
        if numeric:
            recommendations.append(
                {
                    "chart_type": "bar",
                    "reason": "Categorical + numeric columns enable ranking and comparison charts.",
                    "priority": "high",
                    "columns": [categorical[0], numeric[0]] if categorical and numeric else [],
                }
            )

    # Time series if date + numeric
    if date and numeric:
        recommendations.append(
            {
                "chart_type": "line",
                "reason": f"Date column(s) ({', '.join(date[:1])}) and numeric data support trend analysis.",
                "priority": "high",
                "columns": [date[0], numeric[0]],
            }
        )

    # Scatter if 2+ numeric
    if len(numeric) >= 2:
        recommendations.append(
            {
                "chart_type": "scatter",
                "reason": "Multiple numeric columns allow exploration of variable relationships.",
                "priority": "medium",
                "columns": numeric[:2],
            }
        )

    # Box plot for outlier detection
    if len(numeric) >= 1:
        recommendations.append(
            {
                "chart_type": "box",
                "reason": "Box plots identify outliers and compare distributions across columns.",
                "priority": "medium",
                "columns": numeric[:5],
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "chart_type": "table",
                "reason": "Dataset structure is unusual. Review raw data before selecting charts.",
                "priority": "low",
                "columns": list(df.columns)[:5],
            }
        )

    return recommendations


# =============================================================================
# G. Lazy Loading: Per-Tab Chart Generation
# =============================================================================


def generate_overview_charts(df: pd.DataFrame) -> list[ChartResult]:
    """Generate only overview tab charts (numeric summaries)."""
    return generate_numeric_summary_charts(df)


def generate_trends_charts(df: pd.DataFrame) -> list[ChartResult]:
    """Generate only trends tab charts (distributions)."""
    return generate_distribution_charts(df)


def generate_categories_charts(df: pd.DataFrame) -> list[ChartResult]:
    """Generate only categories tab charts (categorical + top-n)."""
    cat_charts = generate_categorical_summary_charts(df)
    top_n_charts = generate_top_n_charts(df)
    return cat_charts + top_n_charts


def generate_relationships_charts(df: pd.DataFrame) -> list[ChartResult]:
    """Generate only relationships tab charts (heatmap + scatter)."""
    heatmap = generate_correlation_heatmap(df)
    scatters = generate_scatter_plots(df)
    return [heatmap] + scatters


# =============================================================================
# H. Convenience: Generate All Charts (kept for backward compatibility)
# =============================================================================


def generate_all_charts(df: pd.DataFrame) -> dict[str, list[ChartResult]]:
    """
    Generate all available chart groups in one call.
    NOTE: For large datasets, prefer per-tab lazy loading functions above.

    Returns dict with keys: numeric_summary, distribution, categorical, relationships, top_n.
    """
    return {
        "numeric_summary": generate_numeric_summary_charts(df),
        "distribution": generate_distribution_charts(df),
        "categorical": generate_categorical_summary_charts(df),
        "relationships": [generate_correlation_heatmap(df)] + generate_scatter_plots(df),
        "top_n": generate_top_n_charts(df),
        "recommendations": recommend_best_charts(df),
    }