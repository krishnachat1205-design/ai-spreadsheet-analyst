""
AI Spreadsheet Analyst — Streamlit Application Entry Point
==========================================================

Run with: streamlit run app.py
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from core.config import APP_TITLE, APP_VERSION, PAGE_LAYOUT
from services.audit_service import get_action_history, log_action
from services.cleaning_service import (
    CleaningResult,
    CleaningServiceError,
    clean_everything,
    detect_and_fix_data_types,
    generate_data_quality_report,
    handle_missing_values,
    remove_duplicates,
    remove_empty_columns,
    remove_empty_rows,
    standardize_column_names,
    trim_whitespace,
)
from services.dashboard_service import (
    detect_column_types,
    generate_all_charts,
    generate_categorical_summary_charts,
    generate_correlation_heatmap,
    generate_distribution_charts,
    generate_numeric_summary_charts,
    generate_scatter_plots,
    generate_top_n_charts,
    recommend_best_charts,
)
from services.file_service import (
    get_file_metadata,
    load_dataframe_from_upload,
)
from services.kpi_service import generate_kpis, KPIResult
from services.insight_service import generate_insights, generate_recommendations, Insight, Recommendation
from services.report_service import generate_analyst_report, AnalystReport
from utils.exceptions import FileLoadError, FileServiceError, FileValidationError
from utils.helpers import compute_dataset_health, preview_dataframe


def configure_page() -> None:
    """Set global Streamlit page configuration."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout=PAGE_LAYOUT,
        initial_sidebar_state="expanded",
    )


def init_session_state() -> None:
    """Initialize Streamlit session keys used by the upload and cleaning workflow."""
    defaults = {
        "dataframe": None,
        "original_dataframe": None,
        "file_metadata": None,
        "dataset_health": None,
        "quality_report_before": None,
        "quality_report_after": None,
        "upload_error": None,
        "bi_kpis": None,
        "bi_insights": None,
        "bi_recommendations": None,
        "bi_report": None,
        "dashboard_charts": None,
        "dashboard_recommendations": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _refresh_dataset_state(df: pd.DataFrame) -> None:
    """Sync derived session state after the working DataFrame changes."""
    st.session_state.dataframe = df
    st.session_state.dataset_health = compute_dataset_health(df)
    st.session_state.quality_report_after = generate_data_quality_report(df)

    metadata = st.session_state.file_metadata
    if metadata is not None:
        metadata["row_count"] = len(df)
        metadata["column_count"] = len(df.columns)
        metadata["column_names"] = [str(col) for col in df.columns]
        st.session_state.file_metadata = metadata


def _apply_cleaning_operation(
    action_name: str,
    cleaning_fn: Callable[[pd.DataFrame], CleaningResult],
) -> None:
    """
    Run a cleaning function on the active DataFrame, update session state,
    and record the operation in the audit log.
    """
    df = st.session_state.dataframe
    if df is None:
        st.warning("Upload a file before running cleaning operations.")
        return

    try:
        result = cleaning_fn(df)
        _refresh_dataset_state(result.dataframe)
        log_action(
            action=action_name,
            details=result.details,
            affected_rows=result.affected_rows,
        )
        st.success(result.details)
    except CleaningServiceError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Cleaning failed: {exc}")


def render_sidebar() -> None:
    """Render upload controls, file info, and the AI action log."""
    st.sidebar.header("Upload File")
    uploaded = st.sidebar.file_uploader(
        "Choose a spreadsheet",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX, XLS (max 50 MB)",
    )

    if uploaded is not None:
        _process_upload(uploaded)

    st.sidebar.divider()
    st.sidebar.subheader("File Information")

    metadata = st.session_state.file_metadata
    if metadata is None:
        st.sidebar.caption("No file loaded yet.")
    else:
        st.sidebar.metric("File Name", metadata.get("original_filename", "—"))
        st.sidebar.metric("File Type", metadata.get("file_type", "—").upper())
        st.sidebar.metric("File Size", metadata.get("file_size_human", "—"))

        if metadata.get("row_count") is not None:
            st.sidebar.metric("Rows", f"{metadata['row_count']:,}")
            st.sidebar.metric("Columns", metadata.get("column_count", "—"))

        sheet_names = metadata.get("sheet_names") or []
        if sheet_names:
            st.sidebar.markdown("**Sheets**")
            for name in sheet_names:
                st.sidebar.text(f"• {name}")

    st.sidebar.divider()
    render_action_log_panel()


def render_action_log_panel() -> None:
    """Sidebar section showing timestamped audit entries."""
    st.sidebar.subheader("AI Action Log")

    history = get_action_history()
    if not history:
        st.sidebar.caption("No actions recorded yet.")
        return

    for entry in reversed(history[-15:]):
        timestamp = entry.get("timestamp", "—")
        action = entry.get("action", "—")
        details = entry.get("details", "—")
        affected = entry.get("affected_rows", 0)

        with st.sidebar.expander(f"{action} · {timestamp[:19]}", expanded=False):
            st.markdown(f"**Timestamp:** {timestamp}")
            st.markdown(f"**Action:** {action}")
            st.markdown(f"**Details:** {details}")
            st.markdown(f"**Affected rows/cells:** {affected:,}")


def _process_upload(uploaded_file) -> None:
    """Handle a new upload: validate, save, load, and store in session state."""
    upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_last_upload_key") == upload_key:
        return

    st.session_state.upload_error = None

    try:
        content = uploaded_file.getvalue()
        df, saved_path, _validation = load_dataframe_from_upload(
            content,
            uploaded_file.name,
        )
        metadata = get_file_metadata(
            saved_path,
            df,
            original_filename=uploaded_file.name,
        )
        health = compute_dataset_health(df)
        quality_before = generate_data_quality_report(df)

        st.session_state.dataframe = df
        st.session_state.original_dataframe = df.copy()
        st.session_state.file_metadata = metadata
        st.session_state.dataset_health = health
        st.session_state.quality_report_before = quality_before
        st.session_state.quality_report_after = quality_before
        st.session_state._last_upload_key = upload_key
        st.session_state.bi_kpis = None
        st.session_state.bi_insights = None
        st.session_state.bi_recommendations = None
        st.session_state.bi_report = None
        st.session_state.dashboard_charts = None
        st.session_state.dashboard_recommendations = None

        log_action(
            action="File Upload",
            details=f"Loaded '{uploaded_file.name}' ({quality_before['rows']} rows, "
            f"{quality_before['columns']} columns).",
            affected_rows=0,
        )
        st.sidebar.success(f"Loaded **{uploaded_file.name}** successfully.")

    except FileValidationError as exc:
        st.session_state.upload_error = str(exc)
        st.sidebar.error(str(exc))
    except FileLoadError as exc:
        st.session_state.upload_error = str(exc)
        st.sidebar.error(str(exc))
    except FileServiceError as exc:
        st.session_state.upload_error = str(exc)
        st.sidebar.error(f"File service error: {exc}")
    except Exception as exc:
        st.session_state.upload_error = str(exc)
        st.sidebar.error(f"Unexpected error: {exc}")


def render_data_preview(df: pd.DataFrame) -> None:
    """Section 1 — row/column counts, column names, and first 20 rows."""
    st.subheader("Data Preview")

    col1, col2 = st.columns(2)
    col1.metric("Number of Rows", f"{len(df):,}")
    col2.metric("Number of Columns", len(df.columns))

    st.markdown("**Column Names**")
    st.code(", ".join(str(c) for c in df.columns), language=None)

    st.markdown("**First 20 Rows**")
    st.dataframe(preview_dataframe(df, n_rows=20), use_container_width=True, hide_index=False)


def render_dataset_health(health: dict) -> None:
    """Section 2 — missing values, duplicate rows, and data types."""
    st.subheader("Dataset Health Check")

    missing = health["missing_values"]
    dupes = health["duplicate_rows"]
    dtypes = health["data_types"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Missing Values", f"{missing['total']:,}")
    m2.metric("Duplicate Rows", f"{dupes:,}")
    m3.metric("Columns", len(dtypes))

    tab_missing, tab_dupes, tab_types = st.tabs(
        ["Missing Values", "Duplicate Rows", "Data Types"]
    )

    with tab_missing:
        if missing["by_column"]:
            missing_df = pd.DataFrame(
                [
                    {"Column": col, "Missing Count": count}
                    for col, count in missing["by_column"].items()
                ]
            )
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.info("No columns to inspect.")

    with tab_dupes:
        if dupes == 0:
            st.success("No duplicate rows detected.")
        else:
            st.warning(f"Found **{dupes:,}** duplicate row(s). Use Data Cleaning Center to remove them.")

    with tab_types:
        dtype_df = pd.DataFrame(
            [{"Column": col, "Data Type": dtype} for col, dtype in dtypes.items()]
        )
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)


def render_basic_statistics(df: pd.DataFrame) -> None:
    """Section 3 — pandas describe() for numeric and object columns."""
    st.subheader("Basic Statistics")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        st.info("No numeric columns available for statistical summary.")
    else:
        st.markdown("**Numeric columns** (`describe()`)")
        st.dataframe(numeric_df.describe(), use_container_width=True)

    object_df = df.select_dtypes(include=["object", "string", "category"])
    if not object_df.empty:
        st.markdown("**Text / categorical columns** (`describe()`)")
        st.dataframe(object_df.describe(), use_container_width=True)


def _render_quality_metrics(report: dict[str, Any], label: str) -> None:
    """Render a compact quality summary block."""
    st.markdown(f"**{label}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{report['rows']:,}")
    c2.metric("Columns", report["columns"])
    c3.metric("Duplicates", f"{report['duplicates']:,}")
    c4.metric("Missing Values", f"{report['missing_values']:,}")


def render_data_quality_report() -> None:
    """Before/after quality comparison using stored reports."""
    st.subheader("Data Quality Report")

    before = st.session_state.quality_report_before
    after = st.session_state.quality_report_after

    if before is None or after is None:
        st.info("Quality report will appear after a file is loaded.")
        return

    col_before, col_after = st.columns(2)
    with col_before:
        _render_quality_metrics(before, "Before Cleaning")
    with col_after:
        _render_quality_metrics(after, "After Cleaning")


def render_data_cleaning_center() -> None:
    """Interactive cleaning controls with audit logging on every action."""
    st.subheader("Data Cleaning Center")
    st.caption("Each operation updates the dataset and is recorded in the AI Action Log.")

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("Standardize Column Names", use_container_width=True):
            _apply_cleaning_operation("Standardize Column Names", standardize_column_names)
        if st.button("Remove Duplicates", use_container_width=True):
            _apply_cleaning_operation("Remove Duplicates", remove_duplicates)

    with btn_col2:
        if st.button("Fix Missing Values", use_container_width=True):
            _apply_cleaning_operation(
                "Fix Missing Values",
                lambda df: handle_missing_values(df, strategy="smart"),
            )
        if st.button("Fix Data Types", use_container_width=True):
            _apply_cleaning_operation("Fix Data Types", detect_and_fix_data_types)

    with btn_col3:
        if st.button("Trim Whitespace", use_container_width=True):
            _apply_cleaning_operation("Trim Whitespace", trim_whitespace)
        if st.button("Remove Empty Rows", use_container_width=True):
            _apply_cleaning_operation("Remove Empty Rows", remove_empty_rows)

    with btn_col4:
        if st.button("Remove Empty Columns", use_container_width=True):
            _apply_cleaning_operation("Remove Empty Columns", remove_empty_columns)
        if st.button("Clean Everything", type="primary", use_container_width=True):
            _apply_cleaning_operation("Clean Everything", clean_everything)


# =============================================================================
# Business Intelligence Center
# =============================================================================


def render_business_intelligence_center() -> None:
    """Business Intelligence Center for KPIs, Insights, Recommendations, and Reports."""
    st.subheader("Business Intelligence Center")
    st.caption("Generate actionable intelligence from your dataset.")

    df = st.session_state.dataframe
    if df is None:
        st.info("Upload a file to unlock Business Intelligence features.")
        return

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("Generate KPIs", use_container_width=True):
            with st.spinner("Analyzing metrics..."):
                kpis = generate_kpis(df)
                st.session_state.bi_kpis = kpis
                log_action(
                    action="Generate KPIs",
                    details=f"Generated {len(kpis)} KPIs from dataset.",
                    affected_rows=len(df),
                )
            st.success(f"{len(kpis)} KPIs generated.")

    with btn_col2:
        if st.button("Generate Insights", use_container_width=True):
            with st.spinner("Detecting insights..."):
                insights = generate_insights(df)
                st.session_state.bi_insights = insights
                log_action(
                    action="Generate Insights",
                    details=f"Generated {len(insights)} business insights.",
                    affected_rows=len(df),
                )
            st.success(f"{len(insights)} insights generated.")

    with btn_col3:
        if st.button("Generate Recommendations", use_container_width=True):
            with st.spinner("Building recommendations..."):
                recommendations = generate_recommendations(df)
                st.session_state.bi_recommendations = recommendations
                log_action(
                    action="Generate Recommendations",
                    details=f"Generated {len(recommendations)} recommendations.",
                    affected_rows=len(df),
                )
            st.success(f"{len(recommendations)} recommendations generated.")

    with btn_col4:
        if st.button("Generate Analyst Report", type="primary", use_container_width=True):
            with st.spinner("Compiling analyst report..."):
                report = generate_analyst_report(df)
                st.session_state.bi_report = report
                log_action(
                    action="Generate Analyst Report",
                    details="Full analyst report compiled with KPIs, insights, and recommendations.",
                    affected_rows=len(df),
                )
            st.success("Analyst report ready.")

    _display_kpis()
    _display_insights()
    _display_recommendations()
    _display_report()


def _display_kpis() -> None:
    """Display generated KPIs as metric cards and detail table."""
    kpis = st.session_state.get("bi_kpis")
    if not kpis:
        return

    st.markdown("---")
    st.markdown("### 📊 Key Performance Indicators")

    cols = st.columns(min(len(kpis), 4))
    for idx, kpi in enumerate(kpis):
        with cols[idx % 4]:
            st.metric(label=kpi.name, value=kpi.formatted_value)
            st.caption(kpi.description)

    with st.expander("KPI Detail Table", expanded=False):
        kpi_data = [
            {
                "KPI": k.name,
                "Value": k.formatted_value,
                "Raw Value": k.value,
                "Column": k.column_used or "—",
                "Confidence": k.confidence,
                "Description": k.description,
            }
            for k in kpis
        ]
        st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)


def _display_insights() -> None:
    """Display generated insights with severity icons and expanders."""
    insights = st.session_state.get("bi_insights")
    if not insights:
        return

    st.markdown("---")
    st.markdown("### 💡 Key Insights")

    for insight in insights:
        severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
            insight.severity, "ℹ️"
        )
        with st.expander(
            f"{severity_icon} {insight.observation}",
            expanded=(insight.severity == "critical"),
        ):
            st.markdown(f"**Evidence:** {insight.evidence}")
            st.markdown(f"**Business Interpretation:** {insight.business_interpretation}")
            st.markdown(f"**Category:** `{insight.category}` | **Severity:** `{insight.severity}`")


def _display_recommendations() -> None:
    """Display generated recommendations sorted by priority."""
    recommendations = st.session_state.get("bi_recommendations")
    if not recommendations:
        return

    st.markdown("---")
    st.markdown("### 🎯 Recommendations")

    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_recs = sorted(
        recommendations, key=lambda r: priority_order.get(r.priority, 99)
    )

    for rec in sorted_recs:
        priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
            rec.priority, "⚪"
        )
        with st.expander(
            f"{priority_color} {rec.title} ({rec.priority.upper()})",
            expanded=(rec.priority == "high"),
        ):
            st.markdown(f"**Description:** {rec.description}")
            st.markdown(f"**Evidence:** {rec.evidence}")
            st.markdown(f"**Action Category:** `{rec.action_category}`")


def _display_report() -> None:
    """Display the full structured analyst report."""
    report = st.session_state.get("bi_report")
    if not report:
        return

    st.markdown("---")
    st.markdown("### 📋 Analyst Report")

    st.markdown("#### Dataset Overview")
    overview = report.dataset_overview
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['row_count']:,}")
    c2.metric("Columns", overview["column_count"])
    c3.metric("Memory", f"{overview['memory_usage_mb']} MB")
    c4.metric("Total Columns", len(overview["column_names"]))

    with st.expander("Column Names"):
        st.write(", ".join(overview["column_names"]))

    st.markdown("#### Data Quality Summary")
    q = report.data_quality_summary
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Completeness Score", f"{q['completeness_score']}/100")
    q2.metric("Missing Values", f"{q['missing_values']:,}")
    q3.metric("Duplicate Rows", f"{q['duplicate_rows']:,}")
    q4.metric("Empty Columns", q["empty_columns"])

    if report.kpi_summary:
        st.markdown("#### KPI Summary")
        kpi_df = pd.DataFrame(
            [
                {"KPI": k.name, "Value": k.formatted_value, "Confidence": k.confidence}
                for k in report.kpi_summary
            ]
        )
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    if report.key_insights:
        st.markdown("#### Key Insights")
        for insight in report.key_insights:
            severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
                insight.severity, "ℹ️"
            )
            st.markdown(f"{severity_icon} **{insight.observation}**")
            st.markdown(f"→ *Evidence:* {insight.evidence}")
            st.markdown(f"→ *Interpretation:* {insight.business_interpretation}")
            st.divider()

    if report.recommendations:
        st.markdown("#### Recommendations")
        for rec in report.recommendations:
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                rec.priority, "⚪"
            )
            st.markdown(f"{priority_color} **{rec.title}** ({rec.priority.upper()})")
            st.markdown(f"→ {rec.description}")
            st.markdown(f"→ *Evidence:* {rec.evidence}")
            st.divider()

    if report.cleaning_actions:
        st.markdown("#### Cleaning Actions Performed")
        clean_df = pd.DataFrame(
            [
                {
                    "Timestamp": a.get("timestamp", "—")[:19],
                    "Action": a.get("action", "—"),
                    "Details": a.get("details", "—"),
                    "Affected": a.get("affected_rows", 0),
                }
                for a in report.cleaning_actions
            ]
        )
        st.dataframe(clean_df, use_container_width=True, hide_index=True)

    st.markdown("#### Analyst Notes")
    st.info(report.analyst_notes)


# =============================================================================
# Dashboard Analytics Center
# =============================================================================


def render_dashboard_analytics_center() -> None:
    """Dashboard Analytics Center with Plotly visualizations."""
    st.subheader("Dashboard Analytics Center")
    st.caption("Interactive visualizations and data exploration.")

    df = st.session_state.dataframe
    if df is None:
        st.info("Upload a file to unlock Dashboard Analytics features.")
        return

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Generate Dashboard", type="primary", use_container_width=True):
            with st.spinner("Building visualizations..."):
                try:
                    all_charts = generate_all_charts(df)
                    recommendations = recommend_best_charts(df)
                    st.session_state.dashboard_charts = all_charts
                    st.session_state.dashboard_recommendations = recommendations
                    log_action(
                        action="Generate Dashboard",
                        details="Generated all dashboard visualizations and chart recommendations.",
                        affected_rows=len(df),
                    )
                    st.success("Dashboard generated successfully!")
                except Exception as exc:
                    st.error(f"Dashboard generation failed: {exc}")
                    logger = logging.getLogger(__name__)
                    logger.exception("Dashboard generation error")

    with col2:
        if st.session_state.get("dashboard_recommendations"):
            recs = st.session_state.dashboard_recommendations
            high_priority = [r for r in recs if r.get("priority") == "high"]
            if high_priority:
                st.caption(f"💡 {len(high_priority)} high-priority chart recommendation(s) available.")

    _display_chart_recommendations()

    if st.session_state.get("dashboard_charts"):
        tabs = st.tabs(["Overview", "Trends", "Categories", "Relationships", "Advanced Analytics"])

        with tabs[0]:
            _render_overview_tab()

        with tabs[1]:
            _render_trends_tab()

        with tabs[2]:
            _render_categories_tab()

        with tabs[3]:
            _render_relationships_tab()

        with tabs[4]:
            _render_advanced_analytics_tab()


def _display_chart_recommendations() -> None:
    """Display recommended chart types based on dataset structure."""
    recommendations = st.session_state.get("dashboard_recommendations")
    if not recommendations:
        return

    with st.expander("📋 Chart Recommendations", expanded=False):
        for rec in recommendations:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                rec.get("priority", "low"), "⚪"
            )
            st.markdown(
                f"{priority_emoji} **{rec['chart_type'].title()}** — {rec['reason']} "
                f"*(columns: {', '.join(str(c) for c in rec['columns'])})*"
            )


def _render_chart_group(chart_results: list, max_per_row: int = 2) -> None:
    """Helper to render a group of ChartResult objects in columns."""
    if not chart_results:
        st.info("No charts available for this section.")
        return

    applicable = [c for c in chart_results if c.applicable]
    not_applicable = [c for c in chart_results if not c.applicable]

    if not applicable and not_applicable:
        st.warning(not_applicable[0].description)
        return

    for i in range(0, len(applicable), max_per_row):
        row = st.columns(max_per_row)
        for j, chart in enumerate(applicable[i : i + max_per_row]):
            with row[j]:
                st.plotly_chart(chart.figure, use_container_width=True, key=f"{chart.title}_{i}_{j}")
                st.caption(chart.description)


def _render_overview_tab() -> None:
    """Overview tab: numeric summaries and KPI visualizations."""
    st.markdown("### 📊 Overview")
    st.markdown("Numeric summaries, averages, and key metric visualizations.")

    charts = st.session_state.dashboard_charts
    if charts:
        _render_chart_group(charts.get("numeric_summary", []), max_per_row=2)

        df = st.session_state.dataframe
        if df is not None:
            col_types = detect_column_types(df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Numeric Columns", len(col_types["numeric"]))
            c2.metric("Categorical Columns", len(col_types["categorical"]))
            c3.metric("Date Columns", len(col_types["date"]))
            c4.metric("Total Columns", len(col_types["all"]))


def _render_trends_tab() -> None:
    """Trends tab: distribution analysis and time series."""
    st.markdown("### 📈 Trends & Distributions")
    st.markdown("Histograms, box plots, and temporal trend analysis.")

    charts = st.session_state.dashboard_charts
    if charts:
        _render_chart_group(charts.get("distribution", []), max_per_row=2)

        numeric_summary = charts.get("numeric_summary", [])
        trend_charts = [c for c in numeric_summary if c.chart_type == "line"]
        if trend_charts:
            st.markdown("#### Time Series Trends")
            _render_chart_group(trend_charts, max_per_row=2)


def _render_categories_tab() -> None:
    """Categories tab: pie charts and bar charts for categorical data."""
    st.markdown("### 🗂️ Category Analysis")
    st.markdown("Composition and frequency analysis of categorical columns.")

    charts = st.session_state.dashboard_charts
    if charts:
        _render_chart_group(charts.get("categorical", []), max_per_row=2)

        top_n = charts.get("top_n", [])
        if top_n and any(c.applicable for c in top_n):
            st.markdown("#### Top-N Rankings")
            _render_chart_group(top_n, max_per_row=2)


def _render_relationships_tab() -> None:
    """Relationships tab: scatter plots and correlation heatmap."""
    st.markdown("### 🔗 Relationships")
    st.markdown("Correlation analysis and scatter plot exploration.")

    charts = st.session_state.dashboard_charts
    if charts:
        relationships = charts.get("relationships", [])

        heatmaps = [c for c in relationships if c.chart_type == "heatmap"]
        if heatmaps:
            st.markdown("#### Correlation Matrix")
            for hm in heatmaps:
                if hm.applicable:
                    st.plotly_chart(hm.figure, use_container_width=True, key="corr_heatmap")
                    st.caption(hm.description)
                else:
                    st.warning(hm.description)

        scatters = [c for c in relationships if c.chart_type == "scatter"]
        if scatters:
            st.markdown("#### Scatter Plots")
            _render_chart_group(scatters, max_per_row=2)


def _render_advanced_analytics_tab() -> None:
    """Advanced Analytics tab: comprehensive view and raw data."""
    st.markdown("### 🔬 Advanced Analytics")
    st.markdown("Complete visualization suite and column type breakdown.")

    df = st.session_state.dataframe
    if df is None:
        return

    st.markdown("#### Column Type Detection")
    try:
        col_types = detect_column_types(df)
        type_data = []
        for col in df.columns:
            detected = "other"
            if col in col_types["numeric"]:
                detected = "numeric"
            elif col in col_types["categorical"]:
                detected = "categorical"
            elif col in col_types["date"]:
                detected = "date"
            elif col in col_types["boolean"]:
                detected = "boolean"
            type_data.append({
                "Column": col,
                "Detected Type": detected,
                "Actual Dtype": str(df[col].dtype),
                "Non-Null Count": int(df[col].notna().sum()),
                "Null Count": int(df[col].isna().sum()),
                "Unique Values": int(df[col].nunique()),
            })
        st.dataframe(pd.DataFrame(type_data), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not generate column type breakdown: {exc}")

    charts = st.session_state.dashboard_charts
    if charts:
        st.markdown("#### All Generated Charts")
        all_chart_lists = [
            charts.get("numeric_summary", []),
            charts.get("distribution", []),
            charts.get("categorical", []),
            charts.get("relationships", []),
            charts.get("top_n", []),
        ]
        all_applicable = []
        for chart_list in all_chart_lists:
            all_applicable.extend([c for c in chart_list if c.applicable])

        if all_applicable:
            for i, chart in enumerate(all_applicable):
                st.plotly_chart(chart.figure, use_container_width=True, key=f"all_chart_{i}")
                st.caption(f"**{chart.title}** — {chart.description}")
                st.divider()
        else:
            st.info("No applicable charts were generated for this dataset.")


# =============================================================================
# Main Content Renderer
# =============================================================================


def render_main_content() -> None:
    """Render the main analysis area or an empty-state prompt."""
    df = st.session_state.dataframe

    if df is None:
        st.info("Upload a CSV, XLSX, or XLS file using the sidebar to begin.")
        if st.session_state.upload_error:
            st.error(st.session_state.upload_error)
        return

    health = st.session_state.dataset_health
    if health is None:
        health = compute_dataset_health(df)
        st.session_state.dataset_health = health

    render_data_preview(df)
    st.divider()
    render_dataset_health(health)
    st.divider()
    render_data_quality_report()
    st.divider()
    render_data_cleaning_center()
    st.divider()
    render_basic_statistics(df)
    st.divider()
    render_business_intelligence_center()
    st.divider()
    render_dashboard_analytics_center()


def main() -> None:
    """Main application entry point."""
    configure_page()
    init_session_state()

    st.title(APP_TITLE)
    st.caption(f"Version {APP_VERSION} · Upload, inspect, clean, analyze, and visualize spreadsheet data")

    render_sidebar()

    st.divider()
    render_main_content()


if __name__ == "__m