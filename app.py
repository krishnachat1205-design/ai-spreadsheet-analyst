"""
AI Spreadsheet Analyst — Streamlit Application Entry Point
==========================================================

Run with: streamlit run app.py
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv

from collections.abc import Callable
from typing import Any
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
    generate_categories_charts,
    generate_categorical_summary_charts,
    generate_correlation_heatmap,
    generate_distribution_charts,
    generate_numeric_summary_charts,
    generate_overview_charts,
    generate_relationships_charts,
    generate_scatter_plots,
    generate_top_n_charts,
    generate_trends_charts,
    recommend_best_charts,

)
from services.file_service import (
    get_file_metadata,
    load_dataframe_from_upload,
)
from services.kpi_service import generate_kpis, KPIResult
from services.insight_service import generate_insights, generate_recommendations, Insight, Recommendation
from services.report_service import generate_analyst_report, AnalystReport
from services.excel_service import generate_analysis_workbook
from services.pivot_service import generate_pivot_table, get_recommended_pivot_fields, PivotServiceError
from services.formula_service import (
    FormulaResult,
    FormulaServiceError,
    create_calculated_column,
    extract_column_references,
    supported_functions,
    validate_formula,
)
from services.comparison_service import (
    ComparisonResult,
    ComparisonServiceError,
    analyze_datasets,
    generate_comparison_workbook,
)
from services.copilot_service import (
    CopilotServiceError,
    ask_copilot,
    build_full_context,
    generate_analysis_questions,
    generate_executive_summary,
)
from utils.exceptions import FileLoadError, FileServiceError, FileValidationError
from utils.helpers import compute_dataset_health, preview_dataframe
load_dotenv()


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
        "pivot_recommendations": None,
        "pivot_result": None,
        "formula_history": [],
        "comparison_dataframe": None,
        "comparison_result": None,
        "copilot_history": [],
        "copilot_last_response": None,
        "copilot_executive_summary": None,
        "copilot_suggestions": None,
        # Dashboard caching
        "dashboard_cache_version": 0,
        "dashboard_charts_overview": None,
        "dashboard_charts_trends": None,
        "dashboard_charts_categories": None,
        "dashboard_charts_top_n": None,
        "dashboard_charts_relationships_heatmap": None,
        "dashboard_charts_relationships_scatter": None,
        "dashboard_relationships_generated": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _invalidate_dashboard_cache() -> None:
    """Invalidate all dashboard chart caches and bump the version counter."""
    st.session_state.dashboard_cache_version = st.session_state.get("dashboard_cache_version", 0) + 1
    st.session_state.dashboard_charts_overview = None
    st.session_state.dashboard_charts_trends = None
    st.session_state.dashboard_charts_categories = None
    st.session_state.dashboard_charts_top_n = None
    st.session_state.dashboard_charts_relationships_heatmap = None
    st.session_state.dashboard_charts_relationships_scatter = None
    st.session_state.dashboard_relationships_generated = False
    st.session_state.dashboard_charts = None


def _dashboard_cache_key(tab_name: str) -> str:
    """Return the session-state key for a given tab's chart cache."""
    return f"dashboard_charts_{tab_name}"


def _get_cached_charts(
    tab_name: str,
    df: pd.DataFrame,
    generator_fn: Callable[[pd.DataFrame], Any],
    force: bool = False,
) -> Any:
    """
    Retrieve cached charts for a tab if the data version matches.
    Otherwise, generate via *generator_fn*, cache, and return.
    """
    cache_key = _dashboard_cache_key(tab_name)
    current_version = st.session_state.get("dashboard_cache_version", 0)
    cached = st.session_state.get(cache_key)

    if (
        not force
        and cached is not None
        and isinstance(cached, dict)
        and cached.get("version") == current_version
    ):
        return cached["charts"]

    charts = generator_fn(df)
    st.session_state[cache_key] = {"version": current_version, "charts": charts}
    return charts


def _refresh_dataset_state(df: pd.DataFrame) -> None:
    """Sync derived session state after the working DataFrame changes."""
    old_df = st.session_state.get("dataframe")
    if old_df is not None and set(old_df.columns) != set(df.columns):
        st.session_state.pivot_recommendations = None
        st.session_state.pivot_result = None

    st.session_state.dataframe = df
    st.session_state.dataset_health = compute_dataset_health(df)
    st.session_state.quality_report_after = generate_data_quality_report(df)

    metadata = st.session_state.file_metadata
    if metadata is not None:
        metadata["row_count"] = len(df)
        metadata["column_count"] = len(df.columns)
        metadata["column_names"] = [str(col) for col in df.columns]
        st.session_state.file_metadata = metadata

    # Invalidate dashboard caches whenever the dataset changes
    _invalidate_dashboard_cache()


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
        st.session_state.pivot_recommendations = None
        st.session_state.pivot_result = None
        st.session_state.formula_history = []
        st.session_state.comparison_dataframe = None
        st.session_state.comparison_result = None
        st.session_state._last_comparison_upload_key = None
        st.session_state.copilot_history = []
        st.session_state.copilot_last_response = None
        st.session_state.copilot_executive_summary = None
        st.session_state.copilot_suggestions = None
        _invalidate_dashboard_cache()

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
# Formula Intelligence Center
# =============================================================================


def render_formula_intelligence_center() -> None:
    """Formula Intelligence Center for creating calculated columns."""
    st.subheader("Formula Intelligence Center")
    st.caption("Create calculated columns using arithmetic expressions with existing column names.")

    df = st.session_state.dataframe
    if df is None:
        st.info("Upload a file to unlock Formula Intelligence features.")
        return

    with st.expander("📖 Supported Syntax (Version 1)", expanded=False):
        st.markdown("**Arithmetic Operators:** `+`  `-`  `*`  `/`  `**`  `%`  `()`")
        st.markdown("**Constants:** Numeric literals (e.g., `0.18`)")
        st.markdown("**References:** Use exact column names (case-sensitive).")
        st.markdown("**Examples:**")
        st.markdown("- `Quantity * Price`")
        st.markdown("- `Revenue - Cost`")
        st.markdown("- `(Revenue - Cost) / Revenue`")
        st.markdown("- `Revenue * 0.18`")

    col1, col2 = st.columns(2)
    with col1:
        new_col_name = st.text_input("New Column Name", placeholder="e.g., Revenue")
    with col2:
        formula_input = st.text_input("Formula", placeholder="e.g., Quantity * Price")

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("Validate Formula", use_container_width=True):
            if not formula_input:
                st.warning("Enter a formula to validate.")
            else:
                try:
                    is_valid, msg = validate_formula(formula_input, df.columns.tolist())
                    if is_valid:
                        refs = extract_column_references(formula_input)
                        st.success(f"Formula is valid. References columns: {', '.join(refs)}")
                    else:
                        st.error(msg)
                except FormulaServiceError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Validation failed: {exc}")

    with btn_col2:
        if st.button("Create Column", type="primary", use_container_width=True):
            if not new_col_name or not formula_input:
                st.warning("Enter both a column name and a formula.")
            elif new_col_name in df.columns:
                st.error(f"Column '{new_col_name}' already exists. Choose a different name.")
            else:
                try:
                    result = create_calculated_column(df, new_col_name, formula_input)
                    _refresh_dataset_state(result.dataframe)

                    st.session_state.formula_history.append({
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "column_name": result.new_column,
                        "formula": result.formula,
                        "affected_rows": result.affected_rows,
                    })

                    log_action(
                        action="Create Calculated Column",
                        details=f"Created '{result.new_column}' = {result.formula}",
                        affected_rows=result.affected_rows,
                    )
                    st.success(
                        f"Column '{result.new_column}' created successfully. "
                        f"Affected rows: {result.affected_rows:,}"
                    )
                except FormulaServiceError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Failed to create column: {exc}")

    history = st.session_state.get("formula_history", [])
    if history:
        st.markdown("---")
        st.markdown("### 🧮 Formula History")
        for entry in reversed(history[-10:]):
            st.markdown(
                f"**{entry['column_name']}** = `{entry['formula']}` "
                f"· {entry['timestamp'][:19]} · Affected: {entry['affected_rows']:,}"
            )


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
    _render_export_button()


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

def _render_export_button() -> None:
    """Render the Excel export download button for the analysis workbook."""
    st.markdown("---")
    st.markdown("### 📥 Export")

    df = st.session_state.dataframe
    if df is None:
        return

    kpis = st.session_state.get("bi_kpis")
    insights = st.session_state.get("bi_insights")
    recommendations = st.session_state.get("bi_recommendations")
    report = st.session_state.get("bi_report")

    has_analysis = any([
        kpis is not None and len(kpis) > 0,
        insights is not None and len(insights) > 0,
        recommendations is not None and len(recommendations) > 0,
        report is not None,
    ])

    if not has_analysis:
        st.info("Generate KPIs, Insights, Recommendations, or an Analyst Report to enable export.")
        return

    if st.button("📥 Download Analysis Workbook", type="primary", use_container_width=True):
        with st.spinner("Building workbook..."):
            buffer = generate_analysis_workbook(
                df=df, kpis=kpis, insights=insights,
                recommendations=recommendations, report=report,
            )
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_workbook_{timestamp}.xlsx"

            st.download_button(
                label="⬇️ Click to Download",
                data=buffer, file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            log_action(action="Export Workbook", details=f"Exported '{filename}'", affected_rows=len(df))


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
                    recommendations = recommend_best_charts(df)
                    st.session_state.dashboard_recommendations = recommendations
                    _invalidate_dashboard_cache()
                    log_action(
                        action="Generate Dashboard",
                        details="Activated lazy-loaded dashboard with chart recommendations.",
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

    if st.session_state.get("dashboard_recommendations"):
        tabs = st.tabs(["Overview", "Trends", "Categories", "Relationships", "Advanced Analytics"])

        with tabs[0]:
            _render_overview_tab(df)

        with tabs[1]:
            _render_trends_tab(df)

        with tabs[2]:
            _render_categories_tab(df)

        with tabs[3]:
            _render_relationships_tab(df)

        with tabs[4]:
            _render_advanced_analytics_tab(df)


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


def _render_overview_tab(df: pd.DataFrame) -> None:
    """Overview tab: numeric summaries and KPI visualizations."""
    st.markdown("### 📊 Overview")
    st.markdown("Numeric summaries, averages, and key metric visualizations.")

    with st.spinner("Loading overview..."):
        charts = _get_cached_charts("overview", df, generate_overview_charts)
    _render_chart_group(charts, max_per_row=2)

    col_types = detect_column_types(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numeric Columns", len(col_types["numeric"]))
    c2.metric("Categorical Columns", len(col_types["categorical"]))
    c3.metric("Date Columns", len(col_types["date"]))
    c4.metric("Total Columns", len(col_types["all"]))


def _render_trends_tab(df: pd.DataFrame) -> None:
    """Trends tab: distribution analysis and time series."""
    st.markdown("### 📈 Trends & Distributions")
    st.markdown("Histograms, box plots, and temporal trend analysis.")

    with st.spinner("Loading distributions..."):
        charts = _get_cached_charts("trends", df, generate_trends_charts)
    _render_chart_group(charts, max_per_row=2)

    with st.spinner("Loading time series..."):
        overview_charts = _get_cached_charts("overview", df, generate_overview_charts)
    trend_charts = [c for c in overview_charts if c.chart_type == "line"]
    if trend_charts:
        st.markdown("#### Time Series Trends")
        _render_chart_group(trend_charts, max_per_row=2)


def _render_categories_tab(df: pd.DataFrame) -> None:
    """Categories tab: pie charts and bar charts for categorical data."""
    st.markdown("### 🗂️ Category Analysis")
    st.markdown("Composition and frequency analysis of categorical columns.")

    with st.spinner("Loading category charts..."):
        cat_charts = _get_cached_charts("categories", df, generate_categorical_summary_charts)
    _render_chart_group(cat_charts, max_per_row=2)

    with st.spinner("Loading top-N rankings..."):
        top_n = _get_cached_charts("top_n", df, generate_top_n_charts)
    if top_n and any(c.applicable for c in top_n):
        st.markdown("#### Top-N Rankings")
        _render_chart_group(top_n, max_per_row=2)


def _render_relationships_tab(df: pd.DataFrame) -> None:
    """Relationships tab: scatter plots and correlation heatmap."""
    st.markdown("### 🔗 Relationships")
    st.markdown("Correlation analysis and scatter plot exploration.")

    n_rows = len(df)
    is_large = n_rows > 10000

    # For large datasets, require explicit user opt-in to avoid blocking
    if is_large and not st.session_state.get("dashboard_relationships_generated", False):
        st.info(
            f"📊 **Performance Mode Active**\n\n"
            f"Your dataset has **{n_rows:,} rows**. To keep the dashboard responsive, "
            f"relationship visualizations are generated on-demand.\n\n"
            f"Click below to build the correlation heatmap and scatter plots."
        )
        if st.button("🔍 Generate Relationship Analysis", key="btn_gen_relationships", use_container_width=True):
            st.session_state.dashboard_relationships_generated = True
            # Pre-warm cache so charts appear instantly on rerun
            with st.spinner("Preparing relationship analysis..."):
                _get_cached_charts("relationships_heatmap", df, generate_correlation_heatmap, force=True)
                _get_cached_charts("relationships_scatter", df, generate_scatter_plots, force=True)
            st.rerun()
        return

    with st.spinner("Loading correlation matrix..."):
        hm = _get_cached_charts("relationships_heatmap", df, generate_correlation_heatmap)
    st.markdown("#### Correlation Matrix")
    if hm.applicable:
        st.plotly_chart(hm.figure, use_container_width=True, key="corr_heatmap")
        st.caption(hm.description)
    else:
        st.warning(hm.description)

    with st.spinner("Loading scatter plots..."):
        scatters = _get_cached_charts("relationships_scatter", df, generate_scatter_plots)
    if scatters and any(c.applicable for c in scatters):
        st.markdown("#### Scatter Plots")
        _render_chart_group(scatters, max_per_row=2)


def _render_advanced_analytics_tab(df: pd.DataFrame) -> None:
    """Advanced Analytics tab: comprehensive view and raw data."""
    st.markdown("### 🔬 Advanced Analytics")
    st.markdown("Complete visualization suite and column type breakdown.")

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

            type_data.append(
                {
                    "Column": col,
                    "Detected Type": detected,
                    "Actual Dtype": str(df[col].dtype),
                    "Non-Null Count": int(df[col].notna().sum()),
                    "Null Count": int(df[col].isna().sum()),
                    "Unique Values": int(df[col].nunique()),
                }
            )

        st.dataframe(
            pd.DataFrame(type_data),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.warning(f"Could not generate column type breakdown: {exc}")

    st.markdown("#### All Generated Charts")

    current_version = st.session_state.get("dashboard_cache_version", 0)
    all_charts = []

    for tab_name in (
        "overview",
        "trends",
        "categories",
        "relationships_heatmap",
        "relationships_scatter",
    ):
        cache_key = _dashboard_cache_key(tab_name)
        cached = st.session_state.get(cache_key)

        if (
            cached
            and isinstance(cached, dict)
            and cached.get("version") == current_version
        ):
            charts = cached.get("charts")

            if isinstance(charts, list):
                all_charts.extend(
                    [
                        c
                        for c in charts
                        if getattr(c, "applicable", True)
                    ]
                )

            elif charts is not None:
                if getattr(charts, "applicable", True):
                    all_charts.append(charts)

    if all_charts:
        for i, chart in enumerate(all_charts):
            st.plotly_chart(
                chart.figure,
                use_container_width=True,
                key=f"all_chart_{i}",
            )
            st.caption(f"**{chart.title}** — {chart.description}")
            st.divider()

    else:
        st.info(
            "No charts cached yet. Visit the Overview, Trends, Categories, "
            "or Relationships tabs to generate charts first."
        )

        if st.button(
            "⚡ Force Load All Charts",
            key="btn_advanced_load_all",
            use_container_width=True,
        ):
            with st.spinner(
                "Building all visualizations… This may take a moment."
            ):
                _get_cached_charts(
                    "overview",
                    df,
                    generate_overview_charts,
                    force=True,
                )

                _get_cached_charts(
                    "trends",
                    df,
                    generate_trends_charts,
                    force=True,
                )

                _get_cached_charts(
                    "categories",
                    df,
                    generate_categorical_summary_charts,
                    force=True,
                )

                _get_cached_charts(
                    "top_n",
                    df,
                    generate_top_n_charts,
                    force=True,
                )

                _get_cached_charts(
                    "relationships_heatmap",
                    df,
                    generate_correlation_heatmap,
                    force=True,
                )

                _get_cached_charts(
                    "relationships_scatter",
                    df,
                    generate_scatter_plots,
                    force=True,
                )

            st.rerun()




# =============================================================================
# Pivot Intelligence Center
# =============================================================================


def render_pivot_intelligence_center() -> None:
    """Pivot Intelligence Center for interactive pivot table generation."""
    st.subheader("Pivot Intelligence Center")
    st.caption("Build custom pivot tables from your dataset.")

    df = st.session_state.dataframe
    if df is None:
        st.info("Upload a file to unlock Pivot Intelligence features.")
        return

    recommendations = st.session_state.get("pivot_recommendations")
    if recommendations is None:
        recommendations = get_recommended_pivot_fields(df)
        st.session_state.pivot_recommendations = recommendations

    recs = recommendations
    row_candidates = recs.get("row_candidates", [])
    col_candidates = recs.get("col_candidates", [])
    value_candidates = recs.get("value_candidates", [])

    if not row_candidates or not value_candidates:
        st.warning("This dataset does not have suitable fields for pivot analysis.")
        return

    st.markdown("#### Field Selection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        row_field = st.selectbox(
            "Row Field",
            options=row_candidates,
            index=0 if row_candidates else None,
            help="Categorical field to group rows by",
        )

    with col2:
        col_field = st.selectbox(
            "Column Field (Optional)",
            options=["None"] + col_candidates,
            index=0,
            help="Optional categorical field for column breakdown",
        )

    with col3:
        value_field = st.selectbox(
            "Value Field",
            options=value_candidates,
            index=0 if value_candidates else None,
            help="Numeric field to aggregate",
        )

    with col4:
        agg_function = st.selectbox(
            "Aggregation",
            options=["sum", "mean", "count", "min", "max"],
            index=0,
            help="How to aggregate the value field",
        )

    suggested = recs.get("suggested_combinations", [])
    if suggested:
        with st.expander("💡 Suggested Combinations", expanded=False):
            for combo in suggested[:5]:
                st.markdown(f"• **{combo['reason']}** — Rows: `{combo['rows'][0]}`, Values: `{combo['values'][0]}`, Agg: `{combo['agg']}`")

    if st.button("Generate Pivot Table", type="primary", use_container_width=True):
        with st.spinner("Building pivot table..."):
            try:
                rows = [row_field]
                values = [value_field]
                columns = [col_field] if col_field != "None" else None

                result = generate_pivot_table(
                    df=df,
                    rows=rows,
                    values=values,
                    agg_function=agg_function,
                    columns=columns,
                )
                st.session_state.pivot_result = result

                log_action(
                    action="Generate Pivot Table",
                    details=result.message,
                    affected_rows=len(df),
                )
                st.success(result.message)
            except PivotServiceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    _display_pivot_result()


def _display_pivot_result() -> None:
    """Display the generated pivot table with metrics and export option."""
    result = st.session_state.get("pivot_result")
    if not result:
        return

    st.markdown("---")
    st.markdown("### 📊 Pivot Table Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{result.shape[0]:,}")
    c2.metric("Columns", result.shape[1])
    c3.metric("Total Cells", f"{result.total_cells:,}")

    config_parts = [
        f"**Rows:** {', '.join(result.rows)}",
        f"**Values:** {', '.join(result.values)}",
        f"**Aggregation:** `{result.agg_function}`",
    ]
    if result.columns:
        config_parts.insert(1, f"**Columns:** {', '.join(result.columns)}")
    st.markdown(" | ".join(config_parts))

    st.dataframe(result.pivot_table, use_container_width=True, hide_index=True)

    csv = result.pivot_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Pivot Table as CSV",
        data=csv,
        file_name=f"pivot_table_{result.agg_function}_{'_'.join(result.rows)}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =============================================================================
# Multi-File Comparison Center
# =============================================================================


def render_multi_file_comparison_center() -> None:
    """Multi-File Comparison Center for comparing two datasets."""
    st.subheader("Multi-File Comparison Center")
    st.caption("Upload a second dataset and compare it against the primary dataset.")

    df_a = st.session_state.dataframe
    if df_a is None:
        st.info("Upload a primary file first to unlock Multi-File Comparison.")
        return

    st.markdown("#### Upload Comparison Dataset")
    uploaded_b = st.file_uploader(
        "Choose comparison spreadsheet",
        type=["csv", "xlsx", "xls"],
        key="comparison_file_uploader",
        help="Supported formats: CSV, XLSX, XLS",
    )

    if uploaded_b is not None:
        upload_key_b = f"{uploaded_b.name}_{uploaded_b.size}"
        if st.session_state.get("_last_comparison_upload_key") != upload_key_b:
            try:
                content = uploaded_b.getvalue()
                df_b, _saved_path, _validation = load_dataframe_from_upload(
                    content, uploaded_b.name
                )
                st.session_state.comparison_dataframe = df_b
                st.session_state._last_comparison_upload_key = upload_key_b
                st.session_state.comparison_result = None
                st.success(
                    f"Loaded comparison file **{uploaded_b.name}** "
                    f"({len(df_b):,} rows, {len(df_b.columns)} columns)."
                )
            except Exception as exc:
                st.error(f"Failed to load comparison file: {exc}")
                st.session_state.comparison_dataframe = None
                st.session_state.comparison_result = None

    df_b = st.session_state.get("comparison_dataframe")
    if df_b is None:
        st.info("Upload a comparison file to proceed.")
        return

    common_cols = list(set(df_a.columns) & set(df_b.columns))
    if not common_cols:
        st.error("No common columns found between the two datasets.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        join_key = st.selectbox(
            "Select Join Key",
            options=common_cols,
            help="Column used to match records between datasets (must be unique).",
        )
    with col2:
        if st.button("Run Comparison", type="primary", use_container_width=True):
            with st.spinner("Running comparison..."):
                try:
                    result = analyze_datasets(df_a, df_b, join_key)
                    st.session_state.comparison_result = result
                    log_action(
                        action="Run Comparison",
                        details=(
                            f"Compared using key '{join_key}'. "
                            f"Added: {result.added_count}, "
                            f"Removed: {result.removed_count}, "
                            f"Modified: {result.modified_count}"
                        ),
                        affected_rows=max(result.dataset_a_rows, result.dataset_b_rows),
                    )
                    st.success("Comparison complete!")
                except ComparisonServiceError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")

    result = st.session_state.get("comparison_result")
    if result is None:
        return

    tabs = st.tabs(
        ["Summary", "Added Records", "Removed Records", "Modified Records", "Numeric Differences"]
    )

    with tabs[0]:
        st.markdown("### 📊 Comparison Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset A Rows", f"{result.dataset_a_rows:,}")
        c2.metric("Dataset B Rows", f"{result.dataset_b_rows:,}")
        c3.metric("Matched Records", f"{result.matched_records:,}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Added Records", f"{result.added_count:,}")
        c5.metric("Removed Records", f"{result.removed_count:,}")
        c6.metric("Modified Records", f"{result.modified_count:,}")

        st.markdown("---")
        st.markdown(f"**Common Columns:** {result.summary.get('common_columns', 0)}")
        st.markdown(f"**Numeric Columns Compared:** {result.summary.get('numeric_columns_compared', 0)}")

    with tabs[1]:
        st.markdown("### ➕ Added Records")
        if result.added_records.empty:
            st.info("No added records found.")
        else:
            st.dataframe(result.added_records, use_container_width=True, hide_index=True)
            st.caption(f"{len(result.added_records):,} records present in Dataset B but not in Dataset A.")

    with tabs[2]:
        st.markdown("### ➖ Removed Records")
        if result.removed_records.empty:
            st.info("No removed records found.")
        else:
            st.dataframe(result.removed_records, use_container_width=True, hide_index=True)
            st.caption(f"{len(result.removed_records):,} records present in Dataset A but not in Dataset B.")

    with tabs[3]:
        st.markdown("### ✏️ Modified Records")
        if result.modified_records.empty:
            st.info("No modified records found.")
        else:
            st.dataframe(result.modified_records, use_container_width=True, hide_index=True)
            st.caption(f"{len(result.modified_records):,} field-level changes detected.")

    with tabs[4]:
        st.markdown("### 🔢 Numeric Differences")
        if result.numeric_comparison.empty:
            st.info("No common numeric columns to compare.")
        else:
            st.dataframe(result.numeric_comparison, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("📥 Download Comparison Report", type="primary", use_container_width=True):
        with st.spinner("Building comparison workbook..."):
            buffer = generate_comparison_workbook(result)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Comparison_Report_{timestamp}.xlsx"
            st.download_button(
                label="⬇️ Click to Download",
                data=buffer,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            log_action(
                action="Export Comparison Report",
                details=f"Exported '{filename}'",
                affected_rows=result.dataset_a_rows + result.dataset_b_rows,
            )


# =============================================================================
# AI Business Analyst Copilot
# =============================================================================


def render_ai_business_analyst_copilot() -> None:
    """AI Business Analyst Copilot powered by Gemini."""
    st.subheader("🤖 AI Business Analyst Copilot")
    st.caption("Ask questions, generate executive summaries, and get AI-powered analysis suggestions.")

    df = st.session_state.dataframe
    if df is None:
        st.info("Upload a file to unlock the AI Business Analyst Copilot.")
        return

    # API key check
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("🔑 **GEMINI_API_KEY not configured.**")
        st.markdown(
            "Set the `GEMINI_API_KEY` environment variable and refresh the page to enable AI features. "
            "All other platform features remain fully functional."
        )
        return

    tabs = st.tabs(["💬 Ask Copilot", "📋 Executive Summary", "💡 Suggested Analysis"])

    with tabs[0]:
        _render_ask_copilot_tab()

    with tabs[1]:
        _render_executive_summary_tab()

    with tabs[2]:
        _render_suggested_analysis_tab()


def _build_copilot_context() -> str:
    """Build the full dataset context for the copilot from session state."""
    df = st.session_state.dataframe
    return build_full_context(
        df=df,
        metadata=st.session_state.get("file_metadata"),
        health=st.session_state.get("dataset_health"),
        quality_report=st.session_state.get("quality_report_after"),
        kpis=st.session_state.get("bi_kpis"),
        insights=st.session_state.get("bi_insights"),
        recommendations=st.session_state.get("bi_recommendations"),
        report=st.session_state.get("bi_report"),
        charts=st.session_state.get("dashboard_charts"),
        chart_recommendations=st.session_state.get("dashboard_recommendations"),
        pivot_result=st.session_state.get("pivot_result"),
        comparison_result=st.session_state.get("comparison_result"),
        formula_history=st.session_state.get("formula_history"),
    )


def _render_ask_copilot_tab() -> None:
    """Render the interactive Q&A tab."""
    st.markdown("### 💬 Ask Copilot")
    st.caption(
        "Ask any business or data analysis question. The AI uses your dataset, KPIs, insights, "
        "reports, and comparisons to answer."
    )

    context = _build_copilot_context()

    question = st.text_area(
        "Your question",
        placeholder=(
            "e.g., What are the key risks in this dataset? "
            "What charts should I create? "
            "How do I calculate profit margin? "
            "What changed between the files?"
        ),
        height=100,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_clicked = st.button("Ask Copilot", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.copilot_history = []
            st.session_state.copilot_last_response = None
            st.rerun()

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Consulting AI Business Analyst..."):
                try:
                    history = st.session_state.get("copilot_history", [])
                    response = ask_copilot(
                        question=question.strip(),
                        full_context=context,
                        history=history,
                    )
                    st.session_state.copilot_last_response = response

                    history.append({
                        "question": question.strip(),
                        "answer": response.answer,
                        "timestamp": pd.Timestamp.now().isoformat(),
                    })
                    st.session_state.copilot_history = history

                    log_action(
                        action="Copilot Question",
                        details=f"Q: {question[:120]}",
                        affected_rows=0,
                    )
                    st.success("Response received!")
                except CopilotServiceError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Copilot error: {exc}")

    # Display last response
    response = st.session_state.get("copilot_last_response")
    if response:
        st.markdown("---")
        st.markdown("#### 🤖 Copilot Response")
        st.markdown(response.answer)

        if response.suggested_actions:
            st.markdown("**Suggested Actions:**")
            for action in response.suggested_actions:
                st.markdown(f"- {action}")

    # Conversation history
    history = st.session_state.get("copilot_history", [])
    if len(history) > 1:
        st.markdown("---")
        st.markdown("#### 📝 Conversation History")
        for entry in reversed(history[:-1]):
            with st.expander(f"Q: {entry['question'][:60]}...", expanded=False):
                st.markdown(f"**Q:** {entry['question']}")
                st.markdown(f"**A:** {entry['answer']}")
                st.caption(f"{entry.get('timestamp', '')[:19]}")


def _render_executive_summary_tab() -> None:
    """Render the executive summary generation tab."""
    st.markdown("### 📋 Executive Summary")
    st.caption("Generate a comprehensive executive summary based on all available analysis.")

    if st.button("Generate Executive Summary", type="primary", use_container_width=True):
        with st.spinner("Generating executive summary with Gemini..."):
            try:
                context = _build_copilot_context()
                response = generate_executive_summary(full_context=context)
                st.session_state.copilot_executive_summary = response

                log_action(
                    action="Generate Executive Summary",
                    details="AI-generated executive summary via Gemini",
                    affected_rows=0,
                )
                st.success("Executive summary generated!")
            except CopilotServiceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Failed to generate summary: {exc}")

    summary = st.session_state.get("copilot_executive_summary")
    if summary:
        st.markdown("---")
        st.markdown(summary.answer)


def _render_suggested_analysis_tab() -> None:
    """Render the suggested analysis tab."""
    st.markdown("### 💡 Suggested Analysis")
    st.caption("AI-generated suggestions for questions, charts, pivot tables, and formulas.")

    if st.button("Generate Suggestions", type="primary", use_container_width=True):
        with st.spinner("Analyzing dataset for suggestions..."):
            try:
                context = _build_copilot_context()
                suggestions = generate_analysis_questions(full_context=context)
                st.session_state.copilot_suggestions = suggestions

                log_action(
                    action="Generate Analysis Suggestions",
                    details="AI-generated analysis suggestions via Gemini",
                    affected_rows=0,
                )
                st.success("Suggestions generated!")
            except CopilotServiceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Failed to generate suggestions: {exc}")

    suggestions = st.session_state.get("copilot_suggestions")
    if suggestions:
        st.markdown("---")

        if suggestions.get("questions"):
            st.markdown("#### ❓ Suggested Questions")
            for q in suggestions["questions"]:
                st.markdown(f"- {q}")
            st.markdown("")

        if suggestions.get("charts"):
            st.markdown("#### 📊 Suggested Charts")
            for c in suggestions["charts"]:
                st.markdown(f"- {c}")
            st.markdown("")

        if suggestions.get("pivots"):
            st.markdown("#### 🔄 Suggested Pivot Tables")
            for p in suggestions["pivots"]:
                st.markdown(f"- {p}")
            st.markdown("")

        if suggestions.get("formulas"):
            st.markdown("#### 🧮 Suggested Formulas")
            for f in suggestions["formulas"]:
                st.markdown(f"- {f}")


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
    render_formula_intelligence_center()
    st.divider()
    render_basic_statistics(df)
    st.divider()
    render_business_intelligence_center()
    st.divider()
    render_dashboard_analytics_center()
    st.divider()
    render_pivot_intelligence_center()
    st.divider()
    render_multi_file_comparison_center()
    st.divider()
    render_ai_business_analyst_copilot()


def main() -> None:
    """Main application entry point."""
    configure_page()
    init_session_state()

    st.title(APP_TITLE)
    st.caption(f"Version {APP_VERSION} · Upload, inspect, clean, analyze, and visualize spreadsheet data")

    render_sidebar()

    st.divider()
    render_main_content()


if __name__ == "__main__":
    main()