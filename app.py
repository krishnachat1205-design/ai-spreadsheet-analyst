"""
AI Spreadsheet Analyst — Streamlit Application Entry Point
==========================================================

Run with:  streamlit run app.py
"""

from __future__ import annotations

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
from services.file_service import (
    get_file_metadata,
    load_dataframe_from_upload,
)
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


def main() -> None:
    """Main application entry point."""
    configure_page()
    init_session_state()

    st.title(APP_TITLE)
    st.caption(f"Version {APP_VERSION} · Upload, inspect, clean, and analyze spreadsheet data")

    render_sidebar()

    st.divider()
    render_main_content()


if __name__ == "__main__":
    main()
