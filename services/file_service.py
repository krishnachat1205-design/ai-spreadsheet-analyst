"""
File Service
============

Handles file upload, validation, storage, and DataFrame ingestion for
CSV, XLSX, and XLS spreadsheets.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Literal

import pandas as pd

from core.config import (
    ALLOWED_EXTENSIONS,
    DEFAULT_ENCODING,
    DEFAULT_SHEET_INDEX,
    EXPORT_TIMESTAMP_FORMAT,
    MAX_UPLOAD_SIZE_BYTES,
    UPLOADS_DIR,
)
from utils.exceptions import FileLoadError, FileValidationError
from utils.helpers import (
    ensure_directory,
    format_file_size,
    generate_stored_filename,
    safe_filename,
)

logger = logging.getLogger(__name__)

FileType = Literal["csv", "xlsx", "xls"]

# Maps file extensions to internal type labels and pandas read hints.
_EXTENSION_MAP: dict[str, FileType] = {
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".xls": "xls",
}

# Fallback encodings tried when DEFAULT_ENCODING fails for CSV files.
_CSV_ENCODINGS: tuple[str, ...] = (DEFAULT_ENCODING, "utf-8-sig", "latin-1", "cp1252")


def get_uploads_directory() -> Path:
    """Return the configured uploads directory, creating it if needed."""
    return ensure_directory(UPLOADS_DIR)


def detect_file_type(filename: str) -> FileType:
    """
    Detect spreadsheet type from the file extension.

    Raises:
        FileValidationError: If the extension is missing or unsupported.
    """
    suffix = Path(filename).suffix.lower()
    if not suffix:
        raise FileValidationError("File has no extension. Upload a .csv, .xlsx, or .xls file.")
    file_type = _EXTENSION_MAP.get(suffix)
    if file_type is None:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise FileValidationError(
            f"Unsupported file type '{suffix}'. Allowed extensions: {allowed}."
        )
    return file_type


def validate_file(
    filename: str,
    file_size: int,
    *,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> dict[str, Any]:
    """
    Validate an uploaded file before saving or parsing.

    Returns:
        dict with keys: valid (bool), file_type, filename, file_size,
        file_size_human, message.

    Raises:
        FileValidationError: When validation fails.
    """
    if not filename or not str(filename).strip():
        raise FileValidationError("No filename provided.")

    clean_name = safe_filename(Path(filename).name)
    if clean_name != Path(filename).name:
        logger.warning("Filename sanitized from %r to %r", filename, clean_name)

    file_type = detect_file_type(clean_name)

    if file_size <= 0:
        raise FileValidationError("Uploaded file is empty.")

    if file_size > max_size_bytes:
        raise FileValidationError(
            f"File size ({format_file_size(file_size)}) exceeds the "
            f"{format_file_size(max_size_bytes)} limit."
        )

    return {
        "valid": True,
        "file_type": file_type,
        "filename": clean_name,
        "file_size": file_size,
        "file_size_human": format_file_size(file_size),
        "message": "File passed validation.",
    }


def save_uploaded_file(
    file_content: bytes | BinaryIO,
    original_filename: str,
) -> Path:
    """
    Validate and persist uploaded file bytes to the uploads directory.

    Args:
        file_content: Raw bytes or a readable binary stream.
        original_filename: Original name from the upload widget.

    Returns:
        Path to the saved file on disk.

    Raises:
        FileValidationError: If validation fails.
        FileServiceError: If the file cannot be written.
    """
    if isinstance(file_content, bytes):
        content = file_content
    else:
        content = file_content.read()

    validation = validate_file(original_filename, len(content))
    uploads_dir = get_uploads_directory()

    stored_name = generate_stored_filename(validation["filename"])
    destination = uploads_dir / stored_name

    try:
        destination.write_bytes(content)
        logger.info("Saved upload to %s", destination)
    except OSError as exc:
        raise FileLoadError(f"Could not save file to disk: {exc}") from exc

    return destination


def _read_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file, trying multiple encodings."""
    last_error: Exception | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.debug("CSV loaded with encoding %s", encoding)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise FileLoadError("CSV file contains no data.") from exc
        except pd.errors.ParserError as exc:
            raise FileLoadError(f"Could not parse CSV file: {exc}") from exc

    raise FileLoadError(
        f"Could not decode CSV file with supported encodings. Last error: {last_error}"
    )


def _read_excel(file_path: Path, file_type: FileType, sheet_name: int | str | None) -> pd.DataFrame:
    """Load an Excel workbook using the appropriate pandas engine."""
    sheet = DEFAULT_SHEET_INDEX if sheet_name is None else sheet_name
    engine = "xlrd" if file_type == "xls" else "openpyxl"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet, engine=engine)
    except ImportError as exc:
        if file_type == "xls":
            raise FileLoadError(
                "Reading .xls files requires the 'xlrd' package. "
                "Install it with: pip install xlrd"
            ) from exc
        raise FileLoadError(f"Missing Excel reader dependency: {exc}") from exc
    except ValueError as exc:
        raise FileLoadError(f"Excel sheet error: {exc}") from exc
    except Exception as exc:
        raise FileLoadError(f"Could not read Excel file: {exc}") from exc

    return df


def load_dataframe(
    file_path: Path | str,
    *,
    sheet_name: int | str | None = None,
) -> pd.DataFrame:
    """
    Load a spreadsheet file into a pandas DataFrame.

    Automatically detects CSV vs Excel from the file extension.

    Raises:
        FileValidationError: For unsupported extensions.
        FileLoadError: When parsing fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileLoadError(f"File not found: {path}")

    file_type = detect_file_type(path.name)

    try:
        if file_type == "csv":
            df = _read_csv(path)
        else:
            df = _read_excel(path, file_type, sheet_name)
    except FileLoadError:
        raise
    except Exception as exc:
        raise FileLoadError(f"Unexpected error loading file: {exc}") from exc

    if df.empty:
        logger.warning("Loaded DataFrame is empty: %s", path.name)

    return df


def load_dataframe_from_upload(
    file_content: bytes,
    original_filename: str,
    *,
    sheet_name: int | str | None = None,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """
    Validate, save, and load an uploaded file in one step.

    Returns:
        Tuple of (DataFrame, saved_path, validation_info).
    """
    validation = validate_file(original_filename, len(file_content))
    saved_path = save_uploaded_file(file_content, original_filename)
    df = load_dataframe(saved_path, sheet_name=sheet_name)
    return df, saved_path, validation


def _excel_sheet_names(file_path: Path, file_type: FileType) -> list[str]:
    """Return sheet names for Excel files; empty list for CSV."""
    if file_type == "csv":
        return []
    engine = "xlrd" if file_type == "xls" else "openpyxl"
    try:
        workbook = pd.ExcelFile(file_path, engine=engine)
        return list(workbook.sheet_names)
    except Exception as exc:
        logger.warning("Could not read sheet names from %s: %s", file_path, exc)
        return []


def get_file_metadata(
    file_path: Path | str,
    df: pd.DataFrame | None = None,
    *,
    original_filename: str | None = None,
) -> dict[str, Any]:
    """
    Collect descriptive metadata for a saved file and optional DataFrame.

    Returns:
        dict with filename, path, file_type, size, timestamps, dimensions,
        column names, and sheet names (Excel only).
    """
    path = Path(file_path)
    file_type = detect_file_type(path.name)
    stat = path.stat()

    metadata: dict[str, Any] = {
        "original_filename": original_filename or path.name,
        "stored_filename": path.name,
        "file_path": str(path.resolve()),
        "file_type": file_type,
        "extension": path.suffix.lower(),
        "file_size": stat.st_size,
        "file_size_human": format_file_size(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sheet_names": _excel_sheet_names(path, file_type),
        "loaded_sheet": DEFAULT_SHEET_INDEX if file_type != "csv" else None,
    }

    if df is not None:
        metadata.update(
            {
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "column_names": [str(col) for col in df.columns],
                "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            }
        )

    return metadata
