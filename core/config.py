"""
Application Configuration
=========================

Central configuration module for paths, constants, and runtime settings.
All environment-specific values should be defined or loaded here.

Future responsibilities:
------------------------
- Load settings from environment variables (.env) for secrets and overrides.
- Define base directory paths for uploads, exports, history, and assets.
- Expose file-size limits, allowed extensions, and parsing defaults.
- Configure logging levels and output destinations.
- Provide Plotly/chart theming defaults for dashboard_service.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

APP_TITLE: str = "AI Spreadsheet Analyst"
APP_VERSION: str = "0.1.0"
APP_DESCRIPTION: str = (
    "An intelligent tool for uploading, cleaning, analyzing, and exporting "
    "spreadsheet data with AI-powered insights."
)

# ---------------------------------------------------------------------------
# Streamlit UI defaults
# ---------------------------------------------------------------------------

PAGE_LAYOUT: str = "wide"
SIDEBAR_STATE: str = "expanded"

# ---------------------------------------------------------------------------
# Directory paths (resolved relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"
EXPORTS_DIR: Path = PROJECT_ROOT / "exports"
HISTORY_DIR: Path = PROJECT_ROOT / "history"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: tuple[str, ...] = (".xlsx", ".xls", ".csv")
MAX_UPLOAD_SIZE_MB: int = 50
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Data processing defaults
# ---------------------------------------------------------------------------

DEFAULT_SHEET_INDEX: int = 0
DEFAULT_ENCODING: str = "utf-8"
NULL_PLACEHOLDERS: tuple[str, ...] = ("", "NA", "N/A", "null", "None", "-")

# ---------------------------------------------------------------------------
# Export defaults
# ---------------------------------------------------------------------------

EXPORT_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
DEFAULT_EXPORT_FORMAT: str = "xlsx"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
