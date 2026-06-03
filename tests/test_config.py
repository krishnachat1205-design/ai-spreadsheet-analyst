"""
Configuration Tests
===================

Tests for core/config.py path resolution and default values.

Future responsibilities:
------------------------
- Assert PROJECT_ROOT resolves correctly relative to config.py.
- Verify ALLOWED_EXTENSIONS and size limits match expected tuples/ints.
- Test directory path constants are Path instances under PROJECT_ROOT.
"""

from pathlib import Path

from core.config import (
    ALLOWED_EXTENSIONS,
    APP_TITLE,
    APP_VERSION,
    EXPORTS_DIR,
    HISTORY_DIR,
    PROJECT_ROOT,
    UPLOADS_DIR,
)


def test_app_metadata_is_set() -> None:
    """Verify application title and version are non-empty strings."""
    assert isinstance(APP_TITLE, str) and len(APP_TITLE) > 0
    assert isinstance(APP_VERSION, str) and len(APP_VERSION) > 0


def test_project_root_is_directory() -> None:
    """Verify PROJECT_ROOT points to an existing directory."""
    assert PROJECT_ROOT.is_dir()


def test_data_directories_are_under_project_root() -> None:
    """Verify runtime data directories are configured under the project root."""
    for directory in (UPLOADS_DIR, EXPORTS_DIR, HISTORY_DIR):
        assert isinstance(directory, Path)
        assert directory.parent == PROJECT_ROOT or PROJECT_ROOT in directory.parents


def test_allowed_extensions() -> None:
    """Verify allowed file extensions include common spreadsheet formats."""
    assert ".xlsx" in ALLOWED_EXTENSIONS
    assert ".csv" in ALLOWED_EXTENSIONS
