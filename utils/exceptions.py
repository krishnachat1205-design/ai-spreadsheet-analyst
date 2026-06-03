"""
Custom Exceptions
=================

Application-specific exception types for consistent error handling
across services and the UI layer.
"""


class FileServiceError(Exception):
    """Base exception for file upload and ingestion operations."""


class FileValidationError(FileServiceError):
    """Raised when an uploaded file fails validation checks."""


class FileLoadError(FileServiceError):
    """Raised when a file cannot be parsed into a DataFrame."""
