"""
Core Package
============

Contains foundational configuration, domain models, and shared abstractions
used across the application. This layer has no UI dependencies.

Future responsibilities:
- Export config, base classes, and domain exceptions.
- Define data schemas and validation rules for spreadsheet entities.
- Provide interfaces that services implement.
"""

from core.config import APP_TITLE, APP_VERSION

__all__ = ["APP_TITLE", "APP_VERSION"]
