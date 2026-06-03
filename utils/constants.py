"""
Application Constants
=====================

Module-level constants that do not depend on environment or filesystem paths.
For path and runtime configuration, prefer core/config.py.

Future responsibilities:
------------------------
- Define command vocabulary (allowed verbs for command_service).
- Chart type enums and default color palettes for dashboard_service.
- Insight severity levels and category labels.
- User-facing message templates and error codes.
- API response status strings shared across services.
"""

# ---------------------------------------------------------------------------
# Application identity (mirrors core/config for convenience in utils layer)
# ---------------------------------------------------------------------------

APP_NAME: str = "AI Spreadsheet Analyst"

# ---------------------------------------------------------------------------
# Command vocabulary (for command_service)
# ---------------------------------------------------------------------------

COMMAND_VERBS: tuple[str, ...] = (
    "filter",
    "sort",
    "group",
    "aggregate",
    "pivot",
    "select",
    "rename",
    "drop",
)

# ---------------------------------------------------------------------------
# Insight severity levels (for insight_service)
# ---------------------------------------------------------------------------

SEVERITY_INFO: str = "info"
SEVERITY_WARNING: str = "warning"
SEVERITY_CRITICAL: str = "critical"

SEVERITY_LEVELS: tuple[str, ...] = (SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_CRITICAL)

# ---------------------------------------------------------------------------
# Chart types (for dashboard_service)
# ---------------------------------------------------------------------------

CHART_TYPES: tuple[str, ...] = (
    "bar",
    "line",
    "scatter",
    "histogram",
    "pie",
    "heatmap",
)
