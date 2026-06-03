"""
Command Service
===============

Parse and execute user commands against spreadsheet data.

Future responsibilities:
------------------------
- Accept structured commands (filter, sort, group, aggregate, pivot) from UI or NL input.
- Validate command syntax and parameters before execution.
- Translate commands into safe pandas operations (no arbitrary code execution).
- Maintain a command history stack for undo/redo support.
- Return execution results with preview DataFrames and status messages.
- Log every command via audit_service for traceability.
"""

import pandas as pd


def execute_command_placeholder(
    df: pd.DataFrame,
    command: str,
) -> pd.DataFrame:
    """
    Stub command executor that returns the input DataFrame unchanged.

    Future responsibilities:
    - Parse `command` into an operation tree.
    - Apply transformations and return modified DataFrame + metadata.
    """
    return df
