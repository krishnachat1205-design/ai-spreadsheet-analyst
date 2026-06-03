"""
Audit Service
=============

Persistent logging of user actions, cleaning operations, and export events.
History is stored in history/actions.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import HISTORY_DIR
from utils.helpers import ensure_directory

logger = logging.getLogger(__name__)

ACTIONS_FILENAME = "actions.json"


def get_history_directory() -> Path:
    """Return the configured history directory, creating it if needed."""
    return ensure_directory(HISTORY_DIR)


def _actions_file_path() -> Path:
    """Return the full path to the JSON audit log file."""
    return get_history_directory() / ACTIONS_FILENAME


def _empty_history_file() -> None:
    """Initialize an empty JSON array in the actions file."""
    path = _actions_file_path()
    path.write_text("[]", encoding="utf-8")


def save_history_to_json(history: list[dict[str, Any]]) -> None:
    """
    Persist the full action history list to history/actions.json.

    Args:
        history: List of audit entry dictionaries.
    """
    path = _actions_file_path()
    get_history_directory()

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False)
        logger.debug("Saved %s audit entries to %s", len(history), path)
    except OSError as exc:
        logger.error("Failed to save audit history: %s", exc)
        raise


def get_action_history() -> list[dict[str, Any]]:
    """
    Load and return all audit entries from history/actions.json.

    Returns:
        Chronological list of action dictionaries. Empty list when no file exists.
    """
    path = _actions_file_path()
    if not path.exists():
        _empty_history_file()
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            content = file.read().strip()
            if not content:
                return []
            history = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("Corrupt audit log at %s; resetting. Error: %s", path, exc)
        _empty_history_file()
        return []
    except OSError as exc:
        logger.error("Failed to read audit history: %s", exc)
        return []

    if not isinstance(history, list):
        logger.warning("Audit log format invalid; resetting to empty list.")
        _empty_history_file()
        return []

    return history


def clear_history() -> None:
    """Remove all entries from the audit log and rewrite the JSON file."""
    save_history_to_json([])
    logger.info("Audit history cleared.")


def log_action(
    action: str,
    details: str,
    affected_rows: int = 0,
) -> dict[str, Any]:
    """
    Append a new audit entry and persist it to history/actions.json.

    Args:
        action: Short label describing the operation (e.g. 'Remove Duplicates').
        details: Human-readable description of what changed.
        affected_rows: Count of rows or cells affected by the operation.

    Returns:
        The newly created audit entry dictionary.
    """
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "details": details,
        "affected_rows": int(affected_rows),
    }

    history = get_action_history()
    history.append(entry)

    try:
        save_history_to_json(history)
    except OSError:
        # Keep in-memory history available even if disk write fails.
        logger.exception("Audit entry created but could not be saved to disk.")

    logger.info("Audit: [%s] %s (affected=%s)", action, details, affected_rows)
    return entry
