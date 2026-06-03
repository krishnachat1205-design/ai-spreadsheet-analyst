"""
Services Package
================

Orchestration layer that coordinates data operations between the UI and core.
Each service module owns a single domain of responsibility.

Future responsibilities:
------------------------
- Expose a clean public API for app.py and app/ components to consume.
- Coordinate between utils/, core/config, and external libraries (pandas, etc.).
- Remain stateless where possible; accept and return plain data structures.

Planned services:
- file_service       — upload validation, storage, and retrieval
- cleaning_service   — data normalization and quality fixes
- excel_service      — read/write Excel via OpenPyXL and XlsxWriter
- dashboard_service  — Plotly charts and summary metrics
- insight_service    — AI-driven analysis and natural-language summaries
- command_service    — user command parsing and execution pipeline
- audit_service      — action logging and history persistence
"""
