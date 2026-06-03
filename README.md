# AI Spreadsheet Analyst

An intelligent web application for uploading, cleaning, analyzing, and exporting spreadsheet data — powered by **Python**, **Streamlit**, **Pandas**, and **Plotly**.

---

## Project Goals

The AI Spreadsheet Analyst helps users work with Excel and CSV files without writing code. It aims to:

1. **Upload and validate** spreadsheets safely with size and format checks.
2. **Clean and normalize** messy data (missing values, inconsistent types, duplicates).
3. **Visualize** key metrics and trends through interactive Plotly dashboards.
4. **Generate insights** using statistical analysis and (future) AI-driven summaries.
5. **Execute commands** — filter, sort, group, pivot — via a safe, auditable pipeline.
6. **Export** cleaned data and reports back to Excel with professional formatting.
7. **Audit** every action for traceability and undo support.

---

## Tech Stack

| Layer        | Technology                          |
| ------------ | ----------------------------------- |
| UI           | Streamlit                           |
| Data         | Pandas                              |
| Excel I/O    | OpenPyXL (read), XlsxWriter (write) |
| Charts       | Plotly                              |
| Language     | Python 3.10+                        |

---

## Architecture

The project follows a **modular, layered architecture** that separates concerns:

```
┌─────────────────────────────────────────────────────────┐
│                      app.py                             │
│              (Streamlit entry point)                    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                     app/                                │
│           (UI components & pages)                       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   services/                             │
│  file · cleaning · excel · dashboard · insight          │
│  command · audit                                        │
└──────────┬─────────────────────────────┬────────────────┘
           │                             │
┌──────────▼──────────┐     ┌──────────▼──────────┐
│       core/         │     │       utils/          │
│  config · models    │     │  helpers · constants  │
└─────────────────────┘     └───────────────────────┘
```

### Layer Responsibilities

| Directory   | Role                                                                 |
| ----------- | -------------------------------------------------------------------- |
| `app.py`    | Streamlit entry point; wires UI to services                          |
| `app/`      | Streamlit pages, widgets, and layout components                      |
| `core/`     | Configuration, domain models, shared abstractions                    |
| `services/` | Business orchestration — one module per domain                       |
| `utils/`    | Pure helper functions and application-wide constants                 |
| `uploads/`  | Temporary storage for user-uploaded files                            |
| `exports/`  | Generated Excel reports and download artifacts                       |
| `history/`  | Audit logs and command history                                       |
| `assets/`   | Static files (CSS, images, icons)                                    |
| `tests/`    | Unit and integration tests                                           |
| `docs/`     | Extended documentation and design notes                              |

### Service Modules

| Module               | Future Responsibility                                      |
| -------------------- | ---------------------------------------------------------- |
| `file_service`       | Upload validation, storage, and DataFrame loading          |
| `cleaning_service`   | Data normalization, dtype coercion, quality reports        |
| `excel_service`      | Excel read/write with formatting via OpenPyXL/XlsxWriter   |
| `dashboard_service`  | Plotly charts and aggregate summary metrics                |
| `insight_service`    | AI-powered pattern detection and natural-language summaries|
| `command_service`    | Safe command parsing and pandas operation execution        |
| `audit_service`      | Persistent action logging and history retrieval            |

---

## Project Structure

```
ai-spreadsheet-analyst/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore
│
├── app/                        # Streamlit UI layer
│   └── __init__.py
│
├── core/                       # Configuration & domain core
│   ├── __init__.py
│   └── config.py               # Paths, limits, defaults
│
├── services/                   # Business orchestration
│   ├── __init__.py
│   ├── file_service.py
│   ├── cleaning_service.py
│   ├── excel_service.py
│   ├── dashboard_service.py
│   ├── insight_service.py
│   ├── command_service.py
│   └── audit_service.py
│
├── utils/                      # Shared helpers
│   ├── __init__.py
│   ├── helpers.py
│   └── constants.py
│
├── uploads/                    # User-uploaded files (gitignored)
├── exports/                    # Generated exports (gitignored)
├── history/                    # Audit trail (gitignored)
├── assets/                     # Static assets
├── tests/                      # Test suite
└── docs/                       # Documentation
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone or navigate to the project directory
cd "project 4 (ai spreadsheet analyst)"

# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Configuration

All runtime settings live in `core/config.py`:

- **Paths** — `UPLOADS_DIR`, `EXPORTS_DIR`, `HISTORY_DIR`, `ASSETS_DIR`
- **File limits** — `ALLOWED_EXTENSIONS`, `MAX_UPLOAD_SIZE_MB`
- **Defaults** — sheet index, encoding, null placeholders, export format

Environment-specific overrides (API keys, secrets) will be loaded from `.env` in a future iteration.

---

## Development Status

This repository currently contains **scaffold and boilerplate only**. Business logic is not yet implemented. Each module includes detailed docstrings describing its future responsibilities.

### Next Steps

1. Implement `file_service` upload and validation pipeline.
2. Build `cleaning_service` automatic data quality fixes.
3. Wire `dashboard_service` Plotly charts into Streamlit pages.
4. Add `insight_service` statistical and AI-driven analysis.
5. Enable `command_service` safe pandas operations from the UI.
6. Persist audit entries via `audit_service` to `history/`.
7. Write unit tests in `tests/`.

---

## License

TBD
