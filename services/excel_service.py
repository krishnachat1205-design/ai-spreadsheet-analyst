"""
Excel Service
=============

Read and write Excel workbooks using OpenPyXL and XlsxWriter.

Future responsibilities:
------------------------
- Read .xlsx/.xls files into pandas DataFrames with sheet selection support.
- List sheet names and preview row counts without loading full workbooks.
- Write cleaned or analyzed data back to .xlsx with formatting (XlsxWriter).
- Apply cell styles: headers, number formats, conditional formatting.
- Support multi-sheet exports (raw data, summary, charts metadata).
- Handle large files with chunked reads and memory-efficient strategies.
"""

from pathlib import Path

import pandas as pd

from core.config import DEFAULT_SHEET_INDEX


def read_excel_placeholder(file_path: Path) -> pd.DataFrame:
    """
    Stub for reading an Excel file into a DataFrame.

    Future responsibilities:
    - Use pandas + openpyxl engine with configurable sheet index/name.
    - Surface parse errors with actionable messages for the UI.
    - Respect DEFAULT_SHEET_INDEX from config when no sheet is specified.
    """
    raise NotImplementedError(
        f"Excel reading not yet implemented for: {file_path} "
        f"(default sheet index: {DEFAULT_SHEET_INDEX})"
    )
