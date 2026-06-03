"""
Dashboard Service
=================

Build interactive visualizations and summary metrics with Plotly.

Future responsibilities:
------------------------
- Compute aggregate statistics (sum, mean, count, min, max) per numeric column.
- Generate Plotly figures: bar, line, scatter, histogram, pie, heatmap.
- Apply consistent theming and color palettes from config/assets.
- Return figure objects ready for st.plotly_chart() in the UI layer.
- Support drill-down filters driven by user selections or commands.
- Cache expensive chart computations when the underlying data is unchanged.
"""

import pandas as pd


def get_summary_metrics_placeholder(df: pd.DataFrame) -> dict:
    """
    Return a stub summary metrics dictionary.

    Future responsibilities:
    - Compute describe() stats, null percentages, and unique value counts.
    - Format numbers for display (currency, percentages, compact notation).
    """
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
    }
