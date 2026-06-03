"""
Insight Service
===============

AI-powered analysis, pattern detection, and natural-language summaries.

Future responsibilities:
------------------------
- Analyze DataFrame structure and suggest relevant questions or charts.
- Detect trends, anomalies, correlations, and data quality issues.
- Generate plain-language summaries of key findings for non-technical users.
- Accept natural-language queries and translate them into pandas operations.
- Rank insights by relevance and confidence scores.
- Integrate with external LLM APIs when configured (keys via environment).
"""

import pandas as pd


def generate_insights_placeholder(df: pd.DataFrame) -> list[dict]:
    """
    Return a stub list of insight objects.

    Future responsibilities:
    - Each insight: {title, description, severity, suggested_action, chart_type}.
    - Run statistical tests and heuristic rules over column types and distributions.
    """
    return [
        {
            "title": "Scaffold insight",
            "description": f"Dataset contains {len(df)} rows and {len(df.columns)} columns.",
            "severity": "info",
        }
    ]
