"""
AI Business Analyst Copilot Service
===================================

Gemini-powered intelligent assistant for spreadsheet analysis.
Uses deterministic context builders to summarize dataset state
and sends structured prompts to Gemini 2.5 Flash.

Never transmits raw dataframe rows — only metadata, KPIs,
insights, and summaries. Safe for 100k+ row datasets.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CopilotServiceError(Exception):
    """Raised when the copilot service encounters an error."""
    pass


# ---------------------------------------------------------------------------
# Response Model
# ---------------------------------------------------------------------------

@dataclass
class CopilotResponse:
    """Structured response from the AI copilot."""
    answer: str
    confidence: str = "medium"  # high, medium, low
    suggested_actions: list[str] = field(default_factory=list)
    referenced_charts: list[str] = field(default_factory=list)
    referenced_kpis: list[str] = field(default_factory=list)
    referenced_insights: list[str] = field(default_factory=list)
    tokens_used: Optional[int] = None


# ---------------------------------------------------------------------------
# Gemini Client Setup
# ---------------------------------------------------------------------------

def _get_gemini_client():
    """Initialize and return the google.generativeai module."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise CopilotServiceError(
            "GEMINI_API_KEY environment variable not set. "
            "Please configure your API key to use the AI Copilot."
        )

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise CopilotServiceError(
            "google-generativeai package is not installed. "
            "Install it with: pip install google-generativeai"
        ) from exc

    try:
        genai.configure(api_key=api_key)
        return genai
    except Exception as exc:
        raise CopilotServiceError(f"Failed to initialize Gemini client: {exc}") from exc


def _get_model():
    """Return the configured Gemini generative model."""
    genai = _get_gemini_client()
    return genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return """You are a Senior Business Analyst, Senior Data Analyst, and Excel Consultant.
You are assisting a user who is analyzing spreadsheet data using an AI-powered analytics platform.

Your role:
- Provide practical, actionable business advice grounded in the data provided
- Explain your reasoning clearly and cite specific numbers/KPIs when available
- Suggest next actions the user should take inside the platform
- Recommend relevant charts, pivot tables, and formulas when appropriate
- Be concise but thorough; use professional business language
- Format responses with clear Markdown headings and bullet points

Rules:
1. Ground every answer in the provided dataset context.
2. If data is missing or incomplete, state that explicitly.
3. Prioritize critical insights and high-priority recommendations.
4. Suggest specific platform features (cleaning, formulas, pivots, charts) when relevant.
5. If you do not know something, say so rather than hallucinating.
6. NEVER ask the user to upload data or perform actions outside the platform.
7. For formula suggestions, use only arithmetic operators: +, -, *, /, **, %, () and exact column names."""


# ---------------------------------------------------------------------------
# Context Builders
# ---------------------------------------------------------------------------

def build_dataset_context(
    df: Optional[pd.DataFrame] = None,
    metadata: Optional[dict] = None,
    health: Optional[dict] = None,
    quality_report: Optional[dict] = None,
) -> str:
    """Build a concise dataset context string."""
    if df is None or df.empty:
        return "No dataset is currently loaded."

    lines = ["## Dataset Overview"]

    if metadata:
        lines.append(f"- File: {metadata.get('original_filename', 'Unknown')}")
        lines.append(f"- Rows: {metadata.get('row_count', len(df)):,}")
        lines.append(f"- Columns: {metadata.get('column_count', len(df.columns))}")
    else:
        lines.append(f"- Rows: {len(df):,}")
        lines.append(f"- Columns: {len(df.columns)}")

    cols = list(df.columns)
    if len(cols) > 50:
        cols_display = cols[:50] + [f"... and {len(cols) - 50} more"]
    else:
        cols_display = cols
    lines.append(f"- Column Names: {', '.join(str(c) for c in cols_display)}")

    dtype_counts = df.dtypes.value_counts().to_dict()
    lines.append(f"- Data Types: {dict(dtype_counts)}")

    if health:
        missing = health.get("missing_values", {})
        lines.append(f"- Total Missing Values: {missing.get('total', 0):,}")
        dupes = health.get("duplicate_rows", 0)
        lines.append(f"- Duplicate Rows: {dupes:,}")

    if quality_report:
        score = quality_report.get("completeness_score", "N/A")
        lines.append(f"- Completeness Score: {score}/100")

    return "\n".join(lines)


def build_kpi_context(kpis: Optional[list] = None) -> str:
    """Build KPI context string."""
    if not kpis:
        return "No KPIs have been generated yet."

    lines = ["## Key Performance Indicators"]
    for kpi in kpis[:10]:
        lines.append(
            f"- {kpi.name}: {kpi.formatted_value} (Confidence: {kpi.confidence})"
        )
        if hasattr(kpi, "description") and kpi.description:
            lines.append(f"  *{kpi.description}*")
    return "\n".join(lines)


def build_insight_context(
    insights: Optional[list] = None,
    recommendations: Optional[list] = None,
) -> str:
    """Build insights and recommendations context."""
    lines = []

    if insights:
        lines.append("## Business Insights")
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_insights = sorted(
            insights, key=lambda x: severity_order.get(x.severity, 99)
        )
        for ins in sorted_insights[:10]:
            lines.append(f"- [{ins.severity.upper()}] {ins.observation}")
            lines.append(f"  Evidence: {ins.evidence}")
            lines.append(f"  Interpretation: {ins.business_interpretation}")

    if recommendations:
        lines.append("\n## Recommendations")
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_recs = sorted(
            recommendations, key=lambda r: priority_order.get(r.priority, 99)
        )
        for rec in sorted_recs[:10]:
            lines.append(f"- [{rec.priority.upper()}] {rec.title}")
            lines.append(f"  {rec.description}")
            lines.append(f"  Evidence: {rec.evidence}")

    if not lines:
        return "No insights or recommendations have been generated yet."

    return "\n".join(lines)


def build_report_context(report: Optional[Any] = None) -> str:
    """Build analyst report context."""
    if report is None:
        return "No analyst report has been generated yet."

    lines = ["## Analyst Report Summary"]

    if hasattr(report, "dataset_overview"):
        ov = report.dataset_overview
        lines.append(f"- Rows: {ov.get('row_count', 'N/A'):,}")
        lines.append(f"- Columns: {ov.get('column_count', 'N/A')}")

    if hasattr(report, "data_quality_summary"):
        q = report.data_quality_summary
        lines.append(f"- Completeness Score: {q.get('completeness_score', 'N/A')}/100")

    if hasattr(report, "analyst_notes"):
        lines.append(f"\nAnalyst Notes: {report.analyst_notes}")

    return "\n".join(lines)


def build_dashboard_context(
    charts: Optional[dict] = None,
    recommendations: Optional[list] = None,
) -> str:
    """Build dashboard context."""
    lines = []

    if recommendations:
        lines.append("## Chart Recommendations")
        for rec in recommendations:
            chart_type = rec.get("chart_type", "Unknown").title()
            reason = rec.get("reason", "")
            cols = ", ".join(str(c) for c in rec.get("columns", []))
            lines.append(f"- {chart_type}: {reason} (Columns: {cols})")

    if charts:
        lines.append("\n## Generated Charts")
        for category, chart_list in charts.items():
            if chart_list and any(getattr(c, "applicable", True) for c in chart_list):
                lines.append(f"- {category.replace('_', ' ').title()}: {len(chart_list)} chart(s)")

    if not lines:
        return "No dashboard analytics have been generated yet."

    return "\n".join(lines)


def build_pivot_context(pivot_result: Optional[Any] = None) -> str:
    """Build pivot table context."""
    if pivot_result is None:
        return "No pivot table has been generated yet."

    lines = ["## Pivot Table Results"]
    lines.append(f"- Shape: {pivot_result.shape[0]:,} rows × {pivot_result.shape[1]} columns")
    lines.append(f"- Row Fields: {', '.join(pivot_result.rows)}")
    lines.append(f"- Value Fields: {', '.join(pivot_result.values)}")
    lines.append(f"- Aggregation: {pivot_result.agg_function}")
    if pivot_result.columns:
        lines.append(f"- Column Fields: {', '.join(pivot_result.columns)}")
    return "\n".join(lines)


def build_comparison_context(result: Optional[Any] = None) -> str:
    """Build comparison context."""
    if result is None:
        return "No multi-file comparison has been performed yet."

    lines = ["## Multi-File Comparison Results"]
    lines.append(f"- Dataset A Rows: {result.dataset_a_rows:,}")
    lines.append(f"- Dataset B Rows: {result.dataset_b_rows:,}")
    lines.append(f"- Matched Records: {result.matched_records:,}")
    lines.append(f"- Added Records: {result.added_count:,}")
    lines.append(f"- Removed Records: {result.removed_count:,}")
    lines.append(f"- Modified Records: {result.modified_count:,}")

    return "\n".join(lines)


def build_formula_context(history: Optional[list] = None) -> str:
    """Build formula context."""
    if not history:
        return "No calculated columns have been created yet."

    lines = ["## Formula History"]
    for entry in history[-5:]:
        col_name = entry.get("column_name", "Unknown")
        formula = entry.get("formula", "Unknown")
        affected = entry.get("affected_rows", 0)
        lines.append(f"- {col_name} = {formula} ({affected:,} rows)")
    return "\n".join(lines)


def build_full_context(
    df: Optional[pd.DataFrame] = None,
    metadata: Optional[dict] = None,
    health: Optional[dict] = None,
    quality_report: Optional[dict] = None,
    kpis: Optional[list] = None,
    insights: Optional[list] = None,
    recommendations: Optional[list] = None,
    report: Optional[Any] = None,
    charts: Optional[dict] = None,
    chart_recommendations: Optional[list] = None,
    pivot_result: Optional[Any] = None,
    comparison_result: Optional[Any] = None,
    formula_history: Optional[list] = None,
) -> str:
    """Build complete context for the copilot."""
    sections = [
        build_dataset_context(df, metadata, health, quality_report),
        build_kpi_context(kpis),
        build_insight_context(insights, recommendations),
        build_report_context(report),
        build_dashboard_context(charts, chart_recommendations),
        build_pivot_context(pivot_result),
        build_comparison_context(comparison_result),
        build_formula_context(formula_history),
    ]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Core AI Functions
# ---------------------------------------------------------------------------

def ask_copilot(
    question: str,
    full_context: str = "",
    history: Optional[list] = None,
) -> CopilotResponse:
    """
    Send a user question to Gemini and return a structured response.

    Args:
        question: The user's question.
        full_context: Complete dataset context built by build_full_context().
        history: Previous conversation turns (list of dicts with 'question', 'answer').

    Returns:
        CopilotResponse with the AI answer and metadata.
    """
    if not question or not question.strip():
        raise CopilotServiceError("Question cannot be empty.")

    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n"
        for turn in history[-5:]:
            history_text += f"User: {turn.get('question', '')}\n"
            history_text += f"Assistant: {turn.get('answer', '')}\n\n"

    prompt = f"""{history_text}

=== CURRENT DATASET CONTEXT ===
{full_context}

=== USER QUESTION ===
{question}

Please provide a structured response with:
1. A direct answer to the question
2. Your reasoning
3. Specific references to KPIs/insights/data points used
4. Suggested next actions (if any)
5. Relevant chart/pivot/formula suggestions (if any)
"""

    try:
        model = _get_model()
        response = model.generate_content(
            contents=[{"role": "user", "parts": [f"{_build_system_prompt()}\n\n{prompt}"]}],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
            },
        )

        if not response:
            raise CopilotServiceError("Gemini returned an empty response.")

        try:
            answer = response.text
        except ValueError:
            raise CopilotServiceError(
                "The AI response was blocked by safety filters. Please rephrase your question."
            )

        if not answer or not answer.strip():
            raise CopilotServiceError("Gemini returned an empty response.")

        # Heuristic extraction of suggested actions
        suggested_actions = []
        for line in answer.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ")) and any(
                kw in line.lower()
                for kw in ["suggest", "recommend", "try", "consider", "use", "create", "build"]
            ):
                suggested_actions.append(line[2:])

        return CopilotResponse(
            answer=answer,
            confidence="high",
            suggested_actions=suggested_actions[:5],
        )

    except CopilotServiceError:
        raise
    except Exception as exc:
        raise CopilotServiceError(f"Gemini API error: {exc}") from exc


def generate_executive_summary(full_context: str = "") -> CopilotResponse:
    """
    Generate an executive summary from all available data.
    """
    prompt = f"""Based on the following dataset analysis, generate a comprehensive Executive Summary.

=== DATASET CONTEXT ===
{full_context}

=== REQUIRED SECTIONS ===
1. **Executive Overview** — One-paragraph summary of what this dataset contains and its business significance
2. **Key Findings** — 3-5 bullet points of the most important discoveries
3. **Business Risks** — Critical risks identified from the data (include severity)
4. **Opportunities** — Specific growth or efficiency opportunities
5. **Recommendations** — Prioritized actionable recommendations
6. **Next Steps** — Concrete next actions for the user to take in the platform

Format with clear Markdown headings and bullet points. Be specific and reference actual numbers/KPIs where possible.
"""

    try:
        model = _get_model()
        response = model.generate_content(
            contents=[{"role": "user", "parts": [f"{_build_system_prompt()}\n\n{prompt}"]}],
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 4096,
            },
        )

        if not response:
            raise CopilotServiceError("Gemini returned an empty response.")

        try:
            answer = response.text
        except ValueError:
            raise CopilotServiceError(
                "The AI response was blocked by safety filters. Please try again."
            )

        if not answer or not answer.strip():
            raise CopilotServiceError("Gemini returned an empty response.")

        return CopilotResponse(answer=answer, confidence="high")

    except CopilotServiceError:
        raise
    except Exception as exc:
        raise CopilotServiceError(f"Gemini API error during executive summary: {exc}") from exc


def generate_analysis_questions(full_context: str = "") -> dict:
    """
    Generate suggested analysis questions, charts, pivots, and formulas.
    Returns a dictionary with categorized suggestions.
    """
    prompt = f"""Based on the following dataset context, generate specific suggestions for deeper analysis.

=== DATASET CONTEXT ===
{full_context}

Generate suggestions in these exact categories:

## Suggested Questions
List 5 specific business questions the user should ask about this data.

## Suggested Charts
List 3-5 chart types with the specific columns/fields to use.

## Suggested Pivot Tables
List 2-3 pivot table configurations with row fields, value fields, and aggregation functions.

## Suggested Formulas
List 2-3 calculated column formulas using existing column names. Use arithmetic operators: +, -, *, /, **, %, ().

Format each section with clear Markdown bullet points. Be specific about column names.
"""

    try:
        model = _get_model()
        response = model.generate_content(
            contents=[{"role": "user", "parts": [f"{_build_system_prompt()}\n\n{prompt}"]}],
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 2048,
            },
        )

        if not response:
            raise CopilotServiceError("Gemini returned an empty response.")

        try:
            text = response.text
        except ValueError:
            raise CopilotServiceError(
                "The AI response was blocked by safety filters. Please try again."
            )

        if not text or not text.strip():
            raise CopilotServiceError("Gemini returned an empty response.")

        # Parse sections
        suggestions = {
            "questions": [],
            "charts": [],
            "pivots": [],
            "formulas": [],
        }

        current_section = None
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if line.startswith("##"):
                if "question" in lower:
                    current_section = "questions"
                elif "chart" in lower:
                    current_section = "charts"
                elif "pivot" in lower:
                    current_section = "pivots"
                elif "formula" in lower:
                    current_section = "formulas"
                continue

            if current_section and (line.startswith("- ") or line.startswith("* ")):
                suggestions[current_section].append(line[2:])

        # Fallback: if nothing parsed, return raw text in questions
        if not any(suggestions.values()):
            suggestions["questions"] = [text]

        return suggestions

    except CopilotServiceError:
        raise
    except Exception as exc:
        raise CopilotServiceError(f"Gemini API error during suggestion generation: {exc}") from exc