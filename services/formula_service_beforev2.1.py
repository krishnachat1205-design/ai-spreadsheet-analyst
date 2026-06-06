"""
Formula Intelligence Engine — Version 1
======================================

Safe arithmetic formula evaluation for calculated columns.
Expandable architecture for V2+ functions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd


class FormulaServiceError(Exception):
    """Raised when a formula cannot be validated or executed."""


@dataclass
class FormulaResult:
    """Result of a successful calculated column creation."""
    dataframe: pd.DataFrame
    new_column: str
    formula: str
    affected_rows: int
    details: str


# =============================================================================
# Public API
# =============================================================================

def supported_functions() -> dict[str, str]:
    """Return supported operators and syntax for Version 1."""
    return {
        "Arithmetic": "+  -  *  /  **  %  ()",
        "Constants": "Numeric literals (e.g., 0.18)",
        "References": "Column names (case-sensitive, exact match)",
    }


def extract_column_references(formula: str) -> list[str]:
    """Extract all column name references from a formula string."""
    if not formula or not formula.strip():
        return []

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise FormulaServiceError(f"Invalid formula syntax: {exc}")

    refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            refs.append(node.id)
    # Preserve order, remove duplicates
    return list(dict.fromkeys(refs))


def validate_formula(formula: str, available_columns: list[str]) -> tuple[bool, str]:
    """
    Validate that a formula is safe and references only valid columns.
    Returns (is_valid, message).
    """
    if not formula or not formula.strip():
        return False, "Formula cannot be empty."

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        return False, f"Invalid formula syntax: {exc}"

    # Whitelist of allowed AST node types for V1 arithmetic
    allowed_nodes = [
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
    ]
    if hasattr(ast, "Num"):
        allowed_nodes.append(ast.Num)
    allowed_nodes = tuple(allowed_nodes)

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False, f"Unsupported expression or operator: {type(node).__name__}"

    refs = extract_column_references(formula)
    missing = [c for c in refs if c not in available_columns]
    if missing:
        return False, f"Unknown column(s): {', '.join(missing)}"

    return True, "Formula is valid."


def create_calculated_column(
    df: pd.DataFrame,
    new_column: str,
    formula: str,
) -> FormulaResult:
    """
    Create a new calculated column in the dataframe using a safe arithmetic formula.
    """
    if not new_column or not new_column.strip():
        raise FormulaServiceError("New column name cannot be empty.")

    stripped_name = new_column.strip()
    if stripped_name in df.columns:
        raise FormulaServiceError(f"Column '{stripped_name}' already exists.")

    is_valid, msg = validate_formula(formula, df.columns.tolist())
    if not is_valid:
        raise FormulaServiceError(msg)

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise FormulaServiceError(f"Invalid formula syntax: {exc}")

    refs = extract_column_references(formula)
    context = {col: df[col] for col in refs}

    try:
        result_series = _eval_ast_node(tree.body, context, df.index, len(df))
    except Exception as exc:
        raise FormulaServiceError(f"Formula evaluation failed: {exc}")

    # Handle division by zero: replace infinities with NaN
    result_series = result_series.replace([np.inf, -np.inf], np.nan)

    # Ensure pandas Series alignment
    if not isinstance(result_series, pd.Series):
        result_series = pd.Series(result_series, index=df.index)

    new_df = df.copy()
    new_df[stripped_name] = result_series

    affected_rows = int(result_series.notna().sum())

    return FormulaResult(
        dataframe=new_df,
        new_column=stripped_name,
        formula=formula.strip(),
        affected_rows=affected_rows,
        details=f"Created column '{stripped_name}' using formula '{formula.strip()}'.",
    )


# =============================================================================
# Internal Helpers
# =============================================================================

def _eval_ast_node(
    node: ast.AST,
    context: dict[str, pd.Series],
    index: pd.Index,
    length: int,
) -> pd.Series:
    """Recursively evaluate an AST node using pandas Series context."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return pd.Series([node.value] * length, index=index)
        raise FormulaServiceError(f"Unsupported constant type: {type(node.value).__name__}")

    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # Python < 3.8 compatibility
        return pd.Series([node.n] * length, index=index)

    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        raise FormulaServiceError(f"Unknown column reference: {node.id}")

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, context, index, length)
        right = _eval_ast_node(node.right, context, index, length)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right

        raise FormulaServiceError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, context, index, length)
        if isinstance(node.op, ast.USub):
            return -operand
        raise FormulaServiceError(f"Unsupported unary operator: {type(node.op).__name__}")

    raise FormulaServiceError(f"Unsupported expression: {type(node).__name__}")