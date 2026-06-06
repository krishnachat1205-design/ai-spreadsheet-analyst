"""
Formula Intelligence Engine — Version 2.1
============================================

Safe arithmetic and function formula evaluation for calculated columns.
Supports nested mathematical, logical, conditional, and text functions.
Excel-compatible behavior for IFS, SUMIF, COUNTIF, AVERAGEIF.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

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
# Supported Functions Registry
# =============================================================================

_SUPPORTED_FUNCTIONS = {
    "SUM", "AVERAGE", "MIN", "MAX", "ROUND", "ABS",
    "IF", "IFS",
    "SUMIF", "COUNTIF", "AVERAGEIF",
    "LEFT", "RIGHT", "MID", "LEN", "UPPER", "LOWER", "CONCAT",
}


# =============================================================================
# Public API
# =============================================================================

def supported_functions() -> dict[str, Any]:
    """Return supported operators and functions for Version 2.1."""
    return {
        "Arithmetic": ["+", "-", "*", "/", "**", "%", "()"],
        "Mathematical": ["SUM()", "AVERAGE()", "MIN()", "MAX()", "ROUND()", "ABS()"],
        "Logical": ["IF()", "IFS()"],
        "Conditional": ["SUMIF()", "COUNTIF()", "AVERAGEIF()"],
        "Text": ["LEFT()", "RIGHT()", "MID()", "LEN()", "UPPER()", "LOWER()", "CONCAT()"],
        "Constants": "Numeric and string literals",
        "References": "Column names (case-sensitive, exact match)",
    }


def extract_column_references(formula: str) -> list[str]:
    """Extract all column name references from a formula string.

    Function names (e.g., SUM, IF) are NOT treated as column references.
    Only names that appear as standalone identifiers (not function calls)
    are considered column references.
    """
    if not formula or not formula.strip():
        return []

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise FormulaServiceError(f"Invalid formula syntax: {exc}")

    def _walk(node, parent_is_call_func=False):
        """Walk AST tree. parent_is_call_func=True when this node is the function name of a Call."""
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Call):
            refs = []
            # The function name itself is NOT a column reference
            for arg in node.args:
                refs.extend(_walk(arg))
            return refs
        if isinstance(node, ast.Name):
            # Only count as column reference if it's not a function name
            if not parent_is_call_func and node.id not in _SUPPORTED_FUNCTIONS:
                return [node.id]
            return []
        if isinstance(node, ast.BinOp):
            return _walk(node.left) + _walk(node.right)
        if isinstance(node, ast.UnaryOp):
            return _walk(node.operand)
        if isinstance(node, ast.Compare):
            refs = _walk(node.left)
            for comp in node.comparators:
                refs.extend(_walk(comp))
            return refs
        if isinstance(node, ast.BoolOp):
            refs = []
            for v in node.values:
                refs.extend(_walk(v))
            return refs
        if isinstance(node, ast.Constant) or (hasattr(ast, "Num") and isinstance(node, ast.Num)):
            return []
        # Fallback for any other nodes
        refs = []
        for child in ast.iter_child_nodes(node):
            refs.extend(_walk(child))
        return refs

    refs = _walk(tree)
    return list(dict.fromkeys(refs))


def validate_formula(formula: str, available_columns: list[str]) -> tuple[bool, str]:
    """
    Validate that a formula is safe and references only valid columns/functions.
    Returns (is_valid, message).
    """
    if not formula or not formula.strip():
        return False, "Formula cannot be empty."

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        return False, f"Invalid formula syntax: {exc}"

    # Whitelist of allowed AST node types for V2.1
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
        ast.Not,
        ast.Call,
        ast.Compare,
        ast.Gt,
        ast.Lt,
        ast.GtE,
        ast.LtE,
        ast.Eq,
        ast.NotEq,
        ast.BoolOp,
        ast.And,
        ast.Or,
    ]
    if hasattr(ast, "Num"):
        allowed_nodes.append(ast.Num)
    if hasattr(ast, "Str"):
        allowed_nodes.append(ast.Str)
    allowed_nodes = tuple(allowed_nodes)

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False, f"Unsupported expression or operator: {type(node).__name__}"

    # Check function names
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SUPPORTED_FUNCTIONS:
                    return False, f"Unknown function: '{node.func.id}'"
            else:
                return False, "Only simple function names are supported (e.g., SUM, IF)"

    # Check column references
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
    Create a new calculated column in the dataframe using a safe formula.
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

    # Normalize to pandas Series
    if not isinstance(result_series, pd.Series):
        result_series = pd.Series(result_series, index=df.index)

    # Handle division by zero: replace infinities with NaN
    result_series = result_series.replace([np.inf, -np.inf], np.nan)

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
        if isinstance(node.value, (int, float, str)):
            return pd.Series([node.value] * length, index=index)
        raise FormulaServiceError(f"Unsupported constant type: {type(node.value).__name__}")

    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # Python < 3.8 compatibility
        return pd.Series([node.n] * length, index=index)

    if hasattr(ast, "Str") and isinstance(node, ast.Str):  # Python < 3.8 compatibility
        return pd.Series([node.s] * length, index=index)

    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        # If it's a supported function name used without calling (shouldn't happen in valid formulas)
        if node.id in _SUPPORTED_FUNCTIONS:
            raise FormulaServiceError(f"Function '{node.id}' must be called with parentheses")
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
        if isinstance(node.op, ast.Not):
            return ~operand.astype(bool)
        raise FormulaServiceError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _eval_ast_node(node.left, context, index, length)
        comparators = [_eval_ast_node(c, context, index, length) for c in node.comparators]

        if len(node.ops) != len(comparators):
            raise FormulaServiceError("Malformed comparison expression")

        results = []
        current_left = left
        for op, right in zip(node.ops, comparators):
            if isinstance(op, ast.Gt):
                results.append(current_left > right)
            elif isinstance(op, ast.Lt):
                results.append(current_left < right)
            elif isinstance(op, ast.GtE):
                results.append(current_left >= right)
            elif isinstance(op, ast.LtE):
                results.append(current_left <= right)
            elif isinstance(op, ast.Eq):
                results.append(current_left == right)
            elif isinstance(op, ast.NotEq):
                results.append(current_left != right)
            else:
                raise FormulaServiceError(f"Unsupported comparison operator: {type(op).__name__}")
            current_left = right

        result = results[0]
        for r in results[1:]:
            result = result & r
        return result

    if isinstance(node, ast.BoolOp):
        values = [_eval_ast_node(v, context, index, length) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = result & v.astype(bool)
            return result
        if isinstance(node.op, ast.Or):
            result = values[0]
            for v in values[1:]:
                result = result | v.astype(bool)
            return result
        raise FormulaServiceError(f"Unsupported boolean operator: {type(node.op).__name__}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FormulaServiceError("Only simple function names are supported")
        func_name = node.func.id
        args = [_eval_ast_node(arg, context, index, length) for arg in node.args]
        return _eval_function(func_name, args, context, index, length)

    raise FormulaServiceError(f"Unsupported expression: {type(node).__name__}")


def _eval_function(
    name: str,
    args: list,
    context: dict[str, pd.Series],
    index: pd.Index,
    length: int,
) -> pd.Series:
    """Evaluate a supported function call."""
    # -------------------------------------------------------------------------
    # Mathematical Functions
    # -------------------------------------------------------------------------
    if name == "SUM":
        if len(args) == 0:
            raise FormulaServiceError("SUM requires at least 1 argument")
        if len(args) == 1:
            return pd.Series([args[0].sum()] * length, index=index)
        result = args[0]
        for arg in args[1:]:
            result = result + arg
        return result

    if name == "AVERAGE":
        if len(args) == 0:
            raise FormulaServiceError("AVERAGE requires at least 1 argument")
        if len(args) == 1:
            return pd.Series([args[0].mean()] * length, index=index)
        result = args[0]
        for arg in args[1:]:
            result = result + arg
        return result / len(args)

    if name == "MIN":
        if len(args) == 0:
            raise FormulaServiceError("MIN requires at least 1 argument")
        if len(args) == 1:
            return pd.Series([args[0].min()] * length, index=index)
        result = args[0]
        for arg in args[1:]:
            result = np.minimum(result, arg)
        return result

    if name == "MAX":
        if len(args) == 0:
            raise FormulaServiceError("MAX requires at least 1 argument")
        if len(args) == 1:
            return pd.Series([args[0].max()] * length, index=index)
        result = args[0]
        for arg in args[1:]:
            result = np.maximum(result, arg)
        return result

    if name == "ROUND":
        if len(args) not in (1, 2):
            raise FormulaServiceError("ROUND requires 1 or 2 arguments: number, [num_digits]")
        number = args[0]
        if len(args) == 1:
            return pd.Series(np.round(number), index=index)
        num_digits = args[1]
        if isinstance(num_digits, pd.Series):
            num_digits = int(num_digits.iloc[0])
        else:
            num_digits = int(num_digits)
        return pd.Series(np.round(number, num_digits), index=index)

    if name == "ABS":
        if len(args) != 1:
            raise FormulaServiceError("ABS requires 1 argument: number")
        return pd.Series(np.abs(args[0]), index=index)

    # -------------------------------------------------------------------------
    # Logical Functions
    # -------------------------------------------------------------------------
    if name == "IF":
        if len(args) != 3:
            raise FormulaServiceError("IF requires exactly 3 arguments: condition, true_value, false_value")
        condition = args[0]
        true_val = args[1]
        false_val = args[2]
        if hasattr(condition, "dtype"):
            condition = condition.fillna(False).astype(bool)
        return pd.Series(np.where(condition, true_val, false_val), index=index)

    if name == "IFS":
        if len(args) < 2 or len(args) % 2 != 0:
            raise FormulaServiceError("IFS requires pairs of condition and value arguments")

        # Excel-style IFS: first matching condition wins
        # Start with all NaN, then fill in matching values
        result = pd.Series([np.nan] * length, index=index)

        # Track which rows have been assigned a value
        assigned = pd.Series([False] * length, index=index)

        for i in range(0, len(args), 2):
            condition = args[i]
            value = args[i + 1]

            if hasattr(condition, "dtype"):
                condition = condition.fillna(False).astype(bool)

            # Only apply to rows that haven't been assigned yet
            applicable = condition & ~assigned

            # Assign value where applicable
            result = pd.Series(np.where(applicable, value, result), index=index)

            # Mark these rows as assigned
            assigned = assigned | applicable

        return result

    # -------------------------------------------------------------------------
    # Conditional Functions
    # -------------------------------------------------------------------------
    if name == "SUMIF":
        if len(args) not in (2, 3):
            raise FormulaServiceError("SUMIF requires 2 or 3 arguments: range, criteria, [sum_range]")
        condition_range = args[0]
        criteria = args[1]
        sum_range = args[2] if len(args) == 3 else condition_range
        mask = _parse_criteria(criteria, condition_range)
        total = sum_range[mask].sum()
        return pd.Series([total] * length, index=index)

    if name == "COUNTIF":
        if len(args) != 2:
            raise FormulaServiceError("COUNTIF requires 2 arguments: range, criteria")
        condition_range = args[0]
        criteria = args[1]
        mask = _parse_criteria(criteria, condition_range)
        count = mask.sum()
        return pd.Series([count] * length, index=index)

    if name == "AVERAGEIF":
        if len(args) not in (2, 3):
            raise FormulaServiceError("AVERAGEIF requires 2 or 3 arguments: range, criteria, [average_range]")
        condition_range = args[0]
        criteria = args[1]
        avg_range = args[2] if len(args) == 3 else condition_range
        mask = _parse_criteria(criteria, condition_range)
        avg = avg_range[mask].mean()
        return pd.Series([avg] * length, index=index)

    # -------------------------------------------------------------------------
    # Text Functions
    # -------------------------------------------------------------------------
    if name == "LEFT":
        if len(args) not in (1, 2):
            raise FormulaServiceError("LEFT requires 1 or 2 arguments: text, [num_chars]")
        text = args[0].astype(str)
        num = args[1] if len(args) == 2 else 1
        if isinstance(num, pd.Series):
            num = int(num.iloc[0])
        else:
            num = int(num)
        return text.str.slice(0, num)

    if name == "RIGHT":
        if len(args) not in (1, 2):
            raise FormulaServiceError("RIGHT requires 1 or 2 arguments: text, [num_chars]")
        text = args[0].astype(str)
        num = args[1] if len(args) == 2 else 1
        if isinstance(num, pd.Series):
            num = int(num.iloc[0])
        else:
            num = int(num)
        return text.str.slice(-num, None)

    if name == "MID":
        if len(args) != 3:
            raise FormulaServiceError("MID requires 3 arguments: text, start_num, num_chars")
        text = args[0].astype(str)
        start = args[1]
        num_chars = args[2]
        if isinstance(start, pd.Series):
            start = int(start.iloc[0])
        else:
            start = int(start)
        if isinstance(num_chars, pd.Series):
            num_chars = int(num_chars.iloc[0])
        else:
            num_chars = int(num_chars)
        start_idx = start - 1  # Excel uses 1-based indexing
        end_idx = start_idx + num_chars
        return text.str.slice(start_idx, end_idx)

    if name == "LEN":
        if len(args) != 1:
            raise FormulaServiceError("LEN requires 1 argument: text")
        return args[0].astype(str).str.len()

    if name == "UPPER":
        if len(args) != 1:
            raise FormulaServiceError("UPPER requires 1 argument: text")
        return args[0].astype(str).str.upper()

    if name == "LOWER":
        if len(args) != 1:
            raise FormulaServiceError("LOWER requires 1 argument: text")
        return args[0].astype(str).str.lower()

    if name == "CONCAT":
        if len(args) < 1:
            raise FormulaServiceError("CONCAT requires at least 1 argument")
        result = args[0].astype(str)
        for arg in args[1:]:
            result = result + arg.astype(str)
        return result

    raise FormulaServiceError(f"Unknown function: '{name}'")


def _parse_criteria(criteria, series: pd.Series) -> pd.Series:
    """Parse a criteria value/string and return a boolean mask.

    Supports:
    - Numeric criteria: 1000, 0.5
    - Text criteria: "Active", "North"  
    - Comparison operators: ">1000", "<=500", ">=0", "!=0", "==Active"
    - Series criteria (equality comparison)
    """
    if isinstance(criteria, pd.Series):
        # If criteria is a Series, use element-wise equality
        return series == criteria

    if isinstance(criteria, str):
        criteria = criteria.strip()

        # Check for comparison operators (sort by length descending to avoid partial matches)
        ops = [">=", "<=", "!=", "==", ">", "<"]
        for op in ops:
            if criteria.startswith(op):
                val_str = criteria[len(op):].strip()
                # Try numeric first, fall back to string
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str

                if op == ">=":
                    return series >= val
                if op == "<=":
                    return series <= val
                if op == ">":
                    return series > val
                if op == "<":
                    return series < val
                if op == "==":
                    return series == val
                if op == "!=":
                    return series != val

        # No operator found, treat as equality (text or numeric)
        # Try numeric first
        try:
            val = float(criteria)
            return series == val
        except ValueError:
            return series == criteria

    # Numeric or other scalar
    return series == criteria