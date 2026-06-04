"""
KPI Service
===========

Automatically detect and compute business-relevant Key Performance Indicators
from a pandas DataFrame based on heuristic column name matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KPIResult:
    """Structured result for a single KPI."""

    name: str
    value: Any
    formatted_value: str
    description: str
    column_used: Optional[str] = None
    confidence: str = "high"  # high, medium, low


class KPIEngine:
    """
    Detects relevant columns by heuristic naming and computes KPIs.
    Gracefully skips KPIs when required columns are missing or unsuitable.
    """

    _REVENUE_KEYWORDS = (
        "revenue", "sales", "amount", "total", "price", "value", "income", "turnover"
    )
    _ORDER_KEYWORDS = (
        "order", "transaction", "count", "quantity", "qty", "units", "sales"
    )
    _CUSTOMER_KEYWORDS = (
        "customer", "client", "buyer", "user", "consumer", "account"
    )
    _CATEGORY_KEYWORDS = (
        "category", "type", "segment", "group", "class", "department", "genre"
    )
    _PRODUCT_KEYWORDS = (
        "product", "item", "sku", "goods", "merchandise", "service"
    )
    _REGION_KEYWORDS = (
        "region", "country", "state", "city", "area", "location", "territory", "zone", "province"
    )
    _DATE_KEYWORDS = (
        "date", "time", "year", "month", "day", "period", "timestamp"
    )

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.revenue_col: Optional[str] = None
        self.order_col: Optional[str] = None
        self.customer_col: Optional[str] = None
        self.category_col: Optional[str] = None
        self.product_col: Optional[str] = None
        self.region_col: Optional[str] = None
        self.date_col: Optional[str] = None
        self._detect_columns()

    def _detect_columns(self) -> None:
        """Heuristic column detection based on lowercase name matching."""
        actual_cols = list(self.df.columns)
        cols_lower = [str(c).lower() for c in actual_cols]

        def _find(keywords: tuple[str, ...]) -> Optional[str]:
            for idx, col in enumerate(cols_lower):
                if any(kw in col for kw in keywords):
                    return actual_cols[idx]
            return None

        self.revenue_col = _find(self._REVENUE_KEYWORDS)
        self.order_col = _find(self._ORDER_KEYWORDS)
        self.customer_col = _find(self._CUSTOMER_KEYWORDS)
        self.category_col = _find(self._CATEGORY_KEYWORDS)
        self.product_col = _find(self._PRODUCT_KEYWORDS)
        self.region_col = _find(self._REGION_KEYWORDS)
        self.date_col = _find(self._DATE_KEYWORDS)

    def _is_numeric(self, col: Optional[str]) -> bool:
        if col is None or col not in self.df.columns:
            return False
        return pd.api.types.is_numeric_dtype(self.df[col])

    @staticmethod
    def _format_currency(val: float) -> str:
        if pd.isna(val):
            return "—"
        if abs(val) >= 1_000_000:
            return f"${val:,.0f}"
        elif abs(val) >= 1_000:
            return f"${val:,.2f}"
        return f"${val:,.2f}"

    @staticmethod
    def _format_number(val: float) -> str:
        if pd.isna(val):
            return "—"
        if abs(val) >= 1_000_000:
            return f"{val:,.1f}M"
        elif abs(val) >= 1_000:
            return f"{val:,.0f}"
        return f"{val:,.0f}"

    def generate_all(self) -> list[KPIResult]:
        """Compute all detectable KPIs."""
        kpis: list[KPIResult] = []

        # Revenue KPIs
        if self.revenue_col and self._is_numeric(self.revenue_col):
            series = self.df[self.revenue_col].dropna()
            if not series.empty:
                total = float(series.sum())
                avg = float(series.mean())
                max_val = float(series.max())
                min_val = float(series.min())

                kpis.append(
                    KPIResult(
                        name="Total Revenue",
                        value=total,
                        formatted_value=self._format_currency(total),
                        description=f"Sum of '{self.revenue_col}' across all records.",
                        column_used=self.revenue_col,
                        confidence="high",
                    )
                )
                kpis.append(
                    KPIResult(
                        name="Average Revenue",
                        value=avg,
                        formatted_value=self._format_currency(avg),
                        description=f"Mean of '{self.revenue_col}' per record.",
                        column_used=self.revenue_col,
                        confidence="high",
                    )
                )
                kpis.append(
                    KPIResult(
                        name="Maximum Revenue",
                        value=max_val,
                        formatted_value=self._format_currency(max_val),
                        description=f"Largest value in '{self.revenue_col}'.",
                        column_used=self.revenue_col,
                        confidence="high",
                    )
                )
                kpis.append(
                    KPIResult(
                        name="Minimum Revenue",
                        value=min_val,
                        formatted_value=self._format_currency(min_val),
                        description=f"Smallest value in '{self.revenue_col}'.",
                        column_used=self.revenue_col,
                        confidence="high",
                    )
                )

        # Total Orders
        if self.order_col and self._is_numeric(self.order_col):
            total = float(self.df[self.order_col].sum())
            kpis.append(
                KPIResult(
                    name="Total Orders",
                    value=total,
                    formatted_value=self._format_number(total),
                    description=f"Sum of '{self.order_col}'.",
                    column_used=self.order_col,
                    confidence="high",
                )
            )
        else:
            # Proxy: row count
            kpis.append(
                KPIResult(
                    name="Total Orders",
                    value=len(self.df),
                    formatted_value=self._format_number(len(self.df)),
                    description="Row count used as proxy (no explicit order column detected).",
                    column_used=None,
                    confidence="medium",
                )
            )

        # Total Customers
        if self.customer_col:
            unique_count = int(self.df[self.customer_col].nunique())
            kpis.append(
                KPIResult(
                    name="Total Customers",
                    value=unique_count,
                    formatted_value=self._format_number(unique_count),
                    description=f"Unique values in '{self.customer_col}'.",
                    column_used=self.customer_col,
                    confidence="high",
                )
            )

        # Top Category
        if self.category_col:
            mode_result = self.df[self.category_col].mode()
            if not mode_result.empty:
                top_val = str(mode_result.iloc[0])
                count = int((self.df[self.category_col] == top_val).sum())
                pct = (count / len(self.df)) * 100
                kpis.append(
                    KPIResult(
                        name="Top Category",
                        value=top_val,
                        formatted_value=f"{top_val} ({pct:.1f}%)",
                        description=f"Most frequent value in '{self.category_col}' ({count:,} occurrences).",
                        column_used=self.category_col,
                        confidence="high",
                    )
                )

        # Top Product
        if self.product_col:
            mode_result = self.df[self.product_col].mode()
            if not mode_result.empty:
                top_val = str(mode_result.iloc[0])
                count = int((self.df[self.product_col] == top_val).sum())
                pct = (count / len(self.df)) * 100
                kpis.append(
                    KPIResult(
                        name="Top Product",
                        value=top_val,
                        formatted_value=f"{top_val} ({pct:.1f}%)",
                        description=f"Most frequent value in '{self.product_col}' ({count:,} occurrences).",
                        column_used=self.product_col,
                        confidence="high",
                    )
                )

        # Top Region
        if self.region_col:
            mode_result = self.df[self.region_col].mode()
            if not mode_result.empty:
                top_val = str(mode_result.iloc[0])
                count = int((self.df[self.region_col] == top_val).sum())
                pct = (count / len(self.df)) * 100
                kpis.append(
                    KPIResult(
                        name="Top Region",
                        value=top_val,
                        formatted_value=f"{top_val} ({pct:.1f}%)",
                        description=f"Most frequent value in '{self.region_col}' ({count:,} occurrences).",
                        column_used=self.region_col,
                        confidence="high",
                    )
                )

        # Dataset Time Span
        if self.date_col:
            try:
                date_series = pd.to_datetime(self.df[self.date_col], errors="coerce").dropna()
                if not date_series.empty and len(date_series) > 1:
                    days_span = int((date_series.max() - date_series.min()).days)
                    kpis.append(
                        KPIResult(
                            name="Dataset Time Span",
                            value=days_span,
                            formatted_value=f"{days_span:,} days",
                            description=f"Date range from '{self.date_col}' ({date_series.min().date()} to {date_series.max().date()}).",
                            column_used=self.date_col,
                            confidence="high",
                        )
                    )
            except Exception:
                logger.debug("Could not compute time span KPI.")

        return kpis


def generate_kpis(df: pd.DataFrame) -> list[KPIResult]:
    """Convenience function: instantiate engine and return all KPIs."""
    engine = KPIEngine(df)
    return engine.generate_all()