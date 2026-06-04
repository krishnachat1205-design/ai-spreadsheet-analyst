"""
Insight Service
===============

Deterministic business insight and recommendation generation.
No LLMs — pure statistical and heuristic rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from services.kpi_service import KPIEngine

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """Structured business insight."""

    observation: str
    evidence: str
    business_interpretation: str
    severity: str  # info, warning, critical
    category: str  # data_quality, business, trend


@dataclass
class Recommendation:
    """Evidence-based recommendation."""

    title: str
    description: str
    evidence: str
    priority: str  # high, medium, low
    action_category: str


class InsightEngine:
    """
    Generates deterministic insights by analyzing column distributions,
    data quality, and business patterns.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.kpi_engine = KPIEngine(df)
        self.insights: list[Insight] = []
        self.recommendations: list[Recommendation] = []

    def generate_insights(self) -> list[Insight]:
        """Run all insight detectors."""
        self.insights = []
        self._detect_missing_data_insight()
        self._detect_duplicate_insight()
        self._detect_top_category_insight()
        self._detect_top_region_insight()
        self._detect_revenue_concentration_insight()
        self._detect_growth_trend_insight()
        self._detect_customer_concentration_insight()
        self._detect_low_performing_region_insight()
        return self.insights

    def generate_recommendations(self) -> list[Recommendation]:
        """Generate recommendations based on current insights and data state."""
        self.recommendations = []
        self._recommend_data_quality()
        self._recommend_top_category_focus()
        self._recommend_region_investigation()
        self._recommend_customer_diversification()
        self._recommend_growth_analysis()
        return self.recommendations

    # ------------------------------------------------------------------
    # Insight Detectors
    # ------------------------------------------------------------------

    def _detect_missing_data_insight(self) -> None:
        total_missing = int(self.df.isna().sum().sum())
        total_cells = self.df.shape[0] * self.df.shape[1]
        pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

        if total_missing == 0:
            self.insights.append(
                Insight(
                    observation="Dataset has no missing values.",
                    evidence=f"0 missing cells out of {total_cells:,} total cells.",
                    business_interpretation="Data completeness is excellent. Analysis can proceed with high confidence.",
                    severity="info",
                    category="data_quality",
                )
            )
        elif pct > 20:
            self.insights.append(
                Insight(
                    observation="Critical missing data detected.",
                    evidence=f"{total_missing:,} missing cells ({pct:.1f}% of dataset).",
                    business_interpretation="High missingness may introduce bias. Imputation or collection process review is urgently needed.",
                    severity="critical",
                    category="data_quality",
                )
            )
        elif pct > 5:
            self.insights.append(
                Insight(
                    observation="Moderate missing data detected.",
                    evidence=f"{total_missing:,} missing cells ({pct:.1f}% of dataset).",
                    business_interpretation="Some columns may be under-reported. Consider targeted cleaning or source verification.",
                    severity="warning",
                    category="data_quality",
                )
            )
        else:
            self.insights.append(
                Insight(
                    observation="Minor missing data detected.",
                    evidence=f"{total_missing:,} missing cells ({pct:.1f}% of dataset).",
                    business_interpretation="Impact is likely minimal. Standard imputation strategies should suffice.",
                    severity="info",
                    category="data_quality",
                )
            )

    def _detect_duplicate_insight(self) -> None:
        dupes = int(self.df.duplicated().sum())
        if dupes == 0:
            self.insights.append(
                Insight(
                    observation="No duplicate rows found.",
                    evidence=f"0 duplicate rows in {len(self.df):,} records.",
                    business_interpretation="Data integrity is strong with no redundant records.",
                    severity="info",
                    category="data_quality",
                )
            )
        elif dupes > len(self.df) * 0.1:
            self.insights.append(
                Insight(
                    observation="High duplicate row count.",
                    evidence=f"{dupes:,} duplicate rows ({dupes / len(self.df) * 100:.1f}% of dataset).",
                    business_interpretation="Duplicates may inflate metrics or skew aggregations. Deduplication is strongly recommended.",
                    severity="critical",
                    category="data_quality",
                )
            )
        else:
            self.insights.append(
                Insight(
                    observation="Some duplicate rows detected.",
                    evidence=f"{dupes:,} duplicate rows ({dupes / len(self.df) * 100:.1f}% of dataset).",
                    business_interpretation="Review whether duplicates represent true repeated events or data entry errors.",
                    severity="warning",
                    category="data_quality",
                )
            )

    def _detect_top_category_insight(self) -> None:
        cat_col = self.kpi_engine.category_col
        if not cat_col:
            return

        counts = self.df[cat_col].value_counts()
        if counts.empty:
            return

        top = counts.index[0]
        top_count = counts.iloc[0]
        pct = (top_count / len(self.df)) * 100

        self.insights.append(
            Insight(
                observation=f"'{top}' is the dominant category.",
                evidence=f"{top_count:,} records ({pct:.1f}%) fall under '{top}' in '{cat_col}'.",
                business_interpretation="Portfolio or inventory may be overly concentrated. Diversification or deeper segmentation analysis is advised.",
                severity="info" if pct < 50 else "warning",
                category="business",
            )
        )

    def _detect_top_region_insight(self) -> None:
        reg_col = self.kpi_engine.region_col
        if not reg_col:
            return

        counts = self.df[reg_col].value_counts()
        if counts.empty:
            return

        top = counts.index[0]
        top_count = counts.iloc[0]
        pct = (top_count / len(self.df)) * 100

        self.insights.append(
            Insight(
                observation=f"'{top}' is the top region by volume.",
                evidence=f"{top_count:,} records ({pct:.1f}%) originate from '{top}' in '{reg_col}'.",
                business_interpretation="Geographic concentration creates market risk. Expansion or replication of success in other regions should be explored.",
                severity="info" if pct < 50 else "warning",
                category="business",
            )
        )

    def _detect_revenue_concentration_insight(self) -> None:
        rev_col = self.kpi_engine.revenue_col
        if not rev_col or not pd.api.types.is_numeric_dtype(self.df[rev_col]):
            return

        series = self.df[rev_col].dropna()
        if series.empty or len(series) < 3:
            return

        sorted_vals = series.sort_values(ascending=False)
        top_20_pct_count = max(1, int(len(sorted_vals) * 0.2))
        top_20_sum = sorted_vals.iloc[:top_20_pct_count].sum()
        total = sorted_vals.sum()
        concentration = (top_20_sum / total * 100) if total != 0 else 0

        if concentration > 80:
            self.insights.append(
                Insight(
                    observation="Revenue is highly concentrated in a small subset of records.",
                    evidence=f"Top 20% of records contribute {concentration:.1f}% of total '{rev_col}'.",
                    business_interpretation="Extreme dependency on a few high-value records. Losing even one could significantly impact revenue.",
                    severity="critical",
                    category="business",
                )
            )
        elif concentration > 60:
            self.insights.append(
                Insight(
                    observation="Revenue shows moderate concentration.",
                    evidence=f"Top 20% of records contribute {concentration:.1f}% of total '{rev_col}'.",
                    business_interpretation="A significant portion of revenue comes from a minority of records. Customer retention strategy is important.",
                    severity="warning",
                    category="business",
                )
            )
        else:
            self.insights.append(
                Insight(
                    observation="Revenue is well distributed across records.",
                    evidence=f"Top 20% of records contribute {concentration:.1f}% of total '{rev_col}'.",
                    business_interpretation="Low concentration indicates a healthy, diversified revenue base.",
                    severity="info",
                    category="business",
                )
            )

    def _detect_growth_trend_insight(self) -> None:
        date_col = self.kpi_engine.date_col
        rev_col = self.kpi_engine.revenue_col

        if not date_col or not rev_col:
            return

        try:
            df_temp = self.df[[date_col, rev_col]].copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors="coerce")
            df_temp[rev_col] = pd.to_numeric(df_temp[rev_col], errors="coerce")
            df_temp = df_temp.dropna()

            if len(df_temp) < 3:
                return

            df_temp = df_temp.sort_values(by=date_col)
            df_temp["period"] = df_temp[date_col].dt.to_period("M")
            monthly = df_temp.groupby("period")[rev_col].sum().reset_index()
            monthly["period"] = monthly["period"].astype(str)

            if len(monthly) < 2:
                return

            first_half = monthly.iloc[: len(monthly) // 2][rev_col].mean()
            second_half = monthly.iloc[len(monthly) // 2 :][rev_col].mean()

            if first_half == 0:
                growth_pct = 0
            else:
                growth_pct = ((second_half - first_half) / first_half) * 100

            if growth_pct > 20:
                self.insights.append(
                    Insight(
                        observation="Strong positive growth trend detected.",
                        evidence=f"Average monthly '{rev_col}' increased by {growth_pct:.1f}% in recent periods.",
                        business_interpretation="Momentum is building. Consider scaling operations and marketing to capture the trend.",
                        severity="info",
                        category="trend",
                    )
                )
            elif growth_pct < -20:
                self.insights.append(
                    Insight(
                        observation="Significant decline trend detected.",
                        evidence=f"Average monthly '{rev_col}' decreased by {abs(growth_pct):.1f}% in recent periods.",
                        business_interpretation="Revenue is contracting. Immediate root-cause analysis and corrective action are recommended.",
                        severity="critical",
                        category="trend",
                    )
                )
            else:
                self.insights.append(
                    Insight(
                        observation="Stable or modest growth trend.",
                        evidence=f"Average monthly '{rev_col}' changed by {growth_pct:.1f}% across periods.",
                        business_interpretation="Performance is steady. Focus on incremental improvements and efficiency.",
                        severity="info",
                        category="trend",
                    )
                )
        except Exception as exc:
            logger.debug("Growth trend insight failed: %s", exc)

    def _detect_customer_concentration_insight(self) -> None:
        cust_col = self.kpi_engine.customer_col
        rev_col = self.kpi_engine.revenue_col

        if not cust_col or not rev_col:
            return

        try:
            grouped = self.df.groupby(cust_col)[rev_col].sum().sort_values(ascending=False)
            if grouped.empty or len(grouped) < 2:
                return

            top_customer_pct = (grouped.iloc[0] / grouped.sum()) * 100

            if top_customer_pct > 50:
                self.insights.append(
                    Insight(
                        observation="Extreme customer concentration risk.",
                        evidence=f"Single customer accounts for {top_customer_pct:.1f}% of total '{rev_col}'.",
                        business_interpretation="Revenue is dangerously dependent on one customer. Diversification is critical for business continuity.",
                        severity="critical",
                        category="business",
                    )
                )
            elif top_customer_pct > 30:
                self.insights.append(
                    Insight(
                        observation="High customer concentration.",
                        evidence=f"Top customer accounts for {top_customer_pct:.1f}% of total '{rev_col}'.",
                        business_interpretation="Major customer dependency exists. Negotiate long-term contracts and develop secondary accounts.",
                        severity="warning",
                        category="business",
                    )
                )
        except Exception as exc:
            logger.debug("Customer concentration insight failed: %s", exc)

    def _detect_low_performing_region_insight(self) -> None:
        reg_col = self.kpi_engine.region_col
        rev_col = self.kpi_engine.revenue_col

        if not reg_col or not rev_col:
            return

        try:
            region_revenue = (
                self.df.groupby(reg_col)[rev_col].sum().sort_values(ascending=True)
            )
            if region_revenue.empty or len(region_revenue) < 2:
                return

            bottom = region_revenue.index[0]
            bottom_val = region_revenue.iloc[0]
            top_val = region_revenue.iloc[-1]
            gap_pct = ((top_val - bottom_val) / top_val * 100) if top_val != 0 else 0

            self.insights.append(
                Insight(
                    observation=f"'{bottom}' is the lowest-performing region.",
                    evidence=f"Revenue in '{bottom}' is {self.kpi_engine._format_currency(bottom_val)}, {gap_pct:.1f}% below the top region.",
                    business_interpretation="Underperformance may indicate market saturation, operational issues, or lack of presence. Investigation warranted.",
                    severity="warning" if gap_pct > 70 else "info",
                    category="business",
                )
            )
        except Exception as exc:
            logger.debug("Low-performing region insight failed: %s", exc)

    # ------------------------------------------------------------------
    # Recommendation Generators
    # ------------------------------------------------------------------

    def _recommend_data_quality(self) -> None:
        total_missing = int(self.df.isna().sum().sum())
        dupes = int(self.df.duplicated().sum())

        if total_missing > 0 or dupes > 0:
            issues = []
            if total_missing > 0:
                issues.append(f"{total_missing:,} missing values")
            if dupes > 0:
                issues.append(f"{dupes:,} duplicate rows")

            self.recommendations.append(
                Recommendation(
                    title="Improve Data Quality",
                    description="Data quality issues were detected that may compromise analysis accuracy. Apply cleaning operations (missing value imputation, deduplication, type fixing) before making strategic decisions.",
                    evidence="; ".join(issues) + ".",
                    priority="high" if (total_missing > len(self.df) * 0.1 or dupes > len(self.df) * 0.05) else "medium",
                    action_category="data_quality",
                )
            )

    def _recommend_top_category_focus(self) -> None:
        cat_col = self.kpi_engine.category_col
        if not cat_col:
            return

        counts = self.df[cat_col].value_counts()
        if counts.empty:
            return

        top = counts.index[0]
        pct = (counts.iloc[0] / len(self.df)) * 100

        if pct > 40:
            self.recommendations.append(
                Recommendation(
                    title=f"Focus on '{top}' Category",
                    description=f"'{top}' represents {pct:.1f}% of your dataset. Double down on this category through inventory expansion, targeted marketing, or premium pricing.",
                    evidence=f"{counts.iloc[0]:,} records ({pct:.1f}%) are categorized as '{top}'.",
                    priority="high" if pct > 60 else "medium",
                    action_category="business_focus",
                )
            )
        else:
            self.recommendations.append(
                Recommendation(
                    title="Diversify Category Portfolio",
                    description="No single category dominates. Explore cross-selling opportunities across multiple categories.",
                    evidence=f"Top category '{top}' accounts for only {pct:.1f}% of records.",
                    priority="medium",
                    action_category="business_focus",
                )
            )

    def _recommend_region_investigation(self) -> None:
        reg_col = self.kpi_engine.region_col
        rev_col = self.kpi_engine.revenue_col

        if not reg_col:
            return

        try:
            if rev_col and pd.api.types.is_numeric_dtype(self.df[rev_col]):
                region_revenue = (
                    self.df.groupby(reg_col)[rev_col].sum().sort_values(ascending=True)
                )
                if region_revenue.empty or len(region_revenue) < 2:
                    return

                bottom = region_revenue.index[0]
                bottom_val = region_revenue.iloc[0]
                self.recommendations.append(
                    Recommendation(
                        title=f"Investigate Low-Performing Region: {bottom}",
                        description=f"Region '{bottom}' shows significantly lower revenue. Audit local operations, marketing spend, and competitive landscape.",
                        evidence=f"Lowest revenue region: {bottom} with {self.kpi_engine._format_currency(bottom_val)}.",
                        priority="high",
                        action_category="market_expansion",
                    )
                )
            else:
                counts = self.df[reg_col].value_counts().sort_values(ascending=True)
                if counts.empty or len(counts) < 2:
                    return

                bottom = counts.index[0]
                self.recommendations.append(
                    Recommendation(
                        title=f"Investigate Low-Volume Region: {bottom}",
                        description=f"Region '{bottom}' has the fewest records. Evaluate market potential and sales coverage.",
                        evidence=f"Lowest volume region: {bottom} with {counts.iloc[0]:,} records.",
                        priority="medium",
                        action_category="market_expansion",
                    )
                )
        except Exception:
            pass

    def _recommend_customer_diversification(self) -> None:
        cust_col = self.kpi_engine.customer_col
        rev_col = self.kpi_engine.revenue_col

        if not cust_col or not rev_col:
            return

        try:
            grouped = (
                self.df.groupby(cust_col)[rev_col].sum().sort_values(ascending=False)
            )
            if grouped.empty or len(grouped) < 2:
                return

            top_pct = (grouped.iloc[0] / grouped.sum()) * 100
            if top_pct > 30:
                self.recommendations.append(
                    Recommendation(
                        title="Reduce Dependency on a Single Customer",
                        description="A disproportionate share of revenue comes from one customer. Develop account diversification and retention strategies to mitigate risk.",
                        evidence=f"Top customer contributes {top_pct:.1f}% of total revenue.",
                        priority="high" if top_pct > 50 else "medium",
                        action_category="risk_management",
                    )
                )
        except Exception:
            pass

    def _recommend_growth_analysis(self) -> None:
        date_col = self.kpi_engine.date_col
        rev_col = self.kpi_engine.revenue_col

        if not date_col or not rev_col:
            self.recommendations.append(
                Recommendation(
                    title="Capture Date and Revenue Data",
                    description="Date and revenue columns were not detected. Adding these fields will enable trend analysis, forecasting, and seasonal planning.",
                    evidence="No date or revenue column detected by heuristic matching.",
                    priority="medium",
                    action_category="data_enrichment",
                )
            )
            return

        for insight in self.insights:
            if insight.category == "trend":
                if "decline" in insight.observation.lower():
                    self.recommendations.append(
                        Recommendation(
                            title="Address Revenue Decline",
                            description="Recent periods show declining revenue. Conduct root-cause analysis on pricing, churn, and market conditions.",
                            evidence=insight.evidence,
                            priority="high",
                            action_category="revenue_protection",
                        )
                    )
                elif "growth" in insight.observation.lower():
                    self.recommendations.append(
                        Recommendation(
                            title="Capitalize on Growth Momentum",
                            description="Positive growth trend detected. Increase capacity and marketing to capture demand before competitors.",
                            evidence=insight.evidence,
                            priority="medium",
                            action_category="growth",
                        )
                    )
                break


def generate_insights(df: pd.DataFrame) -> list[Insight]:
    """Convenience function."""
    engine = InsightEngine(df)
    return engine.generate_insights()


def generate_recommendations(df: pd.DataFrame) -> list[Recommendation]:
    """Convenience function."""
    engine = InsightEngine(df)
    engine.generate_insights()
    return engine.generate_recommendations()