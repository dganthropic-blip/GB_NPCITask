"""Trend-table and mandate-summary builders used by build_schemas.py's CLI
preview and available for reuse anywhere a small pre-aggregated view is
handy. The conversational agent does NOT call these — it writes its own SQL
against the star schema (see src/tools.py).
"""
from typing import Dict

import pandas as pd


def build_trend_table(schemas: Dict, months: int = 6) -> pd.DataFrame:
    """Last N months of fact_monthly_agg joined to dim_month, most recent last."""
    fact = schemas["fact_monthly_agg"]
    dim_month = schemas["dim_month"]
    merged = fact.merge(dim_month, on="month_key", how="inner")
    merged = merged.sort_values("month_date").tail(months).reset_index(drop=True)
    cols = [
        "month_short", "volume_mn", "value_cr", "ats_rs",
        "volume_mom_pct", "value_mom_pct", "ats_mom_pct",
    ]
    return merged[cols]


def build_mandate_summary(schemas: Dict, months: int = 6) -> pd.DataFrame:
    """Last N months of fact_monthly aggregated across all banks."""
    fact = schemas["fact_monthly"]
    dim_month = schemas["dim_month"]
    merged = fact.merge(dim_month, on="month_key", how="inner")

    grouped = merged.groupby(["month_key", "month_date", "month_short"], as_index=False).agg(
        total_created=("mandates_created", "sum"),
        total_executed=("mandates_executed", "sum"),
    )
    grouped["execution_ratio_pct"] = (
        grouped["total_executed"] / grouped["total_created"] * 100
    ).round(2)
    grouped = grouped.sort_values("month_date").tail(months).reset_index(drop=True)
    return grouped[["month_short", "total_created", "total_executed", "execution_ratio_pct"]]
