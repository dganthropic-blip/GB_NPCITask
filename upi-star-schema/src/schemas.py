"""Build the Kimball star schema (2 fact tables, 3 conformed dimensions) from
the normalized DataFrames produced by src.ingest.

Two fact tables exist because NPCI publishes two incompatible grains:
aggregate month/day totals (no bank split) and bank-level mandate counts.
Combining them into one fact table would mean fabricating per-bank
transaction volume/value, which NPCI does not publish. See notes.md.
"""
from datetime import date
from typing import Dict, List

import pandas as pd

CUTOFF_DATE = date(2026, 6, 30)


def _quarter_of(month_num: int) -> str:
    return f"Q{(month_num - 1) // 3 + 1}"


def _fiscal_year_of(year: int, month_num: int) -> str:
    """Indian fiscal year runs April-March, e.g. FY2026-27 = Apr-2026..Mar-2027."""
    start = year if month_num >= 4 else year - 1
    return f"FY{start}-{str(start + 1)[-2:]}"


# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------

def _build_dim_month(month_dates: pd.Series) -> pd.DataFrame:
    unique_dates = sorted(pd.Series(month_dates).dropna().unique())
    rows = []
    for i, md in enumerate(unique_dates, start=1):
        ts = pd.Timestamp(md).normalize()
        rows.append({
            "month_key": i,
            "month_date": ts,
            "year": ts.year,
            "month_num": ts.month,
            "month_name": ts.strftime("%B"),
            "month_short": f"{ts.strftime('%b')}-{ts.year}",
            "quarter": _quarter_of(ts.month),
            "fiscal_year": _fiscal_year_of(ts.year, ts.month),
        })
    return pd.DataFrame(rows, columns=[
        "month_key", "month_date", "year", "month_num", "month_name",
        "month_short", "quarter", "fiscal_year",
    ])


def _build_dim_date(full_dates: pd.Series) -> pd.DataFrame:
    unique_dates = sorted(pd.Series(full_dates).dropna().unique())
    rows = []
    for i, fd in enumerate(unique_dates, start=1):
        ts = pd.Timestamp(fd).normalize()
        dow = ts.weekday()  # 0 = Monday .. 6 = Sunday
        rows.append({
            "date_key": i,
            "full_date": ts,
            "day": ts.day,
            "month_num": ts.month,
            "year": ts.year,
            "month_name": ts.strftime("%B"),
            "month_short": f"{ts.strftime('%b')}-{ts.year}",
            "day_name": ts.strftime("%A"),
            "day_of_week": dow,
            "is_weekend": dow >= 5,
        })
    return pd.DataFrame(rows, columns=[
        "date_key", "full_date", "day", "month_num", "year", "month_name",
        "month_short", "day_name", "day_of_week", "is_weekend",
    ])


def _build_dim_bank(bank_names: List[str]) -> pd.DataFrame:
    rows = [{"bank_key": i, "bank_name": name} for i, name in enumerate(sorted(bank_names), start=1)]
    return pd.DataFrame(rows, columns=["bank_key", "bank_name"])


# ---------------------------------------------------------------------------
# Fact builders
# ---------------------------------------------------------------------------

def _build_fact_monthly_agg(monthly_df: pd.DataFrame, dim_month: pd.DataFrame) -> pd.DataFrame:
    month_map = dict(zip(dim_month["month_date"], dim_month["month_key"]))
    df = monthly_df.copy()
    df["month_key"] = df["month_date"].map(month_map)
    df = df.dropna(subset=["month_key"]).sort_values("month_date").reset_index(drop=True)
    df["month_key"] = df["month_key"].astype(int)

    df["ats_rs"] = (df["value_cr"] * 10) / df["volume_mn"]
    df["volume_mom_pct"] = df["volume_mn"].pct_change() * 100
    df["value_mom_pct"] = df["value_cr"].pct_change() * 100
    df["ats_mom_pct"] = df["ats_rs"].pct_change() * 100

    cols = [
        "month_key", "volume_mn", "value_cr", "avg_daily_volume_mn",
        "avg_daily_value_cr", "ats_rs", "volume_mom_pct", "value_mom_pct",
        "ats_mom_pct",
    ]
    return df[cols].reset_index(drop=True)


def _build_fact_daily(daily_df: pd.DataFrame, dim_date: pd.DataFrame) -> pd.DataFrame:
    date_map = dict(zip(dim_date["full_date"], dim_date["date_key"]))
    df = daily_df.copy()
    df["date_key"] = df["full_date"].map(date_map)
    df = df.dropna(subset=["date_key"]).sort_values("date_key").reset_index(drop=True)
    df["date_key"] = df["date_key"].astype(int)
    df["ats_rs"] = (df["value_cr"] * 10) / df["volume_mn"]
    return df[["date_key", "volume_mn", "value_cr", "ats_rs"]].reset_index(drop=True)


def _build_fact_monthly(
    creation_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    dim_bank: pd.DataFrame,
    dim_month: pd.DataFrame,
) -> pd.DataFrame:
    bank_map = dict(zip(dim_bank["bank_name"], dim_bank["bank_key"]))
    month_map = dict(zip(dim_month["month_date"], dim_month["month_key"]))

    def _prep(df: pd.DataFrame, volume_col: str, prefix: str) -> pd.DataFrame:
        d = df.copy()
        d["bank_key"] = d["bank_name"].map(bank_map)
        d["month_key"] = d["month_date"].map(month_map)
        d = d.dropna(subset=["bank_key", "month_key"])
        d["bank_key"] = d["bank_key"].astype(int)
        d["month_key"] = d["month_key"].astype(int)
        d = d.rename(columns={
            "total_volume": volume_col,
            "approved_pct": f"{prefix}_approved_pct",
            "bd_pct": f"{prefix}_bd_pct",
            "td_pct": f"{prefix}_td_pct",
        })
        cols = ["bank_key", "month_key", volume_col, f"{prefix}_approved_pct", f"{prefix}_bd_pct", f"{prefix}_td_pct"]
        return d[[c for c in cols if c in d.columns]]

    creation = _prep(creation_df, "mandates_created", "creation")
    execution = _prep(execution_df, "mandates_executed", "execution")

    # Outer join → banks/months absent from one side become NULL, never zero.
    merged = creation.merge(execution, on=["bank_key", "month_key"], how="outer")
    merged["execution_ratio_pct"] = None
    has_both = merged["mandates_created"].notna() & (merged["mandates_created"] != 0)
    merged.loc[has_both, "execution_ratio_pct"] = (
        merged.loc[has_both, "mandates_executed"] / merged.loc[has_both, "mandates_created"] * 100
    )

    merged["mandates_created"] = merged["mandates_created"].astype("Int64")
    merged["mandates_executed"] = merged["mandates_executed"].astype("Int64")
    merged["execution_ratio_pct"] = pd.to_numeric(merged["execution_ratio_pct"], errors="coerce")

    cols = [
        "bank_key", "month_key", "mandates_created", "mandates_executed",
        "creation_approved_pct", "creation_bd_pct", "creation_td_pct",
        "execution_approved_pct", "execution_bd_pct", "execution_td_pct",
        "execution_ratio_pct",
    ]
    return merged[cols].sort_values(["month_key", "bank_key"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_star_schemas(data: Dict) -> Dict:
    """Build all dimension and fact tables from src.ingest.load_all_data() output.

    Returns:
        {"dim_month": df, "dim_date": df, "dim_bank": df,
         "fact_monthly": df, "fact_monthly_agg": df, "fact_daily": df}
    """
    monthly_df = data["monthly"].copy()
    monthly_df["month_date"] = pd.to_datetime(monthly_df["month_date"]).dt.normalize()
    monthly_df = monthly_df[monthly_df["month_date"].dt.date <= CUTOFF_DATE]
    monthly_df = monthly_df.sort_values("month_date").reset_index(drop=True)

    daily_frames = [d.copy() for d in data["daily"]]
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(
        columns=["full_date", "volume_mn", "value_cr"]
    )
    daily_df["full_date"] = pd.to_datetime(daily_df["full_date"]).dt.normalize()
    daily_df = daily_df[daily_df["full_date"].dt.date <= CUTOFF_DATE]
    daily_df = daily_df.drop_duplicates(subset=["full_date"]).sort_values("full_date").reset_index(drop=True)

    creation_frames = [d.copy() for d in data["mandate_creation"]]
    execution_frames = [d.copy() for d in data["mandate_execution"]]
    creation_df = pd.concat(creation_frames, ignore_index=True) if creation_frames else pd.DataFrame(
        columns=["bank_name", "total_volume", "approved_pct", "bd_pct", "td_pct", "month_date"]
    )
    execution_df = pd.concat(execution_frames, ignore_index=True) if execution_frames else pd.DataFrame(
        columns=creation_df.columns
    )
    creation_df["month_date"] = pd.to_datetime(creation_df["month_date"]).dt.normalize()
    execution_df["month_date"] = pd.to_datetime(execution_df["month_date"]).dt.normalize()
    creation_df = creation_df[creation_df["month_date"].dt.date <= CUTOFF_DATE]
    execution_df = execution_df[execution_df["month_date"].dt.date <= CUTOFF_DATE]

    dim_month = _build_dim_month(monthly_df["month_date"])
    dim_date = _build_dim_date(daily_df["full_date"])
    bank_names = sorted(set(creation_df["bank_name"].dropna()) | set(execution_df["bank_name"].dropna()))
    dim_bank = _build_dim_bank(bank_names)

    fact_monthly_agg = _build_fact_monthly_agg(monthly_df, dim_month)
    fact_daily = _build_fact_daily(daily_df, dim_date)
    fact_monthly = _build_fact_monthly(creation_df, execution_df, dim_bank, dim_month)

    return {
        "dim_month": dim_month,
        "dim_date": dim_date,
        "dim_bank": dim_bank,
        "fact_monthly": fact_monthly,
        "fact_monthly_agg": fact_monthly_agg,
        "fact_daily": fact_daily,
    }
