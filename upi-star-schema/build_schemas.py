"""One-shot build pipeline: Excel → star schema DataFrames, cached as CSV.

Run directly to (re)build the cache and print a quick preview:
    python build_schemas.py
"""
from pathlib import Path
from typing import Dict

import pandas as pd

from src.ingest import load_all_data
from src.schemas import build_star_schemas
from src.metrics import build_trend_table, build_mandate_summary

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"

_SCHEMA_NAMES = ["dim_month", "dim_date", "dim_bank", "fact_monthly", "fact_monthly_agg", "fact_daily"]

_DATE_COLUMNS = {
    "dim_month": ["month_date"],
    "dim_date": ["full_date"],
}


def load_or_build_schemas() -> Dict:
    """Load cached CSVs from data/processed/ if present, otherwise run the
    full ingest → build pipeline and write the cache for next time."""
    sentinel = PROCESSED_DIR / "fact_monthly.csv"
    if sentinel.exists():
        schemas = {}
        for name in _SCHEMA_NAMES:
            path = PROCESSED_DIR / f"{name}.csv"
            parse_dates = _DATE_COLUMNS.get(name)
            schemas[name] = pd.read_csv(path, parse_dates=parse_dates)
        return schemas

    data = load_all_data()
    schemas = build_star_schemas(data)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in schemas.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    return schemas


if __name__ == "__main__":
    schemas = load_or_build_schemas()

    print("=" * 70)
    print("STAR SCHEMA BUILD SUMMARY")
    print("=" * 70)
    for name in _SCHEMA_NAMES:
        print(f"  {name:20s} {schemas[name].shape[0]:5d} rows  x  {schemas[name].shape[1]} cols")

    print("\n" + "=" * 70)
    print("6-MONTH TREND (fact_monthly_agg)")
    print("=" * 70)
    print(build_trend_table(schemas).to_string(index=False))

    print("\n" + "=" * 70)
    print("6-MONTH MANDATE SUMMARY (fact_monthly)")
    print("=" * 70)
    print(build_mandate_summary(schemas).to_string(index=False))
