"""Read and normalize the NPCI UPI Excel files (monthly, daily, mandate creation/execution).

NPCI publishes messy, hand-formatted Excel exports: garbage header rows, tab
characters embedded in column names, comma-formatted Indian numbers as text,
percentage strings, and inconsistent bank-name casing across months. This
module is the single place that cleans all of that up before anything is
loaded into the star schema.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Step 1: manual overrides for known problem bank names (see notes.md).
_BANK_NAME_OVERRIDES = {
    "hdfc bank": "HDFC Bank Ltd.",
    "bandhan bank limited": "Bandhan Bank Limited",
}


# ---------------------------------------------------------------------------
# Excel reading helpers
# ---------------------------------------------------------------------------

def _looks_like_bad_header(columns) -> bool:
    """True if the header row looks like data rather than column names."""
    cols = [str(c) for c in columns]
    if not cols:
        return True
    bad = 0
    for c in cols:
        if c.lower().startswith("unnamed"):
            bad += 1
            continue
        try:
            float(c)
            bad += 1
        except ValueError:
            pass
    return bad == len(cols)


def _read_excel_smart(path: Path) -> pd.DataFrame:
    """Read an NPCI Excel file, probing header rows 0-5 until columns look sane."""
    chosen = None
    for header_row in range(6):
        try:
            df = pd.read_excel(path, engine="openpyxl", header=header_row)
        except Exception:
            continue
        if df.empty and df.columns.empty:
            continue
        if not _looks_like_bad_header(df.columns):
            chosen = df
            break
    if chosen is None:
        chosen = pd.read_excel(path, engine="openpyxl", header=0)

    chosen = chosen.dropna(how="all")
    chosen = chosen.dropna(axis=1, how="all")
    chosen.columns = [str(c).replace("\t", "").strip() for c in chosen.columns]
    return chosen


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def _rename_monthly_columns(df: pd.DataFrame) -> pd.DataFrame:
    orig_cols = list(df.columns)
    rename = {}
    month_found = False
    for col in orig_cols:
        lc = col.lower()
        if "month" in lc:
            rename[col] = "month_str"
            month_found = True
        elif "avg" in lc and "vol" in lc:
            rename[col] = "avg_daily_volume_mn"
        elif "avg" in lc and ("val" in lc or "amount" in lc):
            rename[col] = "avg_daily_value_cr"
        elif "vol" in lc:
            rename[col] = "volume_mn"
        elif "val" in lc or "amount" in lc:
            rename[col] = "value_cr"
    df = df.rename(columns=rename)
    if not month_found:
        df = df.rename(columns={orig_cols[0]: "month_str"})
    return df


def _rename_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    orig_cols = list(df.columns)
    rename = {}
    date_found = False
    for col in orig_cols:
        lc = col.lower()
        if "date" in lc or "day" in lc:
            rename[col] = "date_str"
            date_found = True
        elif "vol" in lc:
            rename[col] = "volume_mn"
        elif "val" in lc:
            rename[col] = "value_cr"
    df = df.rename(columns=rename)
    if not date_found:
        df = df.rename(columns={orig_cols[0]: "date_str"})
    return df


def _rename_mandate_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if "bank" in lc or "remitter" in lc:
            rename[col] = "bank_name"
        elif "total" in lc and "vol" in lc:
            rename[col] = "total_volume"
        elif "approved" in lc:
            rename[col] = "approved_pct"
        elif "bd" in lc:
            rename[col] = "bd_pct"
        elif "td" in lc:
            rename[col] = "td_pct"
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Strip Indian-style comma separators and percent signs, coerce to float."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_month_str(value) -> Optional[pd.Timestamp]:
    """Parse strings like 'March-2026' or 'December-2025' into a Timestamp."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    for fmt in ("%B-%Y", "%b-%Y"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def _parse_date_str(value) -> Optional[pd.Timestamp]:
    """Parse strings like 'January 01, 2026'. Non-date rows (e.g. 'Total') → NaT."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    parsed = pd.to_datetime(text, format="%B %d, %Y", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    return parsed


def _extract_month_from_filename(path: Path) -> Optional[pd.Timestamp]:
    """Matches 'June-2026' or 'Jan-2026' style tokens in a filename stem."""
    match = re.search(r"([A-Za-z]+)-(\d{4})", path.stem)
    if not match:
        return None
    month_str, year_str = match.groups()
    for fmt in ("%d-%B-%Y", "%d-%b-%Y"):
        parsed = pd.to_datetime(f"1-{month_str}-{year_str}", format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return None


# ---------------------------------------------------------------------------
# Bank name normalization
# ---------------------------------------------------------------------------

def _normalize_bank_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    lower = name.lower()
    if lower in _BANK_NAME_OVERRIDES:
        return _BANK_NAME_OVERRIDES[lower]
    return name


def _build_canonical_bank_map(all_bank_names) -> Dict[str, str]:
    """Group bank names case-insensitively, prefer the non-ALL-CAPS variant."""
    groups: Dict[str, List[str]] = {}
    for name in all_bank_names:
        key = name.lower()
        groups.setdefault(key, []).append(name)
    canonical: Dict[str, str] = {}
    for key, variants in groups.items():
        preferred = next((v for v in variants if not v.isupper()), variants[0])
        for v in variants:
            canonical[v] = preferred
    return canonical


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _glob_files(raw_dir: Path, *patterns: str) -> List[Path]:
    seen = set()
    result: List[Path] = []
    for pattern in patterns:
        for p in sorted(raw_dir.glob(pattern)):
            if p not in seen:
                seen.add(p)
                result.append(p)
    return sorted(result)


# ---------------------------------------------------------------------------
# Per-category loaders
# ---------------------------------------------------------------------------

def _load_monthly_stats(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = _read_excel_smart(path)
        df = _rename_monthly_columns(df)
        for col in ("volume_mn", "value_cr", "avg_daily_volume_mn", "avg_daily_value_cr"):
            if col in df.columns:
                df[col] = _to_numeric_series(df[col])
        df["month_date"] = df["month_str"].apply(_parse_month_str)
        keep_cols = [c for c in ["month_date", "volume_mn", "value_cr", "avg_daily_volume_mn", "avg_daily_value_cr"] if c in df.columns]
        df = df[keep_cols].dropna(subset=["month_date", "volume_mn", "value_cr"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["month_date", "volume_mn", "value_cr", "avg_daily_volume_mn", "avg_daily_value_cr"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["month_date"]).sort_values("month_date").reset_index(drop=True)
    return combined


def _load_daily_stats(paths: List[Path]) -> List[pd.DataFrame]:
    frames = []
    for path in paths:
        df = _read_excel_smart(path)
        df = _rename_daily_columns(df)
        for col in ("volume_mn", "value_cr"):
            if col in df.columns:
                df[col] = _to_numeric_series(df[col])
        df["full_date"] = df["date_str"].apply(_parse_date_str)
        df = df[["full_date", "volume_mn", "value_cr"]].dropna(subset=["full_date", "volume_mn", "value_cr"])
        frames.append(df.sort_values("full_date").reset_index(drop=True))
    return frames


def _load_mandate_files(paths: List[Path]) -> List[pd.DataFrame]:
    frames = []
    for path in paths:
        df = _read_excel_smart(path)
        df = _rename_mandate_columns(df)
        month_date = _extract_month_from_filename(path)
        if "bank_name" not in df.columns:
            continue
        df["bank_name"] = df["bank_name"].apply(_normalize_bank_name)
        df = df.dropna(subset=["bank_name"])
        for col in ("total_volume", "approved_pct", "bd_pct", "td_pct"):
            if col in df.columns:
                df[col] = _to_numeric_series(df[col])
        df["month_date"] = month_date
        keep_cols = [c for c in ["bank_name", "total_volume", "approved_pct", "bd_pct", "td_pct", "month_date"] if c in df.columns]
        frames.append(df[keep_cols].reset_index(drop=True))
    return frames


def _apply_canonical_bank_map(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    all_names = set()
    for df in frames:
        all_names.update(df["bank_name"].dropna().unique().tolist())
    canonical_map = _build_canonical_bank_map(all_names)
    out = []
    for df in frames:
        df = df.copy()
        df["bank_name"] = df["bank_name"].map(canonical_map).fillna(df["bank_name"])
        out.append(df)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_all_data(raw_dir: Optional[Path] = None) -> Dict:
    """Read every NPCI Excel file in data/raw/, normalize columns and bank names.

    Returns:
        {
            "monthly": DataFrame (month_date, volume_mn, value_cr, avg_daily_volume_mn, avg_daily_value_cr),
            "daily": [DataFrame (full_date, volume_mn, value_cr), ...],
            "mandate_creation": [DataFrame (bank_name, total_volume, approved_pct, bd_pct, td_pct, month_date), ...],
            "mandate_execution": [DataFrame (bank_name, total_volume, approved_pct, bd_pct, td_pct, month_date), ...],
        }
    """
    raw_dir = raw_dir or RAW_DIR

    monthly_paths = _glob_files(raw_dir, "*[Mm]onthly*[Ss]tatist*", "*monthly-statistics*")
    daily_paths = _glob_files(raw_dir, "*[Dd]aily*", "*daily-statistics*")
    creation_paths = _glob_files(raw_dir, "*[Mm]andate*[Cc]reation*")
    execution_paths = _glob_files(raw_dir, "*[Mm]andate*[Ee]xecution*")

    monthly_df = _load_monthly_stats(monthly_paths)
    daily_dfs = _load_daily_stats(daily_paths)
    creation_dfs = _load_mandate_files(creation_paths)
    execution_dfs = _load_mandate_files(execution_paths)

    all_mandate_frames = _apply_canonical_bank_map(creation_dfs + execution_dfs)
    creation_dfs = all_mandate_frames[: len(creation_dfs)]
    execution_dfs = all_mandate_frames[len(creation_dfs):]

    return {
        "monthly": monthly_df,
        "daily": daily_dfs,
        "mandate_creation": creation_dfs,
        "mandate_execution": execution_dfs,
    }
