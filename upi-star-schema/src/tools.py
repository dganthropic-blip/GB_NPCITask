"""DuckDB-backed text-to-SQL query engine exposed to the conversational
agent as 3 tools. The agent writes its own SQL — there are no pre-built
query functions for specific question shapes ("get_trend", "top_banks",
etc). This keeps the agent general: any question answerable by SQL over
the star schema is answerable, not just the ones we anticipated.
"""
import json
import re
from typing import Dict, List, Optional

import duckdb
import pandas as pd

_BLOCKED_KEYWORDS = [
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
    "ATTACH", "COPY", "EXPORT", "IMPORT",
]

_COMMENT_STRIP_RE = re.compile(r"--.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)


def _json_safe(records: List[Dict]) -> List[Dict]:
    """Convert pandas/numpy scalars (Timestamp, NaT, NaN, int64, ...) to
    plain JSON-serializable Python values."""
    safe = []
    for row in records:
        clean = {}
        for k, v in row.items():
            if v is None:
                clean[k] = None
            elif isinstance(v, pd.Timestamp):
                clean[k] = v.strftime("%Y-%m-%d")
            elif isinstance(v, (list, dict, str)):
                clean[k] = v
            elif pd.isna(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        safe.append(clean)
    return safe


class SchemaQueryEngine:
    """Registers the star schema DataFrames as DuckDB tables and answers
    both discovery queries and arbitrary read-only SQL."""

    def __init__(self, schemas: Dict[str, pd.DataFrame]):
        self.schemas = schemas
        self.con = duckdb.connect(":memory:")
        for table_name, df in schemas.items():
            self.con.register(table_name, df)

    # -- core SQL execution -------------------------------------------------

    def execute_sql(self, query: str) -> str:
        try:
            result = self.con.execute(query).fetchdf()
        except Exception as exc:  # noqa: BLE001 - surfaced to the LLM as text
            return json.dumps({"error": str(exc)})
        records = result.to_dict(orient="records")
        return json.dumps(_json_safe(records), default=str)

    @staticmethod
    def _validate_sql(query: str) -> str:
        """Return an error message string if the query is disallowed, else ''."""
        if not query or not query.strip():
            return "Query is empty."
        if len(query) > 2000:
            return "Query exceeds the 2000 character limit."

        stripped = _COMMENT_STRIP_RE.sub(" ", query).strip()
        if not stripped:
            return "Query is empty after stripping comments."

        first_word_match = re.match(r"^\s*(\w+)", stripped, re.IGNORECASE)
        first_word = first_word_match.group(1).upper() if first_word_match else ""
        if first_word not in ("SELECT", "WITH"):
            return "Only SELECT and WITH (CTE) queries are allowed."

        for keyword in _BLOCKED_KEYWORDS:
            if re.search(rf"\b{keyword}\b", stripped, re.IGNORECASE):
                return f"Query contains a blocked keyword: {keyword}."

        return ""

    def run_sql(self, query: str) -> str:
        error = self._validate_sql(query)
        if error:
            return json.dumps({"error": error})
        return self.execute_sql(query)

    # -- schema discovery -----------------------------------------------------

    def describe_tables(self, table_name: Optional[str] = None) -> str:
        if table_name is None:
            tables = []
            for name, df in self.schemas.items():
                tables.append({
                    "table": name,
                    "columns": len(df.columns),
                    "rows": len(df),
                    "column_names": list(df.columns),
                })
            return json.dumps(tables, default=str)

        if table_name not in self.schemas:
            return json.dumps({"error": f"Unknown table '{table_name}'. Known tables: {list(self.schemas.keys())}"})

        df = self.schemas[table_name]
        columns = []
        for col in df.columns:
            series = df[col]
            entry = {"name": col, "dtype": str(series.dtype)}
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    entry["min"] = str(non_null.min())
                    entry["max"] = str(non_null.max())
            else:
                sample_values = series.dropna().unique()[:5].tolist()
                entry["sample_values"] = [str(v) for v in sample_values]
            columns.append(entry)

        sample_row = {}
        if len(df) > 0:
            sample_row = _json_safe(df.head(1).to_dict(orient="records"))[0]

        return json.dumps({
            "table": table_name,
            "rows": len(df),
            "columns": columns,
            "sample_row": sample_row,
        }, default=str)

    # -- convenience tool -----------------------------------------------------

    def query_daily_stats(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 31,
    ) -> str:
        conditions = []
        if date_from:
            conditions.append(f"d.full_date >= '{date_from}'")
        if date_to:
            conditions.append(f"d.full_date <= '{date_to}'")
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT d.full_date, d.day_name, f.volume_mn, f.value_cr, f.ats_rs
            FROM fact_daily f
            JOIN dim_date d ON f.date_key = d.date_key
            {where_clause}
            ORDER BY d.full_date
            LIMIT {int(limit)}
        """
        return self.execute_sql(query)


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI/Groq-compatible function-calling schema)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "describe_tables",
        "description": (
            "Discover the star schema structure. Without table_name: returns all "
            "table names, column counts, and row counts. With table_name: returns "
            "column names, data types, value ranges, and a sample row. Call this "
            "first if unsure which table or column to query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Optional. One of: dim_month, dim_date, dim_bank, fact_monthly, fact_monthly_agg, fact_daily.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "query_daily_stats",
        "description": (
            "Get daily UPI volume (Mn), value (Cr), and ATS (Rs) from fact_daily "
            "joined with dim_date. Returns date, day name, and metrics. Filter by "
            "date range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "ISO date, e.g. '2026-06-01'."},
                "date_to": {"type": "string", "description": "ISO date, e.g. '2026-06-30'."},
                "limit": {"type": "integer", "description": "Max rows to return. Default 31."},
            },
            "required": [],
        },
    },
    {
        "name": "run_sql",
        "description": (
            "Execute a read-only SQL query (DuckDB syntax) against the UPI star "
            "schema. SELECT and WITH (CTEs) only. Max 2000 chars. SCHEMA: "
            "fact_monthly_agg(month_key, volume_mn, value_cr, avg_daily_volume_mn, "
            "avg_daily_value_cr, ats_rs, volume_mom_pct, value_mom_pct, ats_mom_pct); "
            "fact_monthly(bank_key, month_key, mandates_created, mandates_executed, "
            "creation_approved_pct, execution_approved_pct, execution_ratio_pct); "
            "fact_daily(date_key, volume_mn, value_cr, ats_rs); "
            "dim_month(month_key, month_date, year, month_num, month_name, month_short, "
            "quarter, fiscal_year); dim_date(date_key, full_date, day, month_num, year, "
            "month_name, month_short, day_name, day_of_week, is_weekend); "
            "dim_bank(bank_key, bank_name). JOIN on keys: "
            "fact_monthly_agg.month_key→dim_month.month_key, "
            "fact_monthly.bank_key→dim_bank.bank_key, "
            "fact_monthly.month_key→dim_month.month_key, "
            "fact_daily.date_key→dim_date.date_key."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A DuckDB SELECT or WITH query."},
            },
            "required": ["query"],
        },
    },
]
