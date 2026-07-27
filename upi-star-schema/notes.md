# Notes: UPI Star Schema + Conversational Agent

## 1. Data problems found

1. **Aggregate-only monthly/daily data.** NPCI does not publish per-bank
   transaction volume/value anywhere. `fact_monthly_agg` and `fact_daily`
   have no `bank_key` — that granularity simply does not exist in the
   source. Only mandate creation/execution is bank-level.
2. **Partial July 2026.** The FY2026-27 monthly file includes a July-2026
   row (17,766.24 Mn volume, ~22 of 31 days). `CUTOFF_DATE = 2026-06-30`
   in `src/schemas.py` excludes it from every fact table so it can't
   distort a monthly trend or MoM% calculation.
3. **Two FY files for one monthly series.** The 13-month aggregate
   (Jun-2025 → Jun-2026) is split across
   `Upi-monthly-statistics-2025-26-monthly.xlsx` (Jun-2025…Mar-2026) and
   `Upi-monthly-statistics-2026-27-monthly.xlsx` (Apr-2026…Jul-2026).
   `ingest.py` concatenates and de-duplicates by `month_date`.
4. **"Total" footer row in every daily file.** Each daily-statistics sheet
   ends with a `Total` row (e.g. January's sheet has 32 rows for 31
   days). The row's date string doesn't parse, so it becomes `NaT` and is
   dropped by `dropna(subset=["full_date", ...])`.
5. **Tab characters in execution-file headers.** Mandate *execution*
   files have headers like `"Sr. No.\t"` and `"BD%\t"` while creation
   files don't. Column names are stripped of `\t` before matching.
6. **`Total Volume` dtype varies by file.** Some files store it as text
   with Indian comma grouping (`"24,42,536"`), others as a native int64
   column (`29885`). Both are routed through the same
   comma/percent-stripping numeric coercion.
7. **20 separate source files, no fixed naming.** Files are matched by
   glob pattern (`*[Mm]onthly*[Ss]tatist*`, `*[Dd]aily*`,
   `*[Mm]andate*[Cc]reation*`, `*[Mm]andate*[Ee]xecution*`), not literal
   filenames, since NPCI's export naming shifts slightly month to month.
8. **ALL-CAPS bank names, April-2026 creation file only.** e.g.
   `"UTKARSH SMALL FINANCE BANK LIMITED"` vs. `"Utkarsh Small Finance
   Bank Limited"` everywhere else. Resolved by the canonical-name pass
   (group case-insensitively, prefer the non-ALL-CAPS variant).
9. **`HDFC Bank` vs `HDFC Bank Ltd.`** — the mandate files disagree on
   suffix. Fixed via a manual override in `_BANK_NAME_OVERRIDES`.
10. **`Bandhan Bank limited` vs `Bandhan Bank Limited`** — same problem,
    different casing of "limited". Also a manual override.
11. **Variable bank count per month (48-52 rows).** Creation/execution
    files don't list the same 52 banks every month; smaller banks drop
    in and out. Absent bank-months become `NULL` via an outer join, not 0.
12. **Banks in a month's creation file but not that month's execution
    file (and vice versa).** Across Jan-Jun 2026 this happens 5 times
    (e.g. G P Parsik Sahakari Bank in January). A bank creating mandates
    doesn't guarantee any got executed that same month.
13. **Mandate creation volume is wildly seasonal.** 8,460,409 mandates
    created in Jan-2026 vs. 109,705 in Apr-2026 (~77x). This is real
    IPO-application-driven seasonality in the source data, not a bug.
14. **G P Parsik Sahakari Bank Ltd — 0% approval anomaly.** In its one
    appearance (Jan-2026 creation), `approved_pct = 0.00%` and
    `bd_pct = 100.00%` — every mandate it created was business-declined.
15. **Execution count can exceed creation count in the same bank-month.**
    Mandates created in a prior month can execute in a later one, so
    `mandates_executed > mandates_created` for a given (bank, month) is
    expected, not an error.

## 2. Design decisions

1. **Two fact tables, two grains.** `fact_monthly_agg`/`fact_daily`
   (aggregate, no bank) and `fact_monthly` (bank × month, mandates only).
   Merging them would mean fabricating per-bank transaction data NPCI
   doesn't publish.
2. **DuckDB in-process (`duckdb.connect(":memory:")`).** Zero config, no
   server; pandas DataFrames are registered directly as SQL tables.
3. **ATS expressed in ₹ (Rupees), not paise or lakh.** `ats_rs = value_cr
   * 10 / volume_mn` is the most intuitive unit for a chat answer
   ("₹1,305" reads better than "0.1305 lakh").
4. **Complete months only.** `CUTOFF_DATE` excludes partial July-2026 so
   it can't quietly distort a trend or MoM comparison.
5. **CSV cache, not Parquet.** The processed tables are small (<400 rows
   each); CSV avoids adding a `pyarrow` dependency for no real benefit.
6. **Gemini API via Google AI Studio (`gemini-2.5-flash`).** Function
   calling is manual (`automatic_function_calling` disabled) so every
   tool call can be captured into `tools_used` for the UI's query-audit
   trail — the SDK's own auto-calling loop would hide that from us.
7. **Text-to-SQL, no pre-built query functions.** The agent has no
   `get_trend()`/`get_top_banks()`-style tools — it writes SQL itself via
   `run_sql`, so it can answer questions beyond the ones anticipated here.
8. **`run_sql` is the primary tool.** `describe_tables` handles schema
   discovery and `query_daily_stats` is a narrow, ready-made convenience
   for the single most common daily-lookup shape.
9. **13 months in `fact_monthly_agg` (Jun-2025 → Jun-2026).** Built by
   concatenating both FY monthly files and de-duplicating on `month_date`.
10. **System prompt embeds the full schema DDL and worked SQL patterns.**
    Avoids a wasted discovery round-trip via `describe_tables` on every
    single question.
11. **Flask + vanilla JS, no build step.** `templates/index.html` is one
    self-contained file — no React, no npm, nothing to compile.
12. **Canonical bank-name mapping.** Names are grouped case-insensitively
    across *all* mandate files after loading; the non-ALL-CAPS variant
    wins. Two additional manual overrides (HDFC, Bandhan) handle
    suffix/casing NPCI itself is inconsistent about.
13. **Outer join for `fact_monthly`.** A bank missing from a given
    month's file becomes `NULL` for that month's metrics — never 0 —
    since NPCI omits absent banks rather than reporting zero activity.
14. **`run_sql` is validated, not sandboxed by DB permissions.** Max 2000
    chars, blocked DDL/DML keywords (`DROP`, `DELETE`, `INSERT`, `UPDATE`,
    `ALTER`, `CREATE`, `ATTACH`, `COPY`, `EXPORT`, `IMPORT`), and the
    query must start with `SELECT` or `WITH` — enforced in Python before
    the string ever reaches DuckDB.
15. **`bd_pct`/`td_pct` stop at the ingestion layer.** They're parsed and
    available on the raw mandate DataFrames but intentionally excluded
    from `fact_monthly`, which only carries the columns the spec defines
    (`creation_approved_pct`, `execution_approved_pct`,
    `execution_ratio_pct`).

## 3. Star schema diagram

```
                    ┌───────────────┐
                    │   dim_month   │
                    │───────────────│
                    │ month_key PK  │
                    │ month_date    │
                    │ year          │
                    │ month_num     │
                    │ month_name    │
                    │ month_short   │
                    │ quarter       │
                    │ fiscal_year   │
                    └───────┬───────┘
                     ▲      │      ▲
                     │      │      │
        ┌────────────┘      │      └────────────┐
        │                   │                   │
┌───────┴────────┐  ┌───────┴────────┐  ┌────────┴───────┐
│ fact_monthly   │  │fact_monthly_agg│  │  fact_monthly  │
│    _agg        │  │  (same table,  │  │  (bank x month)│
│ (month grain)  │  │  shown twice   │  │────────────────│
│────────────────│  │  for clarity)  │  │ bank_key    FK ─┼──┐
│ month_key   FK │  └────────────────┘  │ month_key   FK  │  │
│ volume_mn      │                      │ mandates_created│  │
│ value_cr       │                      │ mandates_executed│ │
│ avg_daily_vol  │                      │ creation_appr_% │  │
│ avg_daily_val  │                      │ execution_appr_%│  │
│ ats_rs         │                      │ execution_ratio%│  │
│ volume_mom_%   │                      └────────┬─────────┘  │
│ value_mom_%    │                               │            │
│ ats_mom_%      │                               ▼            │
└────────────────┘                      ┌────────────────┐    │
                                         │   dim_bank     │◄───┘
                                         │────────────────│
                                         │ bank_key PK    │
                                         │ bank_name      │
                                         └────────────────┘

┌────────────────┐        ┌────────────────┐
│   fact_daily   │        │    dim_date    │
│  (date grain)  │        │────────────────│
│────────────────│───────►│ date_key PK    │
│ date_key   FK  │        │ full_date      │
│ volume_mn      │        │ day            │
│ value_cr       │        │ month_num      │
│ ats_rs         │        │ year           │
└────────────────┘        │ month_name     │
                           │ month_short    │
                           │ day_name       │
                           │ day_of_week    │
                           │ is_weekend     │
                           └────────────────┘
```

## 4. 6-month trend table (fact_monthly_agg, live build output)

| month_short | volume_mn | value_cr    | ats_rs   | volume_mom_% | value_mom_% | ats_mom_% |
|-------------|-----------|-------------|----------|--------------|-------------|-----------|
| Jan-2026    | 21,703.46 | 2,833,481.26| 1,305.54 | 0.32         | 1.31        | 0.99      |
| Feb-2026    | 20,394.20 | 2,684,229.30| 1,316.17 | -6.03        | -5.27       | 0.81      |
| Mar-2026    | 22,641.14 | 2,952,542.03| 1,304.06 | 11.02        | 10.00       | -0.92     |
| Apr-2026    | 22,346.80 | 2,902,988.05| 1,299.06 | -1.30        | -1.68       | -0.38     |
| May-2026    | 23,201.95 | 2,990,424.18| 1,288.87 | 3.83         | 3.01        | -0.78     |
| Jun-2026    | 22,716.08 | 2,892,138.66| 1,273.17 | -2.09        | -3.29       | -1.22     |

Mandate summary (fact_monthly aggregated across all banks):

| month_short | total_created | total_executed | execution_ratio_% |
|-------------|---------------|-----------------|--------------------|
| Jan-2026    | 8,460,409     | 433,922          | 5.13               |
| Feb-2026    | 1,041,011     | 240,714          | 23.12              |
| Mar-2026    | 348,130       | 270,582          | 77.72              |
| Apr-2026    | 109,705       | 41,941           | 38.23              |
| May-2026    | 485,104       | 120,019          | 24.74              |
| Jun-2026    | 6,235,786     | 253,851          | 4.07               |

(January and June spikes are seasonal IPO-mandate creation waves;
`execution_ratio_%` compares *same-month* created vs. executed and is
expected to be low in a spike month, since a mandate created near
month-end can only execute the following month.)

## 5. Metrics definitions

| Metric                    | Table              | Formula / meaning                                              |
|----------------------------|--------------------|------------------------------------------------------------------|
| `volume_mn`                | fact_monthly_agg, fact_daily | Transaction volume, millions of transactions                |
| `value_cr`                 | fact_monthly_agg, fact_daily | Transaction value, ₹ crores                                  |
| `avg_daily_volume_mn`      | fact_monthly_agg   | NPCI-reported average daily volume for the month, Mn            |
| `avg_daily_value_cr`       | fact_monthly_agg   | NPCI-reported average daily value for the month, ₹ Cr            |
| `ats_rs`                   | fact_monthly_agg, fact_daily | Average ticket size: `(value_cr * 10_000_000) / (volume_mn * 1_000_000)` = `value_cr * 10 / volume_mn`, in ₹ |
| `volume_mom_pct`           | fact_monthly_agg   | `(current - previous) / previous * 100` on `volume_mn`; `NULL` for the first month |
| `value_mom_pct`            | fact_monthly_agg   | Same MoM formula on `value_cr`                                   |
| `ats_mom_pct`               | fact_monthly_agg   | Same MoM formula on `ats_rs`                                     |
| `mandates_created`          | fact_monthly       | Count of UPI mandates created by a bank in a month (`total_volume` from the creation file) |
| `mandates_executed`         | fact_monthly       | Count of UPI mandates executed by a bank in a month (`total_volume` from the execution file) |
| `creation_approved_pct`     | fact_monthly       | `approved_pct` from the creation file (% of created mandates approved) |
| `execution_approved_pct`    | fact_monthly       | `approved_pct` from the execution file (% of executed mandates approved) |
| `execution_ratio_pct`       | fact_monthly       | `mandates_executed / mandates_created * 100`, same bank-month     |

## 6. LLM provider: Gemini (Google AI Studio)

The agent was switched from Groq to Gemini (`google-genai` SDK,
`gemini-2.5-flash`, Google AI Studio key). Everything downstream of the
LLM call is provider-agnostic — `src/tools.py` (SQL engine + tool
definitions), the Flask routes, and the web UI are unchanged. Only
`src/agent.py` and `requirements.txt` differ:

- `client.models.generate_content(..., config=GenerateContentConfig(tools=[...]))`
  replaces Groq's OpenAI-compatible `chat.completions.create`.
- `automatic_function_calling` is explicitly disabled — Gemini's SDK can
  run the tool loop for you, but that would hide each call from the
  `tools_used` audit trail the UI depends on. The loop in `chat()` stays
  manual, matching the original design.
- Gemini returns `FunctionCall.args` as an already-parsed dict (no
  `json.loads` needed, unlike Groq's JSON-string arguments).
- Gemini's chat history only recognizes `role="user"` and `role="model"`
  — there's no dedicated `"tool"` role. A function response is sent back
  as a `role="user"` `Content` containing a `function_response` part;
  `_trim_messages` treats only *plain-text* `role="user"` turns as
  cut points, so a `function_call`/`function_response` pair can never be
  split by trimming.
- Error handling maps `google.genai.errors.ClientError`/`ServerError`
  (with a numeric `.code`) onto the same four buckets as before: rate
  limit (429, retried with backoff), auth/permission (401/403), transient
  server error (5xx, retried), and network failure.

**Live-tested finding, not a code issue:** the Google AI Studio key
supplied for this project (`AIzaSyBTHf...`) authenticates successfully —
`ListModels` and `CountTokens` both return `200` — but every
`generateContent` call is rejected: `gemini-2.5-flash`, `gemini-2.5-pro`,
and `gemini-2.5-flash-lite` all return `403 PERMISSION_DENIED: "Your
project has been denied access. Please contact support."`, and
`gemini-2.0-flash` returns `429 RESOURCE_EXHAUSTED` with a free-tier
quota of `0` requests/day. This is an account/project-level restriction
on Google's side (likely billing not enabled, or the project pending
verification) — it reproduces identically via raw `curl`, independent of
this codebase or the sandbox's network proxy. Check
https://aistudio.google.com/ for the project's billing/status, or
generate a fresh key, then re-run; no code changes should be needed once
the key has real access. The tool-calling loop itself was validated as
far as possible without a live `generateContent` response: the exact
`TOOL_DEFINITIONS` schema was constructed as real `FunctionDeclaration`
objects and passed Gemini's own pydantic validation, and the retry/error
paths were exercised against the real `403`/`429` responses above.
