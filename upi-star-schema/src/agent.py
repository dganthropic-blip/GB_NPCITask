"""Gemini-powered agentic loop. The LLM is a text-to-SQL analyst: it discovers
the schema, writes SQL, runs it via SchemaQueryEngine, and answers from the
results. No pre-built "get_trend"-style query functions exist on purpose —
the model composes SQL at runtime for whatever question it's asked.
"""
import json
import os
import time
from typing import Dict, List, Optional

import httpx
import requests
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from src.tools import SchemaQueryEngine, TOOL_DEFINITIONS

MODEL_NAME = "gemini-flash-latest"

SYSTEM_PROMPT = """You are a UPI data analyst. You answer questions by writing SQL queries against a star schema. NEVER guess or use general knowledge — always query first, then answer from the results.

You have 3 tools: describe_tables (schema discovery), query_daily_stats (daily data shortcut), and run_sql (execute any SELECT query).

SCHEMA (DuckDB SQL):
  dim_month(month_key PK, month_date DATE, year INT, month_num INT, month_name TEXT, month_short TEXT, quarter TEXT, fiscal_year TEXT)
  dim_date(date_key PK, full_date DATE, day INT, month_num INT, year INT, month_name TEXT, month_short TEXT, day_name TEXT, day_of_week INT, is_weekend BOOL)
  dim_bank(bank_key PK, bank_name TEXT)
  fact_monthly_agg(month_key FK→dim_month, volume_mn FLOAT, value_cr FLOAT, avg_daily_volume_mn FLOAT, avg_daily_value_cr FLOAT, ats_rs FLOAT, volume_mom_pct FLOAT, value_mom_pct FLOAT, ats_mom_pct FLOAT)
  fact_monthly(bank_key FK→dim_bank, month_key FK→dim_month, mandates_created INT, mandates_executed INT, creation_approved_pct FLOAT, creation_bd_pct FLOAT, creation_td_pct FLOAT, execution_approved_pct FLOAT, execution_bd_pct FLOAT, execution_td_pct FLOAT, execution_ratio_pct FLOAT)
  fact_daily(date_key FK→dim_date, volume_mn FLOAT, value_cr FLOAT, ats_rs FLOAT)

DATA COVERAGE:
- fact_monthly_agg: Jun-2025 to Jun-2026 (13 months). Latest = Jun-2026.
- fact_daily: Jan 1 – Jun 30, 2026 (181 days).
- fact_monthly (mandates): Jan – Jun 2026 (6 months, 52 banks).
- Only COMPLETE months included — partial July excluded.

KEY CONSTRAINT: Transaction volume/value is AGGREGATE only (no bank split). Bank-level data exists ONLY for IPO mandate creation/execution. If asked "highest-value bank" for transactions, explain this limitation and offer mandate data instead.

UNITS: volume_mn = millions of transactions, value_cr = ₹ crores, ats_rs = ₹ per transaction.
ATS FORMULA: ats_rs = (value_cr × 10^7) / (volume_mn × 10^6). Pre-computed in ats_rs column.
MANDATE DECLINE BREAKDOWN: for each bank-month, approved_pct + bd_pct + td_pct ≈ 100%. bd_pct = Business Decline (customer/bank rejected the request — insufficient funds, limit exceeded, risk hold). td_pct = Technical Decline (infra failure — timeout, server error — unrelated to the transaction itself). A low approved_pct with high bd_pct points to a business/policy issue; high td_pct points to a technical/infra issue.

SQL PATTERNS:
- Latest month: SELECT ... FROM fact_monthly_agg f JOIN dim_month m ON f.month_key = m.month_key ORDER BY m.month_date DESC LIMIT 1
- N-month trend: ... ORDER BY m.month_date DESC LIMIT N
- MoM growth: use volume_mom_pct / value_mom_pct / ats_mom_pct columns (pre-computed)
- Top banks: SELECT b.bank_name, f.mandates_created FROM fact_monthly f JOIN dim_bank b ON f.bank_key = b.bank_key JOIN dim_month m ON f.month_key = m.month_key WHERE m.month_short = 'Jun-2026' ORDER BY f.mandates_created DESC LIMIT 10
- Mandate ratio: ROUND(SUM(mandates_executed) * 100.0 / NULLIF(SUM(mandates_created), 0), 2)
- Month filter: WHERE m.month_short = 'Jun-2026' or WHERE m.month_date >= '2026-01-01'

ANSWERING RULES:
1. Write SQL, execute via run_sql, then answer from results.
2. Default "latest month" = Jun-2026. Default trend = last 6 months.
3. Format numbers with commas. Use ₹ for money. Use tables for multi-row data.
4. Be concise — lead with the answer, add context after.
5. If a question cannot be answered from the data, say so and explain why.
"""

_MAX_MESSAGES = 20
_MAX_ITERATIONS = 10
_MAX_RETRIES = 3

_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionError,
    TimeoutError,
)


class UPIAgent:
    def __init__(self, schemas: Dict, model: str = MODEL_NAME):
        api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.engine = SchemaQueryEngine(schemas)
        self.contents: List[types.Content] = []

        function_declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["input_schema"],
            )
            for t in TOOL_DEFINITIONS
        ]
        self._tool = types.Tool(function_declarations=function_declarations)
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[self._tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def reset(self) -> None:
        self.contents = []

    # -- message trimming -----------------------------------------------------

    @staticmethod
    def _is_user_question(content: types.Content) -> bool:
        """True for a real new user question, not a function-response turn."""
        if content.role != "user":
            return False
        parts = content.parts or []
        return all(getattr(p, "function_response", None) is None for p in parts)

    def _trim_messages(self) -> None:
        """Keep the last ~20 turns, but only cut at a user-question boundary
        so a function_call/function_response pair never gets split."""
        if len(self.contents) <= _MAX_MESSAGES:
            return
        target = len(self.contents) - _MAX_MESSAGES
        cut_at = None
        for i in range(target, len(self.contents)):
            if self._is_user_question(self.contents[i]):
                cut_at = i
                break
        if cut_at is not None:
            self.contents = self.contents[cut_at:]

    # -- tool dispatch ---------------------------------------------------------

    def _execute_tool(self, name: str, args: Dict) -> Dict:
        if name == "describe_tables":
            result = self.engine.describe_tables(args.get("table_name"))
        elif name == "query_daily_stats":
            result = self.engine.query_daily_stats(
                date_from=args.get("date_from"),
                date_to=args.get("date_to"),
                limit=args.get("limit", 31),
            )
        elif name == "run_sql":
            result = self.engine.run_sql(args.get("query", ""))
        else:
            result = json.dumps({"error": f"Unknown tool '{name}'"})

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            parsed = result
        return {"result": parsed}

    # -- Gemini API call with retry ---------------------------------------------

    def _call_api(self):
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=self.contents,
                    config=self._config,
                )
            except ServerError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(1 if attempt == 0 else 2)
                    continue
                raise
            except ClientError as exc:
                if exc.code == 429 and attempt < _MAX_RETRIES - 1:
                    last_exc = exc
                    time.sleep(1 if attempt == 0 else 2)
                    continue
                raise
        raise last_exc  # pragma: no cover - unreachable, loop always returns/raises

    # -- main agentic loop -------------------------------------------------------

    def chat(self, user_message: str) -> Dict:
        """Returns {"response": str, "tools_used": [{"tool": name, "args": {...}}]}"""
        tools_used: List[Dict] = []

        self._trim_messages()
        self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

        try:
            iterations = 0
            while True:
                iterations += 1
                if iterations > _MAX_ITERATIONS:
                    return {"response": "Unable to answer within allowed steps.", "tools_used": tools_used}

                response = self._call_api()
                function_calls = response.function_calls

                if function_calls:
                    self.contents.append(response.candidates[0].content)

                    response_parts = []
                    for fc in function_calls:
                        args = dict(fc.args) if fc.args else {}
                        tools_used.append({"tool": fc.name, "args": args})
                        tool_result = self._execute_tool(fc.name, args)
                        response_parts.append(
                            types.Part.from_function_response(name=fc.name, response=tool_result)
                        )

                    self.contents.append(types.Content(role="user", parts=response_parts))
                else:
                    answer = response.text or ""
                    self.contents.append(types.Content(role="model", parts=[types.Part.from_text(text=answer)]))
                    return {"response": answer, "tools_used": tools_used}

        except ClientError as exc:
            if exc.code == 429:
                return {"response": "Rate limit exceeded, try again in a moment.", "tools_used": tools_used}
            if exc.code in (401, 403) or "api key" in str(exc).lower():
                return {"response": "API key invalid or expired (or access denied). Set GEMINI_API_KEY to a valid key.", "tools_used": tools_used}
            return {"response": f"Unexpected error: {str(exc)[:200]}", "tools_used": tools_used}
        except ServerError:
            return {"response": "Unable to reach the Gemini API — server error, try again shortly.", "tools_used": tools_used}
        except _NETWORK_ERRORS:
            return {"response": "Unable to reach the Gemini API. Check your network connection.", "tools_used": tools_used}
        except Exception as exc:  # noqa: BLE001
            return {"response": f"Unexpected error: {str(exc)[:200]}", "tools_used": tools_used}
