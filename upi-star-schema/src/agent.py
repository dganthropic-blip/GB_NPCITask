"""Groq-powered agentic loop. The LLM is a text-to-SQL analyst: it discovers
the schema, writes SQL, runs it via SchemaQueryEngine, and answers from the
results. No pre-built "get_trend"-style query functions exist on purpose —
the model composes SQL at runtime for whatever question it's asked.
"""
import json
import os
import time
from typing import Dict, List, Optional

from groq import Groq

try:
    from groq import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
except ImportError:  # pragma: no cover - defensive import for older groq versions
    APIConnectionError = ConnectionError
    APITimeoutError = Exception
    AuthenticationError = Exception
    RateLimitError = Exception

from src.tools import SchemaQueryEngine, TOOL_DEFINITIONS

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a UPI data analyst. You answer questions by writing SQL queries against a star schema. NEVER guess or use general knowledge — always query first, then answer from the results.

You have 3 tools: describe_tables (schema discovery), query_daily_stats (daily data shortcut), and run_sql (execute any SELECT query).

SCHEMA (DuckDB SQL):
  dim_month(month_key PK, month_date DATE, year INT, month_num INT, month_name TEXT, month_short TEXT, quarter TEXT, fiscal_year TEXT)
  dim_date(date_key PK, full_date DATE, day INT, month_num INT, year INT, month_name TEXT, month_short TEXT, day_name TEXT, day_of_week INT, is_weekend BOOL)
  dim_bank(bank_key PK, bank_name TEXT)
  fact_monthly_agg(month_key FK→dim_month, volume_mn FLOAT, value_cr FLOAT, avg_daily_volume_mn FLOAT, avg_daily_value_cr FLOAT, ats_rs FLOAT, volume_mom_pct FLOAT, value_mom_pct FLOAT, ats_mom_pct FLOAT)
  fact_monthly(bank_key FK→dim_bank, month_key FK→dim_month, mandates_created INT, mandates_executed INT, creation_approved_pct FLOAT, execution_approved_pct FLOAT, execution_ratio_pct FLOAT)
  fact_daily(date_key FK→dim_date, volume_mn FLOAT, value_cr FLOAT, ats_rs FLOAT)

DATA COVERAGE:
- fact_monthly_agg: Jun-2025 to Jun-2026 (13 months). Latest = Jun-2026.
- fact_daily: Jan 1 – Jun 30, 2026 (181 days).
- fact_monthly (mandates): Jan – Jun 2026 (6 months, 52 banks).
- Only COMPLETE months included — partial July excluded.

KEY CONSTRAINT: Transaction volume/value is AGGREGATE only (no bank split). Bank-level data exists ONLY for IPO mandate creation/execution. If asked "highest-value bank" for transactions, explain this limitation and offer mandate data instead.

UNITS: volume_mn = millions of transactions, value_cr = ₹ crores, ats_rs = ₹ per transaction.
ATS FORMULA: ats_rs = (value_cr × 10^7) / (volume_mn × 10^6). Pre-computed in ats_rs column.

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


class UPIAgent:
    def __init__(self, schemas: Dict, model: str = MODEL_NAME):
        api_key = os.environ.get("GROQ_API_KEY", "gsk_YOUR_KEY_HERE")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.engine = SchemaQueryEngine(schemas)
        self.messages: List[Dict] = []

    def reset(self) -> None:
        self.messages = []

    # -- message trimming -----------------------------------------------------

    def _trim_messages(self) -> None:
        """Keep the last ~20 messages, but only cut at a user-message boundary
        so a tool_call/tool_result pair never gets split."""
        if len(self.messages) <= _MAX_MESSAGES:
            return
        target = len(self.messages) - _MAX_MESSAGES
        cut_at = None
        for i in range(target, len(self.messages)):
            if self.messages[i].get("role") == "user":
                cut_at = i
                break
        if cut_at is not None:
            self.messages = self.messages[cut_at:]

    # -- tool format conversion -----------------------------------------------

    def _convert_tools(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in TOOL_DEFINITIONS
        ]

    # -- tool dispatch ---------------------------------------------------------

    def _execute_tool(self, name: str, args: Dict) -> str:
        if name == "describe_tables":
            return self.engine.describe_tables(args.get("table_name"))
        if name == "query_daily_stats":
            return self.engine.query_daily_stats(
                date_from=args.get("date_from"),
                date_to=args.get("date_to"),
                limit=args.get("limit", 31),
            )
        if name == "run_sql":
            return self.engine.run_sql(args.get("query", ""))
        return json.dumps({"error": f"Unknown tool '{name}'"})

    # -- Groq API call with retry -----------------------------------------------

    def _call_api(self, tools: List[Dict]):
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto",
                )
            except (RateLimitError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(1 if attempt == 0 else 2)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                if status is not None and status >= 500 and attempt < _MAX_RETRIES - 1:
                    last_exc = exc
                    time.sleep(1 if attempt == 0 else 2)
                    continue
                raise
        raise last_exc  # pragma: no cover - unreachable, loop always returns/raises

    # -- main agentic loop -------------------------------------------------------

    def chat(self, user_message: str) -> Dict:
        """Returns {"response": str, "tools_used": [{"tool": name, "args": {...}}]}"""
        tools_used: List[Dict] = []

        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

        self._trim_messages()
        self.messages.append({"role": "user", "content": user_message})

        tools = self._convert_tools()

        try:
            iterations = 0
            while True:
                iterations += 1
                if iterations > _MAX_ITERATIONS:
                    return {"response": "Unable to answer within allowed steps.", "tools_used": tools_used}

                response = self._call_api(tools)
                message = response.choices[0].message

                if message.tool_calls:
                    self.messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in message.tool_calls
                        ],
                    })

                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except json.JSONDecodeError:
                            args = {}

                        tools_used.append({"tool": tc.function.name, "args": args})
                        result = self._execute_tool(tc.function.name, args)

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                else:
                    self.messages.append({"role": "assistant", "content": message.content})
                    return {"response": message.content, "tools_used": tools_used}

        except RateLimitError:
            return {"response": "Rate limit exceeded, try again in a moment.", "tools_used": tools_used}
        except AuthenticationError:
            return {"response": "API key invalid or expired. Set GROQ_API_KEY to a valid key.", "tools_used": tools_used}
        except (APITimeoutError, APIConnectionError, ConnectionError):
            return {"response": "Unable to reach the Groq API. Check your network connection.", "tools_used": tools_used}
        except Exception as exc:  # noqa: BLE001
            return {"response": f"Unexpected error: {str(exc)[:200]}", "tools_used": tools_used}
