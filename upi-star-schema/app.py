"""Flask web server for the UPI Star Schema conversational agent."""
from typing import Dict

import pandas as pd
from flask import Flask, jsonify, render_template, request

from build_schemas import load_or_build_schemas
from src.agent import UPIAgent
from src.schemas import CUTOFF_DATE

app = Flask(__name__)

schemas: Dict = load_or_build_schemas()
agent = UPIAgent(schemas)

_RELATIONSHIPS = [
    {"from_table": "fact_monthly_agg", "from_col": "month_key", "to_table": "dim_month", "to_col": "month_key"},
    {"from_table": "fact_monthly", "from_col": "bank_key", "to_table": "dim_bank", "to_col": "bank_key"},
    {"from_table": "fact_monthly", "from_col": "month_key", "to_table": "dim_month", "to_col": "month_key"},
    {"from_table": "fact_daily", "from_col": "date_key", "to_table": "dim_date", "to_col": "date_key"},
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "").strip()
    if not message:
        return jsonify({"response": "Please enter a question.", "tools_used": []})
    result = agent.chat(message)
    return jsonify(result)


@app.route("/reset", methods=["POST"])
def reset():
    agent.reset()
    return jsonify({"status": "ok"})


@app.route("/lineage")
def lineage():
    monthly_dates = schemas["dim_month"]["month_date"]
    daily_dates = schemas["dim_date"]["full_date"]
    return jsonify({
        "source_files": 20,
        "tables": {name: len(df) for name, df in schemas.items()},
        "date_range": {
            "monthly": {
                "from": str(pd.Timestamp(monthly_dates.min()).date()),
                "to": str(pd.Timestamp(monthly_dates.max()).date()),
            },
            "daily": {
                "from": str(pd.Timestamp(daily_dates.min()).date()),
                "to": str(pd.Timestamp(daily_dates.max()).date()),
            },
        },
        "cutoff": str(CUTOFF_DATE),
    })


@app.route("/schema")
def schema():
    tables = []
    for name, df in schemas.items():
        tables.append({
            "name": name,
            "columns": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
            "rows": len(df),
        })
    return jsonify({"tables": tables, "relationships": _RELATIONSHIPS})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
