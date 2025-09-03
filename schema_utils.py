"""
schema_utils.py
Schema inference (for local CSV/JSON), schema comparison utilities,
and AI-driven explanations & SQL generation (via llm_utils.chat_llm).
Also optional Snowflake fetch helpers.
"""

import json
from typing import Dict, Tuple, List
import pandas as pd

from llm_utils import chat_llm

def infer_schema_from_df(df: pd.DataFrame) -> Dict[str, str]:
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if 'int' in dtype:
            schema[col] = 'INTEGER'
        elif 'float' in dtype:
            schema[col] = 'FLOAT'
        elif 'bool' in dtype:
            schema[col] = 'BOOLEAN'
        elif 'datetime' in dtype or 'date' in dtype:
            schema[col] = 'TIMESTAMP'
        else:
            schema[col] = 'VARCHAR'
    return schema

def compare_schemas(existing: Dict[str, str], new: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    new_cols = sorted(list(set(new) - set(existing)))
    missing_cols = sorted(list(set(existing) - set(new)))
    conflicts = []
    for col in set(existing).intersection(new):
        if existing[col] != new[col]:
            conflicts.append(col)
    conflicts.sort()
    return new_cols, missing_cols, conflicts

def _format_schema(s: Dict[str, str]) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in sorted(s.items())])

def explain_changes_with_ai(existing: Dict[str, str], new: Dict[str, str], table_name: str) -> str:
    prompt = f"""
You are an expert Snowflake engineer. Compare the two table schemas and explain the changes clearly and concisely.

Table: {table_name}

Existing schema:
{_format_schema(existing)}

New schema (candidate):
{_format_schema(new)}

Describe:
1) Added columns (with types)
2) Removed or missing columns and their impact
3) Data type conflicts and safe migration advice
4) Risks (NULLability, backfills, ingestion issues)
Give concise bullet points, and when appropriate, include example Snowflake SQL in a fenced code block.
"""
    return chat_llm(prompt)

def generate_sql_with_ai(existing: Dict[str, str], new: Dict[str, str], table_name: str) -> str:
    prompt = f"""
Generate Snowflake SQL to evolve table `{table_name}` from the existing schema to match the new schema.
Rules:
- Preserve data: avoid destructive operations by default.
- Add new columns as NULLable unless specified.
- For type conflicts, suggest safe ALTER ... USING CAST or create intermediate columns.
- Output only the SQL inside a fenced markdown code block.

Existing schema:
{_format_schema(existing)}

New schema:
{_format_schema(new)}
"""
    return chat_llm(prompt)

# Optional Snowflake helpers (if user provides credentials)
def make_snowflake_connection(cfg: dict):
    import snowflake.connector
    conn = snowflake.connector.connect(
        account=cfg.get("SNOWFLAKE_ACCOUNT"),
        user=cfg.get("SNOWFLAKE_USER"),
        password=cfg.get("SNOWFLAKE_PASSWORD"),
        warehouse=cfg.get("SNOWFLAKE_WAREHOUSE"),
        database=cfg.get("SNOWFLAKE_DATABASE"),
        schema=cfg.get("SNOWFLAKE_SCHEMA"),
    )
    return conn

def fetch_schema_from_snowflake(conn, database: str, schema: str, table: str) -> Dict[str, str]:
    q = f"""
    SELECT COLUMN_NAME, DATA_TYPE
    FROM {database}.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
    ORDER BY ORDINAL_POSITION
    """
    cur = conn.cursor()
    try:
        cur.execute(q, (schema.upper(), table.upper()))
        rows = cur.fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        cur.close()
