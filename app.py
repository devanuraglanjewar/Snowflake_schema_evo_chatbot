"""
Streamlit UI for AI-Powered Snowflake Schema Evolution Assistant
"""

import os
import json
import pandas as pd
import streamlit as st

from schema_utils import (
    infer_schema_from_df, compare_schemas,
    explain_changes_with_ai, generate_sql_with_ai,
    fetch_schema_from_snowflake, make_snowflake_connection,
)
from chatbot import answer_question
from faq import FAQS
from logging_utils import log_user_query

# -------------------------
# Config load (secrets > env)
# -------------------------
def _DEF(k, d=None):
    try:
        return st.secrets.get(k, os.getenv(k, d))
    except Exception:
        return os.getenv(k, d)

st.set_page_config(page_title="Snowflake Schema Evolution Assistant", layout="wide")
st.title("🚀 AI-Powered Snowflake Schema Evolution Assistant")

# -------------------------
# Session state for latest schema change context
# -------------------------
if "latest_context" not in st.session_state:
    st.session_state.latest_context = ""

# Tabs
chat_tab, schema_tab, faq_tab = st.tabs(["💬 Chatbot", "📊 Schema Analysis", "❓ FAQ"])

# -------------------------
# Chatbot Tab
# -------------------------
with chat_tab:
    st.subheader("Interactive Q&A")
    q = st.text_input("Ask anything about Snowflake schema evolution or your latest table changes:")
    if st.button("Get Answer", key="chat_btn") and q.strip():
        extra_ctx = st.session_state.latest_context or None
        with st.spinner("Thinking..."):
            ans = answer_question(q, extra_context=extra_ctx)
        st.markdown(ans)
        log_user_query("guest", q, ans)

# -------------------------
# Schema Analysis Tab
# -------------------------
with schema_tab:
    st.subheader("Analyze Table Schema Changes")

    mode = st.radio("Mode", ["Upload CSV/JSON (Local Demo)", "Snowflake Live (Optional)"])

    if mode == "Upload CSV/JSON (Local Demo)":
        existing = st.text_area(
            "Existing schema (JSON: { column: TYPE })",
            value=json.dumps({
                "first_name": "VARCHAR",
                "last_name": "VARCHAR",
                "email": "VARCHAR",
                "address": "VARCHAR",
                "city": "VARCHAR",
                "job_start_date": "TIMESTAMP"
            }, indent=2)
        )
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        table_name = st.text_input("Target table name", value="employee")
        if uploaded and existing:
            try:
                existing_schema = json.loads(existing)
            except Exception:
                st.error("Invalid existing schema JSON")
                st.stop()

            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_json(uploaded)

            new_schema = infer_schema_from_df(df)
            st.write("### Detected schema (from file)")
            st.json(new_schema)

            new_cols, missing_cols, conflicts = compare_schemas(existing_schema, new_schema)
            st.write("**Added columns:**", new_cols)
            st.write("**Missing columns:**", missing_cols)
            st.write("**Type conflicts:**", conflicts)

            if st.button("Ask AI to explain & generate SQL"):
                with st.spinner("AI analyzing changes..."):
                    explanation = explain_changes_with_ai(existing_schema, new_schema, table_name)
                    sql = generate_sql_with_ai(existing_schema, new_schema, table_name)
                st.markdown("### AI Explanation")
                st.markdown(explanation)
                st.code(sql, language="sql")

                # Save context for Chat tab follow-ups
                ctx = f"Existing: {existing_schema}\nNew: {new_schema}\nSQL: {sql}"
                st.session_state.latest_context = ctx

    else:
        # -------------------------
        # Snowflake Live Mode (with Database → Schema → Table dropdowns)
        # -------------------------
        st.info("🔗 Connect to Snowflake using secrets.toml ...")

        cfg = {k: _DEF(k, "") for k in [
            "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE"
        ]}

        missing = [k for k, v in cfg.items() if not v]
        if missing:
            st.error(f"❌ Missing Snowflake credentials in secrets.toml: {', '.join(missing)}")
            st.stop()

        try:
            conn = make_snowflake_connection(cfg)
        except Exception as e:
            st.error(f"❌ Could not connect to Snowflake: {e}")
            st.stop()

        # Step 1: Select Database
        db_list = []
        try:
            cur = conn.cursor()
            cur.execute("SHOW DATABASES")
            db_list = [row[1] for row in cur.fetchall()]
            cur.close()
        except Exception as e:
            st.error(f"Failed to fetch databases: {e}")

        db = st.selectbox("Select Database", db_list, index=0 if db_list else None)

        # Step 2: Select Schema
        schema_list = []
        if db:
            try:
                cur = conn.cursor()
                cur.execute(f"SHOW SCHEMAS IN DATABASE {db}")
                schema_list = [row[1] for row in cur.fetchall()]
                cur.close()
            except Exception as e:
                st.error(f"Failed to fetch schemas: {e}")

        sc = st.selectbox("Select Schema", schema_list, index=0 if schema_list else None)

        # Step 3: Select Table
        table_list = []
        if db and sc:
            try:
                cur = conn.cursor()
                cur.execute(f"SHOW TABLES IN SCHEMA {db}.{sc}")
                table_list = [row[1] for row in cur.fetchall()]
                cur.close()
            except Exception as e:
                st.error(f"Failed to fetch tables: {e}")

        table = st.selectbox("Select Table", table_list, index=0 if table_list else None)

        # Step 4: Fetch & Compare
        if st.button("Fetch live schema") and db and sc and table:
            try:
                with st.spinner(f"Fetching schema for {db}.{sc}.{table} ..."):
                    live_schema = fetch_schema_from_snowflake(conn, db, sc, table)
                st.success(f"✅ Live schema fetched for {db}.{sc}.{table}")
                st.json(live_schema)
            except Exception as e:
                st.error(f"❌ Snowflake error: {e}")
                live_schema = None

            prev = st.text_area("Paste previous schema snapshot (JSON) for diff", value="{}")
            if live_schema is not None and st.button("Analyze changes with AI"):
                try:
                    prev_schema = json.loads(prev)
                except Exception:
                    st.error("Invalid previous snapshot JSON")
                    st.stop()

                with st.spinner("AI analyzing live schema changes..."):
                    explanation = explain_changes_with_ai(prev_schema, live_schema, table)
                    sql = generate_sql_with_ai(prev_schema, live_schema, table)

                st.markdown("### AI Explanation")
                st.markdown(explanation)
                st.code(sql, language="sql")

                st.session_state.latest_context = f"Prev: {prev_schema}\nLive: {live_schema}\nSQL: {sql}"

# -------------------------
# FAQ Tab
# -------------------------
with faq_tab:
    st.subheader("Frequently Asked Questions")
    q = st.selectbox("Choose a question:", [""] + FAQS)
    if q:
        if st.button("Answer with AI", key="faq_btn"):
            ans = answer_question(q)
            st.markdown(ans)
