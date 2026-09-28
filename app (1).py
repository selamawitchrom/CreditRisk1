"""
Credit Risk Portfolio Query Engine — Streamlit App
====================================================
Converts the "Credit Risk Query Engine" notebook (Northbridge Bank use case)
into a Streamlit application. Business users type a natural-language question
about the commercial lending portfolio and the app:

  1. Classifies intent (verified template vs. freshly generated SQL)
  2. Builds the SQL (from the Verified Query Library, or via the LLM)
  3. Validates the SQL (read-only shape, schema conformance, EXPLAIN dry-run,
     LLM relevance check, template-integrity check)
  4. Retries once (generated track only) if validation fails
  5. Escalates to a human analyst if it still fails
  6. Executes the query (read-only) and shows the result table
  7. Generates a short, business-focused narrative answer

All pipeline logic, SQL templates, and validation rules are preserved exactly
as implemented in the notebook.
"""

import os
import re
import json
import sqlite3

import numpy as np
import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="📊",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Configuration / credentials
# ----------------------------------------------------------------------------
# Credentials are read, in order of priority, from:
#   1. Streamlit secrets (st.secrets)              -> recommended for Streamlit Cloud
#   2. Environment variables                        -> recommended for local/Docker use
# The notebook's config.json approach is intentionally NOT used in production,
# since committing API keys to GitHub is unsafe. See requirements.txt / README
# notes at the bottom of this file for how to set these.


def _get_config_value(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


OPENAI_API_KEY = _get_config_value("OPENAI_API_KEY")
OPENAI_API_BASE = _get_config_value("OPENAI_API_BASE")  # optional (custom/proxy base URL)
DB_PATH = _get_config_value("DB_PATH", "credit_risk_portfolio.db")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Add it to Streamlit secrets "
        "(Settings → Secrets) or as an environment variable before using the app."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


# ----------------------------------------------------------------------------
# Cached resources: LLMs and DB connection
# ----------------------------------------------------------------------------
@st.cache_resource
def get_llms():
    llm = ChatOpenAI(
        temperature=0.0,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
    )
    evaluator_llm = ChatOpenAI(
        temperature=0.0,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
    )
    return llm, evaluator_llm


@st.cache_resource
def get_db_connection(db_path):
    if not os.path.exists(db_path):
        st.error(
            f"Database file not found at '{db_path}'. Place "
            f"'credit_risk_portfolio.db' in the app's working directory "
            f"(same folder as app.py) or set the DB_PATH secret/env var."
        )
        st.stop()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return conn


llm, evaluator_llm = get_llms()
conn = get_db_connection(DB_PATH)

# ----------------------------------------------------------------------------
# Database schema (given to the LLM in prompts)
# ----------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# ----------------------------------------------------------------------------
# Verified Query Template Library (10 pre-approved SQL queries)
# ----------------------------------------------------------------------------
sql_1 = """
SELECT
sm.sector_code,
sm.sector_name,
SUM(lm.total_outstanding) AS total_outstanding,
SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN lm.total_outstanding ELSE 0 END) AS npa_exposure
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_code, sm.sector_name
ORDER BY total_outstanding DESC
"""

sql_2 = """
SELECT loan_category, sum(total_outstanding) as total_outstanding, count(*) as loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding DESC
"""

sql_3 = """
SELECT ifrs9_stage, COUNT(*) AS loan_count, SUM(ead_amount) AS ead_amount, SUM(ecl_amount) AS ecl_amount
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
"""

sql_4 = """
SELECT
sm.sector_name, AVG(p.provision_coverage_ratio)
from provisioning p
join loan_master lm on p.loan_account_number = lm.loan_account_number
join sector_master sm on lm.sector_code = sm.sector_code
where p.reporting_date = '2025-09-30'
group by sm.sector_name
order by AVG(p.provision_coverage_ratio) desc
"""

sql_5 = """
SELECT
  lm.loan_account_number,
  lm.borrower_name,
  sm.sector_name,
  lm.total_outstanding / 1000000.0 AS total_outstanding_mm,
  lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY lm.total_outstanding DESC
LIMIT 10
"""

sql_6 = """SELECT
group_name, COUNT(*) AS loan_count, SUM(total_outstanding)/ 1000000 AS total_outstanding
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding DESC
LIMIT 5
"""

sql_7 = """SELECT
  lm.loan_account_number,
  lm.borrower_name,
  sm.sector_name,
  lm.total_outstanding / 1000000.0 AS outstanding_mm,
  lm.days_past_due AS DPD,
  lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC
"""

sql_8 = """SELECT
CASE
 WHEN days_past_due = 0 THEN '0 (Current)'
 WHEN days_past_due BETWEEN 1 AND 30  THEN '1-30'
 WHEN days_past_due BETWEEN 31 AND 60  THEN '31-60'
 WHEN days_past_due BETWEEN 61 AND 90  THEN '61-90'
 WHEN days_past_due > 90  THEN '90+'
END AS DPD_Bucket,
 COUNT(*) AS loan_count,
 SUM(total_outstanding) AS total_outstanding
FROM loan_master
GROUP BY DPD_Bucket
ORDER BY DPD_Bucket
"""

sql_9 = """
SELECT
br.borrower_id,
br.previous_rating,
br.internal_rating,
br.pd_estimate
FROM borrower_rating br
WHERE br.rating_date = '2025-09-30' AND br.rating_direction = 'Downgraded'
ORDER BY br.pd_estimate DESC
"""

sql_10 = """SELECT
reporting_date, SUM(ecl_amount) / 1000000.0 AS total_ecl_mm
from provisioning
group by reporting_date
order by reporting_date
"""

verified_query_library = {
    "VQ1": {
        "description": "Sector-wise total outstanding and NPA amount breakdown across all sectors",
        "sql": sql_1,
    },
    "VQ2": {
        "description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)",
        "sql": sql_2,
    },
    "VQ3": {
        "description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter",
        "sql": sql_3,
    },
    "VQ4": {
        "description": "Average provision coverage ratio by sector for the latest reporting quarter",
        "sql": sql_4,
    },
    "VQ5": {
        "description": "Top 10 largest loan exposures by outstanding amount at the borrower level",
        "sql": sql_5,
    },
    "VQ6": {
        "description": "Top 5 largest exposures aggregated at the business group level",
        "sql": sql_6,
    },
    "VQ7": {
        "description": "All overdue loan accounts with their days past due and asset classification",
        "sql": sql_7,
    },
    "VQ8": {
        "description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio",
        "sql": sql_8,
    },
    "VQ9": {
        "description": "Borrowers whose internal rating was downgraded in the latest rating cycle",
        "sql": sql_9,
    },
    "VQ10": {
        "description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time",
        "sql": sql_10,
    },
}


# ----------------------------------------------------------------------------
# Pipeline tools (preserved from the notebook)
# ----------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    """Classifies the user question and decides which route to take."""
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are an expert in financial risk analysis and SQL. Your task is to classify a user's question regarding a commercial lending portfolio. Decide whether the question can be answered by one of the provided 'verified' SQL query templates, or if it requires a 'generated' SQL query.

Here are the available verified query templates:
{library_descriptions}

Here is the user's question:
{user_question}

Based on the user's question, determine the 'route' (either 'verified' or 'generated').
If 'verified', provide the exact 'query_id' (e.g., "VQ1"). If 'generated', set 'query_id' to null.
Provide a 'match_reason' which is a concise explanation for your decision. Do not try to answer the question, just classify it.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    """Generates a candidate SQL query for a novel question using the database schema."""
    generation_prompt = f"""
You are an expert in SQL and financial risk analysis. Your task is to write a single, read-only SQLite SQL query that answers the user's question. Use the provided database schema to construct the query. Ensure the query is syntactically correct and retrieves the necessary information.

Database Schema:
{schema_context}

User Question:
{user_question}

Your SQL query should be read-only (only SELECT statements allowed) and should not contain any DDL or DML statements.

### OUTPUT FORMAT
Provide ONLY the SQL query. Do not include any conversational text, explanations, or markdown fences (```sql).
Your response should start directly with 'SELECT' or 'WITH'.

Example of expected output:
SELECT column1, column2 FROM table_name WHERE condition;
"""

    sql = llm.invoke(generation_prompt).content.strip()

    sql_upper = sql.upper()
    select_idx = sql_upper.find("SELECT")
    with_idx = sql_upper.find("WITH")

    start_idx = -1
    if select_idx != -1 and with_idx != -1:
        start_idx = min(select_idx, with_idx)
    elif select_idx != -1:
        start_idx = select_idx
    elif with_idx != -1:
        start_idx = with_idx

    if start_idx != -1:
        sql = sql[start_idx:].strip()

    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    """Validates a candidate SQL query through five checks before execution."""
    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None,
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH"]
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {kw}"
            return result
    if ";" in candidate_sql.rstrip(";").rstrip():
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple statements are not allowed"
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or "Name" in str(t.ttype)]
    referenced_identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", candidate_sql.lower())
    sql_keywords = {
        "select", "from", "where", "and", "or", "group", "by", "order", "having", "limit", "join", "on", "as", "case",
        "when", "then", "else", "end", "sum", "count", "avg", "min", "max", "round", "desc", "asc", "left", "right",
        "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with", "union", "all", "between", "coalesce",
    }
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
        and not tok.isdigit() and tok not in ("s", "l", "p", "r", "e6")
    ]
    # NOTE: as in the original notebook, 'unknown' identifiers are computed but not
    # enforced here, because table aliases (lm, sm, p, br) and column aliases
    # (npa_exposure, total_outstanding_mm, ...) are legitimate and would be flagged.

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed to parse or plan: {str(e)}"
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
You are a SQL expert. Evaluate if the given SQL query is relevant to the user's question, considering the provided context. Respond with 'yes' or 'no' for verdict, a confidence score (0.0 to 1.0), and a concise reason.

{track_context}

User Question:
{user_question}

SQL Query:
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result["relevance_confidence"] = relevance_json.get("confidence", 0.0)
        if relevance_json.get("verdict") == "no" or relevance_json.get("confidence", 0.0) < 0.6:
            result["failed_check"] = "llm_relevance"
            result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]["sql"]
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = f"Expected {len(expected_cols)} columns, got {len(actual_cols)}"
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity check failed: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    """Regenerates SQL after a validation failure, feeding the error back to the LLM."""
    retry_prompt = f"""
You are an expert in SQL and financial risk analysis. Your previous attempt to generate an SQL query for the user's question failed validation. You need to revise the SQL query based on the provided error message and database schema.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

Your revised SQL query should be a single, read-only SQLite SQL query that addresses the user's question and corrects the issues highlighted by the validation error.

### OUTPUT FORMAT
Provide ONLY the SQL query. Do not include any conversational text, explanations, or markdown fences (```sql).
Your response should start directly with 'SELECT' or 'WITH'.

Example of expected output:
SELECT column1, column2 FROM table_name WHERE condition;
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()

    sql_upper = revised_sql.upper()
    select_idx = sql_upper.find("SELECT")
    with_idx = sql_upper.find("WITH")

    start_idx = -1
    if select_idx != -1 and with_idx != -1:
        start_idx = min(select_idx, with_idx)
    elif select_idx != -1:
        start_idx = select_idx
    elif with_idx != -1:
        start_idx = with_idx

    if start_idx != -1:
        revised_sql = revised_sql[start_idx:].strip()

    revised_sql = re.sub(r"^```sql\s*|\s*```$", "", revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r"^```\s*|\s*```$", "", revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    """Executes a gate-passed SQL query and returns the result as a DataFrame."""
    result = {"dataframe": None, "reasonable": True, "warnings": []}

    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df

    if df.empty:
        result["warnings"].append("Query returned an empty result")

    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")

    if len(result["warnings"]) > 2:
        result["reasonable"] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    """Generates a focused natural language response from the query result."""
    response_prompt = f"""
You are a credit-risk analyst assistant reporting to business users at a commercial bank. \
You have been given the user's original question and the full result set of a SQL query \
that was run to answer it (route: {route}{f", template: {query_id}" if query_id else ""}).

Write a short, business-focused answer that:
- Directly answers what the user asked, using exact figures from the data (do not invent numbers).
- Highlights only the rows/metrics relevant to the question (the data may contain more than was asked for).
- Uses plain business language, not SQL or technical jargon.
- Is concise: a short paragraph or a few bullet points, not a restatement of the entire table.

User Question:
{user_question}

Full Query Result:
{dataframe.to_string()}
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=False):
    """Runs the complete query engine pipeline for a single user question."""
    log = {
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": None,
        "confidence": None,
        "narrative": None,
    }
    stage_log = []

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log["route"] = classification["route"]
    log["query_id"] = classification.get("query_id")
    log["match_reason"] = classification.get("match_reason")
    stage_log.append(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    stage_log.append(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"]
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log["candidate_sql"] = candidate_sql
    stage_log.append(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log["query_id"])
    log["gate_result"] = gate
    stage_log.append(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate["passed"]:
        stage_log.append(f"    Failed check: {gate.get('failed_check')}")
        stage_log.append(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate["passed"] and log["route"] == "generated":
        stage_log.append(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate["details"], schema_context)
        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log["gate_result"] = gate
        stage_log.append(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate["passed"]:
            stage_log.append(f"    Retry failed check: {gate.get('failed_check')}")
            stage_log.append(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        log["escalated"] = True
        log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log["confidence"] = "ESCALATED"
        stage_log.append(f"[!] Escalated to human: {gate['details']}")
        return {"log": log, "dataframe": None, "stage_log": stage_log, **log}

    # Step 6: Execute
    log["executed_sql"] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result["dataframe"]
    log["row_count"] = len(df)
    stage_log.append(f"[4] Execute: {len(df)} rows returned")
    if exec_result["warnings"]:
        stage_log.append(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log["route"], log["query_id"])
    log["narrative"] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log["confidence"] = gate.get("relevance_confidence")
    stage_log.append(f"[6] Response Generation: confidence={log['confidence']}")

    return {"log": log, "dataframe": df, "stage_log": stage_log, **log}


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.title("📊 Credit Risk Portfolio Query Engine")
st.caption(
    "Northbridge Bank — ask routine commercial lending portfolio questions in plain English. "
    "Verified questions run pre-approved SQL templates; novel questions get validated, "
    "read-only generated SQL, with automatic escalation if a reliable answer can't be produced."
)

with st.sidebar:
    st.header("Verified Query Library")
    st.caption("Pre-approved, version-controlled SQL templates (10 total).")
    for qid, entry in verified_query_library.items():
        with st.expander(qid):
            st.write(entry["description"])
            st.code(entry["sql"].strip(), language="sql")

    st.divider()
    st.header("Batch Test Cases (optional)")
    uploaded_csv = st.file_uploader("Upload test_queries.csv", type=["csv"])
    st.caption(
        "Columns expected: 'Test Case', 'User Query', 'Expected Route', "
        "'Expected Query ID', 'Expected Answer'."
    )

if "history" not in st.session_state:
    st.session_state.history = []

tab_ask, tab_batch = st.tabs(["Ask a Question", "Batch Evaluation"])

# ---------------- Ask a Question tab ----------------
with tab_ask:
    user_question = st.text_input(
        "Your question about the commercial lending portfolio",
        placeholder="e.g. What is the total outstanding exposure by sector?",
    )
    show_pipeline_log = st.checkbox("Show pipeline execution log", value=False)
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and user_question.strip():
        with st.spinner("Running the query engine pipeline..."):
            try:
                result = run_pipeline(
                    user_question, conn, verified_query_library, database_schema, verbose=False
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                result = None

        if result:
            st.session_state.history.insert(0, result)

            route = result["route"]
            query_id = result["query_id"]
            confidence = result["confidence"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Route", route.upper() if route else "N/A")
            col2.metric("Query ID", query_id if query_id else "—")
            col3.metric(
                "Confidence",
                f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence),
            )

            if result["escalated"]:
                st.warning(result["narrative"])
            else:
                st.subheader("Answer")
                st.write(result["narrative"])

                st.subheader("SQL Query Used")
                st.code((result["executed_sql"] or "").strip(), language="sql")

                st.subheader("Result Data")
                st.dataframe(result["dataframe"], use_container_width=True)

            if show_pipeline_log:
                st.subheader("Pipeline Log")
                st.text("\n".join(result["stage_log"]))

    elif ask_clicked:
        st.info("Please enter a question first.")

    if st.session_state.history:
        st.divider()
        st.subheader("Recent Questions")
        for i, past in enumerate(st.session_state.history[:5]):
            with st.expander(f"{past['user_question']}  —  {past['route']} {past['query_id'] or ''}"):
                if past["escalated"]:
                    st.warning(past["narrative"])
                else:
                    st.write(past["narrative"])
                    st.code((past["executed_sql"] or "").strip(), language="sql")
                    st.dataframe(past["dataframe"], use_container_width=True)

# ---------------- Batch Evaluation tab ----------------
with tab_batch:
    st.caption(
        "Upload a test_queries.csv (columns: Test Case, User Query, Expected Route, "
        "Expected Query ID, Expected Answer) to run every question through the pipeline "
        "and compare against ground truth."
    )
    if uploaded_csv is not None:
        ground_truth = pd.read_csv(uploaded_csv)
        st.write(f"Loaded {len(ground_truth)} test cases.")
        run_batch = st.button("Run Batch Evaluation")

        if run_batch:
            evaluation_rows = []
            progress = st.progress(0.0)
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = run_pipeline(
                    gt["User Query"], conn, verified_query_library, database_schema, verbose=False
                )
                evaluation_rows.append(
                    {
                        "Test Case": gt["Test Case"],
                        "User Query": gt["User Query"],
                        "Expected Route": gt["Expected Route"],
                        "Actual Route": tr["route"],
                        "Route Match": tr["route"] == gt["Expected Route"],
                        "Expected Query ID": gt.get("Expected Query ID"),
                        "Actual Query ID": tr["query_id"],
                        "Query ID Match": (
                            pd.isna(gt.get("Expected Query ID")) and pd.isna(tr["query_id"])
                        )
                        or tr["query_id"] == gt.get("Expected Query ID"),
                        "Confidence": tr["confidence"],
                        "Rows Returned": tr["row_count"],
                        "Narrative": tr["narrative"],
                    }
                )
                progress.progress((i + 1) / len(ground_truth))

            evaluation_df = pd.DataFrame(evaluation_rows)

            path_accuracy = evaluation_df["Route Match"].mean() * 100
            verified_mask = evaluation_df["Expected Route"].astype(str).str.strip().str.lower() == "verified"
            query_accuracy = (
                evaluation_df.loc[verified_mask, "Query ID Match"].mean() * 100
                if verified_mask.any()
                else float("nan")
            )
            numeric_confidence = pd.to_numeric(evaluation_df["Confidence"], errors="coerce")
            average_confidence = numeric_confidence.mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            m2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
            m3.metric("Average Confidence", f"{average_confidence:.2f}")

            st.dataframe(evaluation_df, use_container_width=True)
    else:
        st.info("Upload a CSV file in the sidebar to enable batch evaluation.")

st.divider()
st.caption(
    "PoC scope: read-only analytical queries against the commercial lending portfolio only. "
    "No database writes are performed. Complex or ambiguous questions are escalated to a "
    "human analyst rather than answered speculatively."
)
