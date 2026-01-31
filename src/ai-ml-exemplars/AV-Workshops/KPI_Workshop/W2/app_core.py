from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from sql_tool import run_sql_readonly
from schema_utils import get_sqlite_schema, format_schema_for_prompt

DB_PATH = "data/ecommerce_analytics.sqlite"
CHROMA_DIR = "chroma_kpi"

# Chat model used for classification, SQL generation, and final answers.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Load persisted vector DB for KPI definitions.
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# Preload schema so the SQL prompt is grounded in real tables/columns.
SCHEMA_DICT = get_sqlite_schema(DB_PATH)
SCHEMA_TEXT = format_schema_for_prompt(SCHEMA_DICT)

# Lightweight classifier to decide if we need to query data.
classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the user request. Output ONLY one label: DEFINITIONS or NEEDS_DATA."),
    ("user", "{question}")
])

# SQL generation prompt with safety and schema constraints.
sql_prompt = ChatPromptTemplate.from_messages([
    ("system",
    "You are a careful analytics SQL assistant for SQLite.\n"
     "Generate EXACTLY ONE safe read-only SQL query.\n\n"

     "SQL DIALECT:\n"
     "- Use SQLite dialect only.\n"
     "- Do NOT use FULL OUTER JOIN or RIGHT JOIN. If needed, use LEFT JOIN + UNION workaround.\n\n"

     "SAFETY RULES:\n"
     "- Only SELECT/WITH queries. No INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA.\n"
     "- Return ONLY SQL. No commentary, no markdown, no code fences.\n\n"

     "SCHEMA (USE ONLY THESE TABLES/COLUMNS — DO NOT INVENT ANYTHING):\n"
     "{schema}\n\n"

     "COLUMN RULES (CRITICAL):\n"
     "- Use ONLY columns present in the schema above.\n"
     "- ALWAYS fully qualify columns as table.column (do not use unqualified columns).\n"
     "- Do not invent timestamp columns. Use the actual timestamp columns from the schema.\n\n"

     "TIME GROUPING:\n"
     "- If weekly metrics are requested, group by: strftime('%Y-%W', <timestamp_column>)\n"
     "- Use the correct table timestamp column depending on the metric:\n"
     "  * sessions: sessions.session_start_ts\n"
     "  * orders: orders.order_ts\n"
     "  * events: events.event_ts\n"
    #  "Do not use aggregate functions other than COUNT, SUM, AVG, MIN, MAX, STDDEV, VAR_POP, VAR_SAMPLE, PERCENTILE_CONT, PERCENTILE_DISC."
    #  "Do not use window functions other than ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG, LEAD, FIRST_VALUE, LAST_VALUE, CUME_DIST, PERCENT_RANK, NTH_VALUE, PERCENTILE_CONT, PERCENTILE_DISC."
    #  "Do not use subqueries in the SELECT clause."
    #  "Do not use subqueries in the WHERE clause."
    "Do not use subqueries in the ORDER BY clause."
    "Do not use subqueries in the LIMIT clause."

    # Few shots for better results
    """EXAMPLES:
    Q: 'Weekly sessions and orders'
    SQL:
    WITH s AS (
    SELECT strftime('%Y-%W', sessions.session_start_ts) AS year_week, COUNT(*) AS sessions
    FROM sessions
    GROUP BY 1
    ),
    o AS (
    SELECT strftime('%Y-%W', orders.order_ts) AS year_week, COUNT(*) AS orders
    FROM orders
    GROUP BY 1
    )
    SELECT s.year_week, s.sessions, COALESCE(o.orders,0) AS orders
    FROM s LEFT JOIN o ON o.year_week = s.year_week
    ORDER BY s.year_week;
    """
     ),
    ("user", "{question}")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a KPI copilot for business analytics.\n"
     "Use ONLY the provided KPI definitions/rules as the source of truth.\n"
     "When you state a KPI definition or rule, cite the source.\n"
     "If numbers are provided, ground your narrative in them.\n"
     "Separate FACTS (from data) vs HYPOTHESES (possible drivers).\n"),
    ("user",
     "Question: {question}\n\n"
     "KPI Docs (with citations):\n{docs}\n\n"
     "Data (if any):\n{data}\n\n"
     "Write the answer + (if applicable) a short weekly narrative.")
])

def format_docs(docs):
    # Flatten retrieved docs with their sources for the answer prompt.
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        out.append(f"- SOURCE: {src}\n  {d.page_content}")
    return "\n\n".join(out)

def handle_question(question: str) -> Dict[str, Any]:
    # Decide whether the question needs a data query.
    label = (llm.invoke(classifier_prompt.format_messages(question=question)).content or "").strip()

    # Retrieve KPI definition snippets for grounding.
    docs = retriever.invoke(question)
    docs_text = format_docs(docs)

    data = ""
    sql = ""
    if label == "NEEDS_DATA":
        # Generate and execute a read-only SQL query.
        sql = llm.invoke(sql_prompt.format_messages(question=question, schema=SCHEMA_TEXT)).content.strip()
        rows = run_sql_readonly(DB_PATH, sql)
        data = rows[:50]  # keep responses compact

    # Compose the final response using docs and (optional) data.
    response = llm.invoke(answer_prompt.format_messages(
        question=question, docs=docs_text, data=str(data)
    )).content

    return {"label": label, "sql": sql, "response": response}
