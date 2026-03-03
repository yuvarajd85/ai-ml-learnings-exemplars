'''
Created on 3/3/26 at 1:16 AM
By yuvarajdurairaj
Module Name LCRagGraphDB
'''


import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from neo4j import GraphDatabase

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# Config
# -----------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

TICKER = "AAPL"
ALLOWED_ITEMS = {"1A", "7"}  # Risk Factors + MD&A
VECTOR_K = 30               # retrieve more, then filter/rerank
FINAL_CONTEXT_K = 10         # final chunks to send to LLM


# -----------------------------
# Simple entity extraction (demo)
# Replace with real NER/LLM extraction later.
# -----------------------------
ENTITY_PATTERNS = [
    r"\bChina\b", r"\bPRC\b", r"\bsupply chain\b", r"\btariff(s)?\b",
    r"\bgeopolitical\b", r"\binflation\b", r"\bmanufactur(ing|ers?)\b",
]

def extract_entities(text: str) -> List[str]:
    ents = set()
    lower = text.lower()
    for pat in ENTITY_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            ents.add(re.sub(r"\\b", "", pat).strip("()?:\\").lower())
    # normalize a bit
    norm = []
    for e in ents:
        e = e.replace("(s)?", "s").replace("|", " ").strip()
        norm.append(e)
    return sorted(set(norm))


# -----------------------------
# Neo4j helpers
# -----------------------------
def neo4j_run(driver, cypher: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    with driver.session() as sess:
        res = sess.run(cypher, params or {})
        return [r.data() for r in res]


def neo4j_init_schema(driver) -> None:
    # Minimal constraints / indexes (safe if rerun)
    neo4j_run(driver, "CREATE CONSTRAINT company_key IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE")
    neo4j_run(driver, "CREATE CONSTRAINT filing_key IF NOT EXISTS FOR (f:Filing) REQUIRE f.accession IS UNIQUE")
    neo4j_run(driver, "CREATE CONSTRAINT chunk_key IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE")
    neo4j_run(driver, "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")


def neo4j_upsert_demo_data(driver, docs: List[Document]) -> None:
    """
    docs must include metadata:
      ticker, accession, form, filed_date, fiscal_year, item, chunk_id
    """
    # Create company + filings + chunks + edges
    for d in docs:
        md = d.metadata
        neo4j_run(driver, """
            MERGE (c:Company {ticker:$ticker})
            MERGE (f:Filing {accession:$accession})
              SET f.form=$form, f.filed_date=$filed_date, f.fiscal_year=$fiscal_year
            MERGE (c)-[:FILED]->(f)
            MERGE (ch:Chunk {chunk_id:$chunk_id})
              SET ch.text=$text, ch.item=$item, ch.ticker=$ticker, ch.accession=$accession, ch.fiscal_year=$fiscal_year
            MERGE (f)-[:HAS_CHUNK]->(ch)
        """, {
            "ticker": md["ticker"],
            "accession": md["accession"],
            "form": md["form"],
            "filed_date": md["filed_date"],
            "fiscal_year": md["fiscal_year"],
            "chunk_id": md["chunk_id"],
            "item": md["item"],
            "text": d.page_content,
        })

        # Entities + mentions edges
        ents = extract_entities(d.page_content)
        for e in ents:
            neo4j_run(driver, """
                MERGE (en:Entity {name:$name})
                MATCH (ch:Chunk {chunk_id:$chunk_id})
                MERGE (ch)-[:MENTIONS]->(en)
            """, {"name": e, "chunk_id": md["chunk_id"]})


def neo4j_latest_two_filings(driver, ticker: str, form: str = "10-K") -> List[str]:
    rows = neo4j_run(driver, """
        MATCH (c:Company {ticker:$ticker})-[:FILED]->(f:Filing {form:$form})
        RETURN f.accession AS accession
        ORDER BY f.filed_date DESC
        LIMIT 2
    """, {"ticker": ticker, "form": form})
    return [r["accession"] for r in rows]


def neo4j_allowed_chunk_ids(driver, ticker: str, accessions: List[str], allowed_items: set[str]) -> set[str]:
    rows = neo4j_run(driver, """
        MATCH (c:Company {ticker:$ticker})-[:FILED]->(f:Filing)
        WHERE f.accession IN $accessions
        MATCH (f)-[:HAS_CHUNK]->(ch:Chunk)
        WHERE ch.item IN $items
        RETURN ch.chunk_id AS chunk_id
    """, {"ticker": ticker, "accessions": accessions, "items": list(allowed_items)})
    return {r["chunk_id"] for r in rows}


def neo4j_expand_by_entities(driver, seed_chunk_ids: List[str], accessions: List[str], limit: int = 200) -> List[Tuple[str, str]]:
    """
    Returns list of (chunk_id, chunk_text) neighbors:
    seed -> mentions -> entity <- mentions - neighbor
    constrained to the same accessions
    """
    rows = neo4j_run(driver, """
        MATCH (seed:Chunk)
        WHERE seed.chunk_id IN $seed_ids
        MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(nbr:Chunk)
        WHERE nbr.accession IN $accessions
        RETURN DISTINCT nbr.chunk_id AS chunk_id, nbr.text AS text
        LIMIT $limit
    """, {"seed_ids": seed_chunk_ids, "accessions": accessions, "limit": limit})
    return [(r["chunk_id"], r["text"]) for r in rows]


# -----------------------------
# Hybrid retrieval + generation
# -----------------------------
def keyword_boost(text: str, query: str) -> float:
    q = query.lower()
    t = text.lower()
    boost = 0.0
    for kw in ["china", "prc", "supply chain", "tariff", "geopolitical"]:
        if kw in q and kw in t:
            boost += 0.15
    return boost


def build_rag_prompt():
    # Using from_template as you requested
    return ChatPromptTemplate.from_template(
        """
You are a precise financial analyst assistant.

Rules:
- Use ONLY the provided context excerpts from SEC filings.
- If the context is insufficient, say "I don't know."
- When comparing years, explicitly call out "Latest filing" vs "Prior filing" if present in context.
- Provide citations inline like: (accession={accession}, item={item}, chunk_id={chunk_id})

Question:
{question}

Context excerpts:
{context}

Answer:
"""
    )


def main():
    # -----------------------------
    # 1) Demo SEC-like corpus (swap this with your real EDGAR chunks)
    # -----------------------------
    docs = [
        Document(
            page_content=(
                "Item 1A. Risk Factors. Our operations and performance depend significantly on our supply chain, "
                "including manufacturing partners concentrated in Asia. Geopolitical tensions and trade restrictions, "
                "including tariffs, may disrupt supply chain continuity and increase costs. We have exposure to China "
                "through manufacturing and sales channels."
            ),
            metadata={
                "ticker": "AAPL", "accession": "0000320193-25-000001", "form": "10-K",
                "filed_date": "2025-10-30", "fiscal_year": 2025, "item": "1A", "chunk_id": "aapl25_1A_001"
            },
        ),
        Document(
            page_content=(
                "Item 7. MD&A. We mitigated inflationary pressures through supplier negotiations and logistics optimization. "
                "However, supply chain constraints may still impact product availability. We continue to diversify "
                "manufacturing footprint to manage regional concentration risk."
            ),
            metadata={
                "ticker": "AAPL", "accession": "0000320193-25-000001", "form": "10-K",
                "filed_date": "2025-10-30", "fiscal_year": 2025, "item": "7", "chunk_id": "aapl25_7_001"
            },
        ),
        Document(
            page_content=(
                "Item 1A. Risk Factors. Our supply chain is subject to disruptions from geopolitical events and regulatory actions. "
                "We have material exposure to the PRC in our supply chain. Trade restrictions and tariffs could adversely affect "
                "our results. We are taking steps to diversify suppliers."
            ),
            metadata={
                "ticker": "AAPL", "accession": "0000320193-24-000001", "form": "10-K",
                "filed_date": "2024-10-31", "fiscal_year": 2024, "item": "1A", "chunk_id": "aapl24_1A_001"
            },
        ),
        Document(
            page_content=(
                "Item 7. MD&A. We experienced supply chain-related cost increases and certain component shortages. "
                "We continued efforts to adjust production planning and logistics to address disruptions."
            ),
            metadata={
                "ticker": "AAPL", "accession": "0000320193-24-000001", "form": "10-K",
                "filed_date": "2024-10-31", "fiscal_year": 2024, "item": "7", "chunk_id": "aapl24_7_001"
            },
        ),
    ]

    # -----------------------------
    # 2) Start Neo4j + build graph
    # -----------------------------
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_init_schema(driver)
    neo4j_upsert_demo_data(driver, docs)

    # -----------------------------
    # 3) Build FAISS vector store (local embeddings via Ollama)
    # -----------------------------
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)  # LangChain Ollama embeddings integration  [oai_citation:5‡LangChain Docs](https://docs.langchain.com/oss/python/integrations/text_embedding/ollama?utm_source=chatgpt.com)
    vs = FAISS.from_documents(docs, embeddings)

    # -----------------------------
    # 4) Hybrid retrieval for an SEC question
    # -----------------------------
    question = "What did Apple say about supply chain risks and China exposure in the latest 10-K, and what changed vs prior year?"

    t0 = time.perf_counter()

    # 4A) Graph constraint: latest two filings + only Item 1A/7 chunks
    accessions = neo4j_latest_two_filings(driver, ticker=TICKER, form="10-K")
    allowed_chunk_ids = neo4j_allowed_chunk_ids(driver, ticker=TICKER, accessions=accessions, allowed_items=ALLOWED_ITEMS)

    # 4B) Vector recall (retrieve more, then filter by graph constraint)
    # FAISS doesn't do metadata filtering natively; we over-retrieve then filter.
    vector_hits: List[Document] = vs.similarity_search(question, k=VECTOR_K)

    seeds: List[Tuple[str, Document, float]] = []
    for d in vector_hits:
        cid = d.metadata["chunk_id"]
        if cid in allowed_chunk_ids:
            # naive semantic score proxy: we don't get distance from similarity_search; keep 0 for now
            seeds.append((cid, d, 0.0))

    seed_ids = [cid for cid, _, _ in seeds][:10]  # keep top seed set

    # 4C) Graph expansion: same-entity neighbors within the same accessions
    expanded = neo4j_expand_by_entities(driver, seed_chunk_ids=seed_ids, accessions=accessions, limit=200)

    # Merge seed docs + expanded docs (dedupe by chunk_id)
    merged: Dict[str, Document] = {d.metadata["chunk_id"]: d for _, d, _ in seeds}
    for cid, text in expanded:
        if cid not in merged:
            # pull metadata from graph (simplify: query it)
            md_rows = neo4j_run(driver, """
                MATCH (ch:Chunk {chunk_id:$cid})
                RETURN ch.ticker AS ticker, ch.accession AS accession, ch.item AS item, ch.fiscal_year AS fiscal_year
            """, {"cid": cid})
            md = md_rows[0] if md_rows else {"ticker": TICKER, "accession": "unknown", "item": "unknown", "fiscal_year": -1}
            merged[cid] = Document(page_content=text, metadata={**md, "chunk_id": cid})

    # 4D) Simple rerank: semantic (approx) + keyword boosts + section boost
    ranked: List[Tuple[float, Document]] = []
    for d in merged.values():
        base = 0.0
        base += 0.10 if d.metadata.get("item") == "1A" else 0.0  # risk factors priority
        base += keyword_boost(d.page_content, question)
        # recency boost
        base += 0.08 if d.metadata.get("fiscal_year", 0) == max([docs[0].metadata["fiscal_year"], docs[2].metadata["fiscal_year"]]) else 0.0
        ranked.append((base, d))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top_docs = [d for _, d in ranked[:FINAL_CONTEXT_K]]

    # 4E) Build context with citations
    def fmt(d: Document) -> str:
        md = d.metadata
        return (
            f"[accession={md.get('accession')}, item={md.get('item')}, fiscal_year={md.get('fiscal_year')}, chunk_id={md.get('chunk_id')}]\n"
            f"{d.page_content}"
        )

    context = "\n\n---\n\n".join(fmt(d) for d in top_docs)

    # -----------------------------
    # 5) Generate answer (LangChain + Ollama) using from_template
    # -----------------------------
    prompt = build_rag_prompt()
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    t1 = time.perf_counter()

    print("\n=== Context Used ===\n")
    print(context)
    print("\n=== Answer ===\n")
    print(answer)
    print(f"\nLatency: {(t1 - t0)*1000:.1f} ms")

    driver.close()


if __name__ == "__main__":
    main()