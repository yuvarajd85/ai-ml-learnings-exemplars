'''
Created on 3/3/26 at 12:17 AM
By yuvarajdurairaj
Module Name LCOllamaRagEvalLog
'''

import json
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS


LOG_PATH = "rag_eval_logs.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_write_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_retrieved_docs(docs_with_scores: List[Tuple[Document, float]]) -> str:
    # Keep context clean: only text, no scores
    return "\n\n".join(d.page_content for d, _ in docs_with_scores)


def build_eval_prompt() -> ChatPromptTemplate:
    # Using from_template as requested
    return ChatPromptTemplate.from_template(
        """
You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.

TASK:
Given CONTEXT, QUESTION, and ANSWER, score the ANSWER.

SCORING:
- relevance: 1-5 (does it answer the question?)
- faithfulness: 1-5 (are statements supported by the context?)
- refusal_correctness: 1-5 (if context is insufficient, did it correctly say "I don't know"?)

RULES:
- If the answer includes information not in the context, reduce faithfulness.
- If the context does not contain enough information and the answer still proceeds, reduce refusal_correctness.
- Output MUST be valid JSON only, no markdown, no extra text.

Return JSON with keys:
relevance, faithfulness, refusal_correctness, notes

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""
    )


def main():
    # -----------------------------
    # 1) Sample corpus
    # -----------------------------
    docs = [
        Document(page_content="Polars is a fast DataFrame library built in Rust with lazy execution support."),
        Document(page_content="Pandas is the most common Python DataFrame library; great ecosystem, slower on huge data."),
        Document(page_content="RAG retrieves relevant documents and feeds them to an LLM to reduce hallucinations."),
        Document(page_content="Gradient boosting (e.g., XGBoost/LightGBM) often wins on tabular data."),
    ]

    # -----------------------------
    # 2) Embeddings via Ollama
    # -----------------------------
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # -----------------------------
    # 3) Build vector index
    # -----------------------------
    vs = FAISS.from_documents(docs, embeddings)

    # -----------------------------
    # 4) RAG prompt (from_template)
    # -----------------------------
    rag_prompt = ChatPromptTemplate.from_template(
        """
You are a precise assistant.

INSTRUCTIONS:
- Answer using ONLY the provided context.
- If the context does not contain enough information, say exactly: "I don't know."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )

    # -----------------------------
    # 5) LLMs (one for answering, one for evaluation)
    # -----------------------------
    answer_llm = ChatOllama(model="llama3.2", temperature=0.2)
    judge_llm = ChatOllama(model="llama3.2", temperature=0.0)  # deterministic judging

    eval_prompt = build_eval_prompt()

    # RAG chain (note: context is provided externally after retrieval)
    rag_chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | answer_llm
        | StrOutputParser()
    )

    # Eval chain
    eval_chain = (
        eval_prompt
        | judge_llm
        | StrOutputParser()
    )

    # -----------------------------
    # 6) Ask a question
    # -----------------------------
    question = "When should I prefer Polars over Pandas, and why?"
    k = 3

    t0 = time.perf_counter()

    # Retrieve with scores (better for logging than retriever.as_retriever())
    retrieved: List[Tuple[Document, float]] = vs.similarity_search_with_score(question, k=k)

    context = format_retrieved_docs(retrieved)

    # Generate answer
    answer = rag_chain.invoke({"context": context, "question": question})
    print(f"Rag_Chain Answer: {answer}")

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    # Evaluate answer quality (LLM-as-judge)
    eval_raw = eval_chain.invoke({"context": context, "question": question, "answer": answer})

    # Parse eval JSON safely (judge might still output imperfect JSON sometimes)
    eval_data: Dict[str, Any]
    try:
        eval_data = json.loads(eval_raw)
    except json.JSONDecodeError:
        eval_data = {
            "relevance": None,
            "faithfulness": None,
            "refusal_correctness": None,
            "notes": f"Judge output was not valid JSON. Raw: {eval_raw[:500]}",
        }

    # -----------------------------
    # 7) Log everything
    # -----------------------------
    log_record = {
        "ts_utc": utc_now_iso(),
        "question": question,
        "k": k,
        "retrieval": [
            {
                "score": float(score),  # FAISS distance/score semantics depend on setup; still useful relatively
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc, score in retrieved
        ],
        "context_used": context,
        "answer": answer,
        "latency_ms": latency_ms,
        "evaluation": eval_data,
    }

    safe_write_jsonl(LOG_PATH, log_record)

    # Print result
    print("ANSWER:\n", answer)
    print("\nEVAL:\n", json.dumps(eval_data, indent=2))
    print(f"\nLogged to: {LOG_PATH}")
    print(f"Latency: {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
