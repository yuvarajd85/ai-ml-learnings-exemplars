'''
Created on 1/20/26 at 9:24 PM
By yuvarajdurairaj
Module Name LDDashRagChatbot
'''


import os
import base64
import json
import uuid
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update

from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -----------------------------
# Config
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in env/.env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))  # chat history window

# -----------------------------
# LLM + Embeddings
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=TEMPERATURE,
    google_api_key=GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)

# -----------------------------
# RAG prompt + chain builder (LCEL)
# -----------------------------
SYSTEM_PROMPT = (
    "You are a precise senior engineering mentor. "
    "Use the provided context to answer. "
    "If the context does not contain the answer, say so explicitly and then answer from general knowledge "
    "while clearly labeling it as 'General knowledge' vs 'From document'. "
    "Keep it concise and correct."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Question:\n{question}\n\n"
     "Context:\n{context}\n\n"
     "Answer:")
])

def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "NO_RELEVANT_CONTEXT"
    # Include source metadata if present
    chunks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        loc = f"{src}" + (f" (page {page})" if page is not None else "")
        chunks.append(f"[{i}] {loc}\n{d.page_content}")
    return "\n\n---\n\n".join(chunks)

def build_rag_chain(retriever):
    # Your requested shape: {"context": retriever, "question": RunnablePassthrough | format_docs} | System_prompt | LLM | StrOutputParser
    # Correct LCEL equivalent:
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# -----------------------------
# Upload decoding + parsing
# -----------------------------
def decode_upload(contents: str) -> bytes:
    # contents: "data:<mime>;base64,<payload>"
    header, b64data = contents.split(",", 1)
    return base64.b64decode(b64data)

def parse_txt_like(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")

def parse_json(raw: bytes) -> str:
    try:
        obj = json.loads(raw.decode("utf-8"))
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return parse_txt_like(raw)

def parse_pdf(raw: bytes, filename: str) -> List[Document]:
    # pypdf can read from a file-like buffer
    import io
    reader = PdfReader(io.BytesIO(raw))
    docs: List[Document] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": idx + 1}
            ))
    return docs

def parse_docx(raw: bytes, filename: str) -> List[Document]:
    import io
    doc = DocxDocument(io.BytesIO(raw))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paragraphs).strip()
    if not text:
        return []
    # Treat as single document; splitter will chunk
    return [Document(page_content=text, metadata={"source": filename})]

def load_documents_from_upload(contents: str, filename: str) -> List[Document]:
    raw = decode_upload(contents)
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    if ext in ("txt", "md", "csv"):
        return [Document(page_content=parse_txt_like(raw), metadata={"source": filename})]
    if ext == "json":
        return [Document(page_content=parse_json(raw), metadata={"source": filename})]
    if ext == "pdf":
        return parse_pdf(raw, filename)
    if ext == "docx":
        return parse_docx(raw, filename)

    raise ValueError(f"Unsupported file type: .{ext}. Supported: txt/md/csv/json/pdf/docx")

# -----------------------------
# Chunk + vectorstore build
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def build_vectorstore(docs: List[Document], collection_name: str) -> Chroma:
    chunks = splitter.split_documents(docs)
    # In-memory chroma (persist_directory=None). Good for dev.
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )
    return vs

# -----------------------------
# Simple in-memory per-session storage (DEV ONLY)
# -----------------------------
# Key: session_id -> {"vs": Chroma, "retriever": Retriever, "filename": str}
RAG_SESSIONS = {}

def trim_history(history):
    if not history:
        return []
    return history[-(MAX_TURNS * 2):]

# -----------------------------
# Dash UI
# -----------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "maxWidth": "900px",
        "margin": "0 auto",
        "padding": "16px",
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    },
    children=[
        html.H2("Dash Chatbot (Gemini + LangChain RAG)", style={"marginBottom": "8px"}),

        # Per-browser-session id
        dcc.Store(id="session-id", data=str(uuid.uuid4())),

        # Chat history
        dcc.Store(id="chat-store", data=[]),

        # UI
        html.Div(
            id="chat-window",
            style={
                "height": "520px",
                "overflowY": "auto",
                "border": "1px solid #ddd",
                "borderRadius": "12px",
                "padding": "12px",
                "background": "#fafafa",
            },
        ),

        html.Div(style={"height": "12px"}),

        html.Div(
            style={"display": "flex", "gap": "10px", "alignItems": "center"},
            children=[
                dcc.Upload(
                    id="upload-file",
                    children=html.Button(
                        "Attach (build RAG index)",
                        style={
                            "borderRadius": "12px",
                            "border": "1px solid #ddd",
                            "background": "white",
                            "cursor": "pointer",
                            "padding": "8px 12px",
                            "fontWeight": "600",
                        },
                    ),
                    multiple=False,
                ),
                html.Button(
                    "Clear attachment",
                    id="clear-attachment-btn",
                    n_clicks=0,
                    style={
                        "borderRadius": "12px",
                        "border": "1px solid #ddd",
                        "background": "white",
                        "cursor": "pointer",
                        "padding": "8px 12px",
                    },
                ),
                html.Div(id="attachment-status", style={"color": "#666"}, children="No attachment indexed."),
            ],
        ),

        html.Div(style={"height": "10px"}),

        html.Div(
            style={"display": "flex", "gap": "8px"},
            children=[
                dcc.Input(
                    id="user-input",
                    type="text",
                    placeholder="Type a message… (Enter to send)",
                    debounce=True,
                    style={
                        "flex": "1",
                        "height": "44px",
                        "borderRadius": "12px",
                        "padding": "0 12px",
                        "border": "1px solid #ddd",
                    },
                ),
                html.Button(
                    "Send",
                    id="send-btn",
                    n_clicks=0,
                    style={
                        "width": "110px",
                        "borderRadius": "12px",
                        "border": "1px solid #ddd",
                        "background": "white",
                        "cursor": "pointer",
                        "fontWeight": "600",
                    },
                ),
            ],
        ),

        html.Div(style={"height": "10px"}),

        html.Div(
            style={"display": "flex", "gap": "8px"},
            children=[
                html.Button(
                    "Clear chat",
                    id="clear-btn",
                    n_clicks=0,
                    style={
                        "borderRadius": "12px",
                        "border": "1px solid #ddd",
                        "background": "white",
                        "cursor": "pointer",
                    },
                ),
                html.Div(
                    style={"color": "#666", "paddingTop": "6px"},
                    children=f"Model: {GEMINI_MODEL} | Embeddings: {EMBEDDING_MODEL} | top_k={TOP_K} | chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}",
                ),
            ],
        ),
    ],
)

def render_chat(history):
    bubbles = []
    for msg in history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        is_user = role == "user"

        bubbles.append(
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "flex-end" if is_user else "flex-start",
                    "marginBottom": "10px",
                },
                children=[
                    html.Div(
                        content,
                        style={
                            "maxWidth": "80%",
                            "whiteSpace": "pre-wrap",
                            "padding": "10px 12px",
                            "borderRadius": "12px",
                            "border": "1px solid #ddd",
                            "background": "#e8f0fe" if is_user else "white",
                        },
                    )
                ],
            )
        )
    if not bubbles:
        bubbles = [html.Div("Upload a doc (optional) and ask a question.", style={"color": "#888"})]
    return bubbles

# -----------------------------
# Callbacks: Upload -> Build RAG index
# -----------------------------
@app.callback(
    Output("attachment-status", "children"),
    Input("upload-file", "contents"),
    State("upload-file", "filename"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_upload(contents, filename, session_id):
    if not contents or not filename:
        return no_update

    try:
        docs = load_documents_from_upload(contents, filename)
        if not docs:
            return f"[Attachment error] No extractable text found in {filename}"

        collection_name = f"rag_{session_id}"
        vs = build_vectorstore(docs, collection_name=collection_name)
        retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

        RAG_SESSIONS[session_id] = {
            "vs": vs,
            "retriever": retriever,
            "filename": filename,
            "doc_count": len(docs),
        }

        # quick stats: how many chunks
        chunk_count = vs._collection.count()  # chroma internal; OK for dev
        return f"Indexed: {filename} | pages/docs={len(docs)} | chunks={chunk_count} | top_k={TOP_K}"

    except Exception as e:
        return f"[Attachment error] {type(e).__name__}: {e}"

@app.callback(
    Output("attachment-status", "children", allow_duplicate=True),
    Input("clear-attachment-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def clear_attachment(n, session_id):
    if not n:
        return no_update
    # Drop the in-memory index for this session
    if session_id in RAG_SESSIONS:
        del RAG_SESSIONS[session_id]
    return "No attachment indexed."

# -----------------------------
# Callbacks: Chat
# -----------------------------
@app.callback(
    Output("chat-store", "data"),
    Output("user-input", "value"),
    Input("send-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    Input("user-input", "n_submit"),
    State("user-input", "value"),
    State("chat-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_send_or_clear(send_clicks, clear_clicks, n_submit, user_text, history, session_id):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "clear-btn":
        return [], ""

    if not user_text or not user_text.strip():
        return no_update, no_update

    history = history or []
    user_text = user_text.strip()

    history.append({"role": "user", "content": user_text})
    history = trim_history(history)

    try:
        # If RAG index exists, use LCEL RAG chain. Otherwise fallback to plain LLM.
        rag_state = RAG_SESSIONS.get(session_id)

        if rag_state and rag_state.get("retriever"):
            rag_chain = build_rag_chain(rag_state["retriever"])
            assistant_text = rag_chain.invoke(user_text)
        else:
            # Plain chat fallback (no RAG)
            assistant_text = llm.invoke(user_text).content

        assistant_text = (assistant_text or "").strip() or "[No response]"
    except Exception as e:
        assistant_text = f"[ERROR] {type(e).__name__}: {e}"

    history.append({"role": "assistant", "content": assistant_text})
    history = trim_history(history)

    return history, ""

@app.callback(
    Output("chat-window", "children"),
    Input("chat-store", "data"),
)
def update_chat_window(history):
    return render_chat(history or [])

if __name__ == "__main__":
    app.run(debug=True)
