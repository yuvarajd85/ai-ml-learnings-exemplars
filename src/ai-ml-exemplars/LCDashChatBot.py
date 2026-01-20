'''
Created on 1/20/26 at 12:16 AM
By yuvarajdurairaj
Module Name LCDashChatBot
'''


import os
from dotenv import load_dotenv

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import dash

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# -----------------------------
# LLM setup (Gemini via LangChain)
# -----------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY. Put it in your environment or .env file.")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
)

SYSTEM_PROMPT = (
    "You are a precise senior engineering mentor. "
    "Be direct, correct, and concise. Ask clarifying questions only when necessary."
)

# -----------------------------
# Dash app
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
        html.H2("Gemini Chatbot (LangChain + Dash)", style={"marginBottom": "8px"}),

        # Stores chat history as list of dicts: [{"role": "...", "content": "..."}]
        dcc.Store(id="chat-store", data=[]),

        # Chat window
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

        # Input row
        html.Div(
            style={"display": "flex", "gap": "8px"},
            children=[
                dcc.Textarea(
                    id="user-input",
                    placeholder="Type a message… (Shift+Enter for newline, Enter to send)",
                    style={
                        "flex": "1",
                        "height": "70px",
                        "borderRadius": "12px",
                        "padding": "10px",
                        "border": "1px solid #ddd",
                        "resize": "none",
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
                    id="status",
                    style={"color": "#666", "paddingTop": "6px"},
                    children=f"Model: {GEMINI_MODEL}",
                ),
            ],
        ),
    ],
)

def render_chat(history):
    """Render chat bubbles from history."""
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
        bubbles = [html.Div("Say something to start the chat.", style={"color": "#888"})]
    return bubbles

def history_to_langchain_messages(history):
    """Convert store history into LangChain message objects."""
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        else:
            msgs.append(AIMessage(content=content))
    return msgs

@app.callback(
    Output("chat-store", "data"),
    Output("user-input", "value"),
    Input("send-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    State("user-input", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True,
)
def on_send_or_clear(send_clicks, clear_clicks, user_text, history):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "clear-btn":
        return [], ""

    # Send button
    if not user_text or not user_text.strip():
        return no_update, no_update

    user_text = user_text.strip()
    history = history or []

    # Append user message
    history.append({"role": "user", "content": user_text})

    # Call Gemini with full conversation context
    try:
        lc_msgs = history_to_langchain_messages(history)
        ai = llm.invoke(lc_msgs)
        assistant_text = (ai.content or "").strip() or "[No response]"
    except Exception as e:
        assistant_text = f"[ERROR] {type(e).__name__}: {e}"

    history.append({"role": "assistant", "content": assistant_text})

    return history, ""

@app.callback(
    Output("chat-window", "children"),
    Input("chat-store", "data"),
)
def update_chat_window(history):
    return render_chat(history or [])

if __name__ == "__main__":
    # Run:
    #   python app.py
    # Then open:
    #   http://127.0.0.1:8050
    app.run(debug=True)