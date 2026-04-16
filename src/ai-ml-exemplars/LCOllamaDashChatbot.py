'''
Created on 3/29/26 at 11:49 PM
By yuvarajdurairaj
Module Name LCOllamaDashChatbot
'''

import os
from typing import Annotated
import requests
from dotenv import load_dotenv
from typing_extensions import TypedDict

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, add_messages

load_dotenv()

# ── 1. Tools ──────────────────────────────────────────────────────────────────

@tool(description="Search the web using Writer API for a given query and return the result")
def call_websearch(search_query: str):
    writer_api_url = 'https://api.writer.com/v1/applications/70b99ed7-ca19-468a-80b7-ad298cf018fd'
    writer_bearer_token = os.getenv('WRITER_API_KEY')
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {writer_bearer_token}"
    }
    data = {
        "inputs": [
            {
                "id": "query",
                "value": [search_query]
            }
        ],
        "stream": True
    }

    try:
        response = requests.post(url=writer_api_url, headers=header, json=data)
        return response.text
    except Exception as e:
        return f"Error calling web search: {e}"

# ── 2. Agent Graph Setup ──────────────────────────────────────────────────────

llm_model = ChatOllama(model="llama3.2")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

tools = [call_websearch]
tool_executor = {t.name: t for t in tools}

def agent_node(state: AgentState) -> AgentState:
    """LLM decides whether to call a tool or respond directly."""
    response = llm_model.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState) -> AgentState:
    """Executes any tool calls requested by the LLM."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"\n[Tool Called] {tool_name} with args: {tool_args}")

        result = tool_executor[tool_name].invoke(tool_args)
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}

def should_continue(state: AgentState) -> str:
    """Route to tool_node if LLM made tool calls, otherwise end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()

agent_app = build_graph()

# ── 3. Dash UI ──────────────────────────────────────────────────────────────

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
        html.H2("Ollama Agent Chatbot (LangChain + Dash + Llama 3.2)", style={"marginBottom": "8px"}),

        # Stores chat history as list of dicts for Dash, but we convert to LC messages
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
                    placeholder="Type a message… (Enter to send)",
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
                    children="Model: llama3.2",
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

    # Prepare LangChain messages for Agent invocation
    # We want to maintain conversation history in the agent's state
    lc_messages = []
    for msg in history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    try:
        # Invoke agent
        result = agent_app.invoke({"messages": lc_messages})
        # The last message should be the final AIMessage from the agent
        final_message = result['messages'][-1]
        if isinstance(final_message, AIMessage):
            assistant_text = final_message.content
        else:
            assistant_text = str(final_message)
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
    app.run(debug=True)
