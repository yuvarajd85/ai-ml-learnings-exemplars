import os
import asyncio
from typing import Literal, Any, Dict, List
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

def should_continue(state: MessagesState) -> Literal["tools", END]:  # type: ignore
    """Route to tools if the last assistant message contains tool calls."""
    last = state["messages"][-1]

    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"

    # Defensive fallback for some LC message shapes
    additional = getattr(last, "additional_kwargs", {}) or {}
    if additional.get("tool_calls"):
        return "tools"

    return END


async def call_model(state: MessagesState, model, tools):
    """Ask the model what to do next (possibly emitting tool calls)."""
    model_with_tools = model.bind_tools(tools)
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found in environment.")

    # Use a system message to reduce tool chaos
    system_msg = (
        "You are a tool-using assistant. "
        "Use tools when needed to compute or fetch facts. "
        "If a tool fails, explain what failed and answer with what you have. "
        "Do not call tools more than necessary."
    )

    client_config = {
        "math": {
            "command": "python3",
            "args": ["mathserver.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable_http",
        },
    }

    print("--- Connecting to MCP Servers ---")

    # Prefer context manager if your adapter supports it; otherwise, close manually.
    client = MultiServerMCPClient(client_config)

    try:
        tools = await client.get_tools()
    except Exception as e:
        print(f"Connection Error: {e}")
        print("Ensure your Weather server is running at http://127.0.0.1:8000/mcp")
        # Close any stdio subprocess that might have started
        try:
            await client.aclose()
        except Exception:
            pass
        return

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        # Optional: set temperature low for tool reliability
        temperature=0,
    )

    workflow = StateGraph(MessagesState)

    async def agent_node(state: MessagesState):
        return await call_model(state, model, tools)

    workflow.add_node("agent", agent_node)

    # ToolNode will execute tool calls emitted by the agent.
    # If you want more resilience, you can wrap tools with retry logic before passing them in.
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    agent = workflow.compile()

    query = "What is the square root of 64 and what is the weather in New York City, NY, US?"
    inputs = {"messages": [("system", system_msg), ("user", query)]}

    print(f"--- Calling Agent with query: {query} ---")

    try:
        result = await agent.ainvoke(inputs)
        print("\nFinal Response:", result["messages"][-1].content)
    finally:
        # IMPORTANT: close the client to clean up stdio subprocess/pipes
        try:
            await client.aclose()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())