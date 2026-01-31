import os
import asyncio
from typing import Literal
from dotenv import load_dotenv

# MCP and LangChain/LangGraph imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# --- 1. AGENT LOGIC COMPONENTS ---

def should_continue(state: MessagesState) -> Literal["tools", END]: # type: ignore
    """Check if the last message contains tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

async def call_model(state: MessagesState, model, tools):
    """The logic for the LLM to decide what to do."""
    # Binding tools to the model allows it to 'see' the MCP tools
    model_with_tools = model.bind_tools(tools)
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

# --- 2. MAIN EXECUTION ---

async def main():
    # Setup the MultiServer Client with both Math (Local) and Weather (HTTP)
    client = MultiServerMCPClient( 
        { 
            "math": {
                "command": "python",
                "args": ["mathserver.py"], 
                "transport": "stdio"
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http"
            }
        }
    )

    print("--- Connecting to MCP Servers ---")
    try:
        # This will fetch tools from BOTH servers
        # Note: If the weather server isn't running, this will throw the ConnectError again.
        tools = await client.get_tools()
    except Exception as e:
        print(f"Connection Error: {e}")
        print("Ensure your Weather server is running at http://localhost:8000/mcp")
        return

    # Initialize Groq with a CURRENTLY SUPPORTED model
    model = ChatGroq(
        model="llama-3.3-70b-versatile", 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # --- 3. BUILD THE GRAPH ---
    
    workflow = StateGraph(MessagesState)

    # Define an async wrapper to handle the node call
    async def agent_node(state: MessagesState):
        return await call_model(state, model, tools)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # ReAct loop logic
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    agent = workflow.compile()

    # --- 4. RUN THE AGENT ---
    
    # Let's test both tools in one prompt
    query = "What is the square root of 64 and what is the weather in New York?"
    inputs = {"messages": [("user", query)]}
    
    print(f"--- Calling Agent with query: {query} ---")
    
    result = await agent.ainvoke(inputs)
    
    print("\nFinal Response:", result["messages"][-1].content)

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found.")
    else:
        asyncio.run(main())