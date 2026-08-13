import os
from typing import Annotated, TypedDict, List
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# --- 1. State Definition ---
# Shared memory across the graph. LangGraph uses this to pass data between nodes.
class AgentState(TypedDict):
    # List of messages (history) - Annotated with 'add' so new messages append instead of overwrite
    messages: Annotated[List[BaseMessage], add]
    # Flags for logic control (e.g., whether the current output is verified)
    is_verified: bool

# --- 2. Tools ---
# Actionable capabilities for the agent to interact with the outside world.
@tool
def search_web(query: str):
    """Search the web to find real-time information."""
    # Placeholder for actual implementation (e.g., Tavily or Google Search)
    return f"Results for {query}: [Simulated Search Result]"

# Tools can be bundled and used by LLMs via .bind_tools()
tools = [search_web]

# --- 3. Nodes ---
# Each node is a function that takes the state, performs an action, and returns updates to the state.

def search_node(state: AgentState):
    """Node: Retrieve information from external sources."""
    print("--- SEARCHING NODE ---")
    # Logic to invoke the search tool would go here
    return {"messages": [HumanMessage(content="Found verified data for the query.")]}

def generation_node(state: AgentState):
    """Node: Generate a response based on gathered information."""
    print("--- GENERATION NODE ---")
    # LLM logic to produce content would go here
    return {"messages": [HumanMessage(content="The verified answer is X.")]}

def verification_node(state: AgentState):
    """Node: Critique the generated answer for accuracy and completeness."""
    print("--- VERIFICATION NODE ---")
    # Logic to check if response meets quality standards
    return {"is_verified": True, "messages": [HumanMessage(content="Verified info.")]}

# --- 4. Graph Construction (The Agentic System) ---
# This defines the flow of the agent's thought process.
workflow = StateGraph(AgentState)

# Registering Nodes: Defining what actions are available in the graph
workflow.add_node("search", search_node)
workflow.add_node("generate", generation_node)
workflow.add_node("verify", verification_node)

# Setting Entry Point and Fixed Edges
workflow.set_entry_point("search")
workflow.add_edge("search", "generate")
workflow.add_edge("generate", "verify")

# Conditional Edges: The 'Reasoning' of the Agent.
# Based on verification, decide whether to finish or loop back for refinement.
def decide_to_complete(state: AgentState):
    if state.get("is_verified", False):
        return "end"
    else:
        return "retry"

workflow.add_conditional_edges(
    "verify",
    decide_to_complete,
    {
        "end": END,
        "retry": "generate"  # Loop back to the generation node for correction
    }
)

# Compile the graph into an executable application
app = workflow.compile()

# --- Execution Example ---
if __name__ == "__main__":
    print("Starting Agentic Workflow...")
    inputs = {"messages": [HumanMessage(content="Who won the 2024 Super Bowl?")]}

    for output in app.stream(inputs):
        # Print each step's progress as it happens
        for node_name, state_update in output.items():
            print(f"Node '{node_name}' produced: {state_update}")
