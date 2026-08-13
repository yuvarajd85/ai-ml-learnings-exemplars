"""
LangGraph-based agent for Agentic Chatbot with tools
"""
from typing import Annotated, TypedDict, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.2)

# Tool definitions
@tool(description="Get current date and time")
def get_current_datetime():
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool(description="Fetch weather data from Open-Meteo API")
def get_weather_data(latitude: float, longitude: float):
    import requests
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        response = requests.get(url)
        data = response.json()
        return str(data)
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

@tool(description="Execute a PostgreSQL query")
def execute_postgres_query(query: str):
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            port=os.getenv("DB_PORT", "5432")
        )
        cursor = conn.cursor()
        cursor.execute(query)

        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return f"Query result: {columns} | {results}"
        else:
            conn.commit()
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"

        cursor.close()
        conn.close()
    except Exception as e:
        return f"Error executing PostgreSQL query: {str(e)}"

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    context: str
    tools_used: Annotated[List[str], add]

# Tool executor mapping
tools = [get_current_datetime, get_weather_data, execute_postgres_query]
tool_executor = {t.name: t for t in tools}

# Nodes
def generate_node(state: AgentState):
    """Generate response using LLM"""
    print("--- GENERATION NODE ---")

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant who answers questions based on provided context. Use only the information given in the context to answer the question."),
        ("user", "{question}")
    ])

    # Generate a response
    question = state["messages"][-1].content if state["messages"] else ""

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})

    return {"messages": [AIMessage(content=response)]}

def tool_node(state: AgentState):
    """Execute tools requested by the LLM"""
    print("--- TOOL EXECUTION NODE ---")

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"\n[Tool Called] {tool_name} with args: {tool_args}")

            try:
                result = tool_executor[tool_name].invoke(tool_args)
                tool_results.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                tool_results.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call["id"])
                )

        return {"messages": tool_results}
    else:
        return {"messages": [AIMessage(content="No tools to execute")]}

def should_continue(state: AgentState) -> str:
    """Determine whether to continue with tool execution or end"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return END

# Build the graph
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes (without retrieve since we removed Pinecone)
    workflow.add_node("generate", generate_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("generate")

    # Add edges
    workflow.add_conditional_edges(
        "generate",
        should_continue,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "generate")

    return workflow.compile()

# Initialize and run the agent
def run_agent(query: str):
    """Run the agent with a given query"""
    try:
        # Initialize graph
        graph = build_graph()

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "context": "",
            "tools_used": []
        }

        # Run the agent
        result = graph.invoke(initial_state)

        return result["messages"][-1].content if result["messages"] else "No response generated"

    except Exception as e:
        print(f"Error running agent: {e}")
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    # Test the agent
    test_query = "What's the current weather in Berlin?"
    response = run_agent(test_query)
    print(f"Agent Response: {response}")