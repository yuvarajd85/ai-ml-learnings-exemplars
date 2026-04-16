'''
Created on 3/27/26 at 11:49 PM
By yuvarajdurairaj
Module Name LCOllamaAgent
'''
import os
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, add_messages

load_dotenv()

@tool(description="Search the web using Writer API for a given query and return the result")
def call_websearch(search_query:str):
    writer_api_url = 'https://api.writer.com/v1/applications/70b99ed7-ca19-468a-80b7-ad298cf018fd'
    print(os.getenv('WRITER_API_KEY'))
    writer_bearer_token = os.getenv('WRITER_API_KEY')
    header = {
        "Content-Type" : "application/json",
        "Authorization" : f"Bearer {writer_bearer_token}"
    }
    data = {
        "inputs" : [
            {
                "id" : "query",
                "value" : [
                    search_query
                ]
            }
        ],
        "stream" : True
    }

    response = requests.post(url=writer_api_url, headers=header,json=data)
    print(response.text)
    return response.text


llm_model = ChatOllama(model="llama3.2")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

tools = [call_websearch]
tool_executor ={t.name: t for t in tools}

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


# ── 4. Routing Logic ──────────────────────────────────────────────────────────

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

def main():
    query = "Explain Tokens, embeddings, Tokenizer, vectors"
    agent = build_graph()

    result = agent.invoke({
        "messages" : [HumanMessage(content=query)]
    })

    for message in result['messages']:
        if isinstance(message, AIMessage) and message.content:
            print(f"\n[Agent] {message.content}")

if __name__ == '__main__':
    main()
