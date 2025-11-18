'''
Created on 11/17/2025 at 10:10 PM
By yuvaraj
Module Name: LangChainAgentExemplar
'''
import os

from dotenv import load_dotenv
from instructor.cli.usage import api_key
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

def get_tavily_client():
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_weather(city:str) -> str:
    """Get weather for a given city."""
    tavily_client = get_tavily_client()
    query = f"Find the weather for the city: {city}"
    results = tavily_client.search(query=query, max_results=5)
    print(f"Raw result from tavily search: {results}")
    return "\n\n".join([f"Source: {r['title']}\n{r['content']}" for r in results.get('results', [])])

def get_anthropic_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        api_key=os.getenv("CLAUDE_API_KEY"),
        temperature=0.2
    )

def get_openai_llm():
    return ChatOpenAI(
        model="gpt-5",
        # temperature=0.2,
        max_tokens=2048,
        timeout=None,
        max_retries=2,
    )
def main():
    langchain_agent = create_agent(
        model=get_openai_llm(),
        tools=[get_weather],
        system_prompt="You are a smart AI Assistant Agent"
    )

    #Invoking the Agent
    agent_result = langchain_agent.invoke(
        {
            "messages" :  [
                {
                    "role" : "user",
                    "content" : "What is current weather in Downingtown, Pennsylvania"
                }
            ]
        }
    )

    print(f"Reponse from smart agent assistant: {agent_result}")


if __name__ == '__main__':
    main()
