import os 
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
import warnings
import warnings
warnings.filterwarnings("ignore")
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

groq_llm=ChatGroq(model="llama-3.3-70b-versatile")
search_tool=TavilySearchResults(max_results=2)

search_tool=TavilySearchResults(max_results=2)
#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"
print('hello')
agent = create_react_agent(
    model=groq_llm,
    tools=[search_tool],
    prompt=system_prompt
)

query = "Tell me about the trends in crypto markets"

state = {"messages": query}

response = agent.invoke(state)

messages = response.get("messages")

ai_messages = [ message.content for message in messages if isinstance(message, AIMessage) ]

print(ai_messages[-1])
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]