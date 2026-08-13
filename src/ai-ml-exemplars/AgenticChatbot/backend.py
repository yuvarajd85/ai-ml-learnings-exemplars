"""
FastAPI backend for Agentic Chatbot with LangGraph and Pinecone RAG
"""
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Agentic Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama LLM and embeddings
llm = ChatOllama(model="llama3.2", temperature=0.2)
embeddings = OpenAIEmbeddings(model="nomic-embed-text")

# Global vector store for Pinecone (initialized on first use)
vector_store = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    user_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

# Tools for the agent
@tool(description="Get current date and time")
def get_current_datetime():
    from datetime import datetime
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
        # Get database connection details from environment
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

# Initialize vector store
def initialize_vector_store():
    global vector_store
    if vector_store is None:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = "rag-text-embedding"

            # Check if index exists, create if not
            if index_name not in pc.list_indexes().names():
                pc.create_index(name=index_name,
                                dimension=3072,
                                metric='cosine',
                                spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}})

                # Wait for index to be ready
                import time
                while not pc.describe_index(index_name).index.status['ready']:
                    time.sleep(1)

            vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                namespace="default"
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
    return vector_store

# Initialize the agent state and graph
class AgentState(BaseModel):
    messages: List[Dict[str, Any]] = []
    context: str = ""

# Agent workflow definition (simplified for now)
def create_agent_graph():
    # This would be expanded with proper nodes and edges later
    pass

@app.get("/")
async def root():
    return {"message": "Agentic Chatbot API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Convert messages to LangChain format
        messages = []
        for msg in request.messages:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        # For now, just return a simple response - this would be expanded with proper agent logic
        response = llm.invoke(messages)

        return ChatResponse(
            response=str(response.content),
            sources=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/upload")
async def upload_document(document_content: str):
    try:
        # Initialize vector store if needed
        vs = initialize_vector_store()
        if not vs:
            raise HTTPException(status_code=500, detail="Failed to initialize vector store")

        # Add document to vector store (simplified)
        # In a real implementation, this would use proper document splitting and embedding

        return {"status": "success", "message": "Document uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)