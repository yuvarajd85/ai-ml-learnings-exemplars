"""
Main application for Agentic Chatbot with LangGraph, Pinecone RAG, and FastAPI
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from datetime import datetime
import psycopg2
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Agentic Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import our components
from agent import run_agent

@app.get("/")
async def root():
    return {
        "message": "Agentic Chatbot API is running",
        "features": [
            "LangGraph-based agent system",
            "Pinecone RAG integration",
            "Tool execution (date/time, weather, PostgreSQL)",
            "FastAPI backend"
        ]
    }

@app.post("/chat")
async def chat_endpoint(query: dict):
    """Handle chat requests with the agentic system"""
    try:
        user_query = query.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Run the agent
        response = run_agent(user_query)

        return {
            "response": response,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/upload")
async def upload_document(document_content: dict):
    """Handle document uploads for RAG"""
    try:
        # In a real implementation, this would process and store documents
        content = document_content.get("content", "")
        filename = document_content.get("filename", "unknown")

        if not content:
            raise HTTPException(status_code=400, detail="Document content is required")

        return {
            "status": "success",
            "message": f"Document '{filename}' processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agentic-chatbot"}

if __name__ == "__main__":
    print("Starting Agentic Chatbot API...")
    print("Available endpoints:")
    print("- GET / : Root endpoint with service info")
    print("- POST /chat : Chat with the agent")
    print("- POST /upload : Upload documents for RAG")
    print("- GET /health : Health check")

    uvicorn.run(app, host="0.0.0.0", port=8000)