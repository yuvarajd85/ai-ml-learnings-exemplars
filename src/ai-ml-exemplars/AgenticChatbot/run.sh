#!/bin/bash

# Simple script to run the agentic chatbot

echo "Starting Agentic Chatbot..."

# Install dependencies (if needed)
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the FastAPI application
echo "Starting FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000

echo "Server is running at http://localhost:8000"