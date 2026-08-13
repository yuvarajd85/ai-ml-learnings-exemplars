# Agentic Chatbot with LangGraph

This is an agentic chatbot built using LangGraph that supports:
- RAG (Retrieval Augmented Generation) capabilities (currently simplified)
- Tool execution (datetime, weather, PostgreSQL queries)
- Multi-turn conversations
- FastAPI web interface

## Features Implemented

1. **LangGraph-based Agent**: Uses LangGraph for managing agent states and flows
2. **Tool Integration**: 
   - Get current date/time
   - Fetch weather data from Open-Meteo API
   - Execute PostgreSQL queries
3. **Conversation Management**: Maintains conversation history
4. **Web Interface**: FastAPI endpoint for chat interactions

## Architecture

The agent follows a LangGraph workflow:
1. Generate response using LLM
2. If the LLM requests tools, execute them
3. Return results back to the LLM for final response

## Dependencies

- langchain-core==1.4.7
- langchain-community==0.4.2
- langchain-ollama==0.3.10
- langchain-openai==1.3.2
- fastapi==0.137.0
- uvicorn[standard]==0.32.0
- psycopg2-binary==2.9.9
- python-dotenv==1.0.1
- requests==2.32.3

## Running the Application

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```env
DB_HOST=localhost
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=
DB_PORT=5432
```

3. Start the application:
```bash
uvicorn main:app --reload
```

4. The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /chat` - Send a message to the chatbot
- `GET /health` - Health check endpoint

## Testing

Run tests with:
```bash
python test_agent.py
```

## Notes

- Pinecone integration was removed due to compatibility issues with Python 3.14 and dependency conflicts
- The agent can be extended with additional tools and RAG capabilities as needed
- The current implementation focuses on core functionality and tool execution