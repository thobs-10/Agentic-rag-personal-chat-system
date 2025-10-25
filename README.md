# Agentic RAG Personal Chat System

An intelligent conversational system using LangGraph for agent orchestration and RAG (Retrieval Augmented Generation) for enhanced responses.

## Features

- 🤖 Multi-agent system with specialized agents for different query types
- 📚 Retrieval-augmented generation for knowledge-grounded responses
- 🧠 Smart query routing based on content analysis
- 🔄 Modular architecture with LangGraph orchestration
- 🌐 FastAPI backend with easy-to-use endpoints

## Architecture

This system consists of:

1. **LangGraph Orchestration**: Manages the flow between different agents
2. **Specialized Agents**: Technical assistant and personal assistant
3. **Vector Database**: Uses Qdrant for efficient similarity search
4. **FastAPI Backend**: Provides REST endpoints for interaction

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/agentic-rag-personal-chat-system.git
cd agentic-rag-personal-chat-system
```

Install dependencies:

```bash
pip install -e .
```

### Running with Docker

Build and run the containers:

```bash
docker-compose up -d
```

The API will be accessible at `http://localhost:8000`.

### Running Locally

Start the FastAPI server:

```bash
uvicorn src.agentic_rag_personal_chat_system.backend.src.api:app --reload
```

## API Usage

### Chat Endpoint

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I fix a memory leak in my Python code?", "mode": "auto"}'
```

Response:

```json
{
  "response": "To fix a memory leak in Python...",
  "agent_type": "technical",
  "sources": [
    {"text": "Example source", "source": "technical_collection", "relevance": 0.98}
  ],
  "metadata": {},
  "error": null
}
```

## Development

### Project Structure

```
src/
  backend/
    src/
      graph.py           # LangGraph implementation
      api.py             # FastAPI application
  ingestion/             # Data ingestion pipeline
  frontend/              # Frontend components (placeholder)
```

## License

[License information]
