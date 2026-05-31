# Workflow Checklists

Detailed checklists and templates for each phase of the senior AI engineer workflow.

---

## Design Checklist

Before writing any code, confirm:

- [ ] Input/output contract is documented (what comes in, what goes out, what types)
- [ ] Architecture pattern selected (pure RAG / agentic RAG / tool-calling / multi-agent)
- [ ] Data sources identified: file types, update frequency, access method
- [ ] Chunking strategy decided and justified
- [ ] Embedding model chosen — record name + version + dimension
- [ ] Vector store collection name and schema defined
- [ ] LLM chosen — record provider, model name, temperature, max tokens
- [ ] Latency budget set (p50 / p95 targets)
- [ ] Cost estimate: embedding cost + LLM cost per query
- [ ] PII / sensitive data handling decided
- [ ] Failure modes identified: what happens when retrieval returns 0 results? When LLM call fails?

---

## Implementation Checklist

### Ingestion
- [ ] Loader handles all required file types (PDF, MD, TXT, DOCX…)
- [ ] Chunking produces non-empty chunks of expected size
- [ ] Metadata fields set on every chunk: `source`, `doc_type`, `chunk_index`, `timestamp`
- [ ] Embeddings generated with the correct model (matches retrieval model)
- [ ] Upsert is idempotent — re-running does not create duplicates
- [ ] Round-trip verified: ingest → query → check document appears in results

### Retrieval
- [ ] `top_k` and similarity threshold configured (not hardcoded)
- [ ] Metadata filtering works correctly
- [ ] Empty result case handled gracefully
- [ ] Retrieved chunks logged at DEBUG level with scores

### LLM Integration
- [ ] Prompt template stored in a dedicated config/file — not inline in logic
- [ ] Output parsed with Pydantic model (structured output)
- [ ] Token budget validated: prompt + context + response ≤ model max
- [ ] Retry with back-off on rate limit / timeout errors
- [ ] Timeout set on every API call

### Agent / Graph
- [ ] State schema defined as Pydantic model
- [ ] Each node has a single responsibility
- [ ] Conditional edges cover all state branches (no missing transitions)
- [ ] Termination condition is explicit — no infinite loops
- [ ] State mutations are traceable (log state at entry/exit of each node)

### API
- [ ] Request and response models are typed Pydantic models
- [ ] Endpoint versioned (`/v1/`)
- [ ] 4xx errors return structured `{ "error": { "code": ..., "message": ... } }`
- [ ] Auth/authorization implemented on protected endpoints
- [ ] No raw exception tracebacks returned to clients

---

## Testing {#testing}

### Unit Test Template — Retrieval

```python
# tests/test_retrieval/test_retriever.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.similarity_search.return_value = [
        {"text": "Expected chunk text", "source": "doc.pdf", "score": 0.91}
    ]
    return store

async def test_retriever_returns_top_k(mock_vector_store):
    retriever = MyRetriever(vector_store=mock_vector_store, top_k=3)
    results = await retriever.retrieve("test query")
    assert len(results) <= 3
    assert results[0]["score"] >= 0.0

async def test_retriever_handles_empty_results(mock_vector_store):
    mock_vector_store.similarity_search.return_value = []
    retriever = MyRetriever(vector_store=mock_vector_store, top_k=3)
    results = await retriever.retrieve("obscure query with no match")
    assert results == []
```

### Unit Test Template — LLM Call

```python
# tests/test_llm/test_llm_client.py
import pytest
from unittest.mock import patch, MagicMock

def test_llm_returns_structured_output():
    mock_response = MagicMock()
    mock_response.content = '{"answer": "Paris", "confidence": 0.95}'

    with patch("src.llm.client.ChatOpenAI.invoke", return_value=mock_response):
        result = my_llm_client.ask("What is the capital of France?", context="France is a country in Europe.")

    assert result.answer == "Paris"
    assert result.confidence > 0.0

def test_llm_retries_on_rate_limit():
    with patch("src.llm.client.ChatOpenAI.invoke", side_effect=[RateLimitError(), mock_response]):
        result = my_llm_client.ask("question", context="context")
    assert result is not None  # succeeded on retry
```

### Unit Test Template — Agent Node

```python
# tests/test_agents/test_nodes.py
import pytest
from src.agents.nodes import retrieve_context_node
from src.models.state import AgentState

def test_retrieve_context_node_populates_state():
    initial_state = AgentState(query="What is LangGraph?", context=[], messages=[])
    result_state = retrieve_context_node(initial_state)
    assert len(result_state.context) > 0
    assert all("text" in chunk for chunk in result_state.context)

def test_retrieve_context_node_with_empty_results(mock_empty_retriever):
    initial_state = AgentState(query="xyzzy nonsense", context=[], messages=[])
    result_state = retrieve_context_node(initial_state)
    assert result_state.context == []
    # node should not raise — empty retrieval is handled
```

### Integration Test Template — API

```python
# tests/test_api/test_chat_endpoint.py
import pytest
from httpx import AsyncClient
from src.backend.src.api import app

@pytest.mark.asyncio
async def test_chat_endpoint_returns_200():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/v1/chat", json={
            "message": "Hello, what can you help me with?",
            "session_id": "test-session"
        })
    assert response.status_code == 200
    body = response.json()
    assert "response" in body
    assert isinstance(body["response"], str)

@pytest.mark.asyncio
async def test_chat_endpoint_rejects_empty_message():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/v1/chat", json={"message": "", "session_id": "x"})
    assert response.status_code == 422  # Pydantic validation error
```

---

## Code Review Checklist

Use this when reviewing a PR or self-reviewing before requesting review.

### General
- [ ] PR description explains *why* (not just *what*)
- [ ] No unrelated changes in the diff
- [ ] No commented-out code left in
- [ ] No TODO/FIXME left without a linked issue

### AI-Specific
- [ ] Prompt templates are not hardcoded inline — they are in config or dedicated files
- [ ] Embedding model name is loaded from config, not hardcoded
- [ ] Vector store collection name is loaded from config, not hardcoded
- [ ] LLM responses are validated before use (not assumed to match expected format)
- [ ] No PII or sensitive data is logged or sent to external APIs without explicit handling

### Performance
- [ ] No N+1 embedding: batch embed documents where possible
- [ ] No redundant LLM calls in a loop that could be parallelised
- [ ] Large document sets are streamed/paginated, not loaded entirely into memory

### Security
- [ ] No hardcoded API keys, tokens, or connection strings
- [ ] User input sanitised before prompt injection
- [ ] Dependency versions pinned in `pyproject.toml`

---

## Debugging Guide

### Retrieval Not Working?
1. Check the embedding model used at ingestion vs. query time — they must match
2. Print the query embedding shape and a stored vector shape — must be identical
3. Lower the similarity threshold temporarily and inspect raw scores
4. Check the collection name in config matches what was used during ingestion
5. Verify the document was actually ingested: query Qdrant directly

### LLM Output Unexpected?
1. Log the exact prompt being sent (before any truncation)
2. Check token count — is context being silently truncated?
3. Test the prompt in an interactive playground with the same model
4. Check if temperature is set too high (> 0.7 causes high variance)
5. Validate that the output parser / Pydantic model matches the prompt instructions

### Agent Graph Stuck or Looping?
1. Enable LangGraph debug tracing: `graph.invoke(..., debug=True)`
2. Print state at every node boundary
3. Check that every conditional edge has a default/fallback branch
4. Verify the termination node is actually reachable from the current state
5. Check for missing keys in the state that a downstream node expects

### API Returning 500?
1. Check server logs — FastAPI exception handler should log the full traceback
2. Reproduce with a minimal request body in the docs UI (`/docs`)
3. Check if the Qdrant connection is healthy (`GET /healthz` or `client.get_collections()`)
4. Check if the LLM API key is set and valid in the environment
