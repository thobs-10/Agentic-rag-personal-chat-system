---
name: senior-ai-engineer
description: 'Full senior AI engineer workflow for designing, implementing, testing, and reviewing AI systems. Use when: building RAG pipelines, agentic systems, LLM integrations, vector database operations, model evaluation, prompt engineering, API design for AI backends, orchestration with LangGraph/LangChain/LlamaIndex, embedding management, and Qdrant/Chroma/Pinecone vector stores. Covers software engineering fundamentals applied to AI: system design, code quality, observability, CI/CD, and production readiness.'
argument-hint: 'Task description, e.g. "add a new RAG retriever" or "review the agent graph"'
user-invocable: true
---

# Senior AI Engineer Workflow

## When to Use
- Designing or extending RAG pipelines and agentic systems
- Implementing LLM integrations (HuggingFace, OpenAI, Anthropic, Ollama)
- Working with vector databases (Qdrant, Chroma, Pinecone, FAISS)
- Building or reviewing agent graphs (LangGraph, LangChain, LlamaIndex)
- Writing production-quality Python for AI backends
- Evaluating model/retrieval performance and adding observability
- Reviewing or refactoring AI system components for correctness and safety

---

## Phase 1 — Design

Before writing any code, establish:

1. **Clarify the requirement**
   - What is the input → output contract?
   - Is this a retrieval task, a generation task, or an orchestration task?
   - What latency, accuracy, and cost constraints apply?

2. **Choose the right architecture pattern** — see [Tech Stack Reference](./references/tech-stack.md)
   - Pure RAG (retrieve → generate)
   - Agentic RAG (retrieve → reason → act → generate)
   - Tool-calling agent (plan → select tool → execute → synthesize)
   - Multi-agent (router + specialist agents)

3. **Define data flow**
   - Document source → chunking strategy → embedding model → vector store → retriever type
   - Identify where context is injected into the prompt

4. **Identify dependencies and risks**
   - External API rate limits and fallback strategies
   - PII / sensitive data handling before embedding or passing to LLMs
   - Determinism requirements (temperature, seed, caching)

---

## Phase 2 — Implementation

Follow this order to build incrementally and catch issues early.

### 2a. Data / Ingestion Layer
- Choose a chunking strategy appropriate to document type (see [Tech Stack Reference](./references/tech-stack.md#chunking))
- Embed with a consistent model — record model name, version, and dimension in config
- Upsert into the vector store with metadata (`source`, `doc_type`, `chunk_index`, `timestamp`)
- Verify round-trip: ingest one document, query it back, confirm retrieval

### 2b. Retrieval Layer
- Implement the retriever with explicit `top_k` and similarity threshold
- Add a reranker if precision matters more than recall
- Unit-test retrieval: given a known query, assert the expected document is in top-k
- Log retrieved chunks (source, score) for debugging

### 2c. LLM / Generation Layer
- Isolate prompt templates in dedicated files/config — never hardcode in logic
- Use structured output (Pydantic models) wherever the downstream consumer is code
- Handle token budget: estimate (context + retrieved chunks + response) < model max tokens
- Implement retry logic with exponential back-off for API calls

### 2d. Orchestration / Agent Layer
- Define the graph nodes and edges explicitly before coding (draw or describe the state machine)
- Keep node functions pure where possible (input state → output state)
- Validate state schema with Pydantic at every node boundary
- Use conditional edges for branching; avoid deeply nested logic inside nodes

### 2e. API / Interface Layer
- Follow RESTful conventions; use FastAPI with typed request/response models
- Version endpoints (`/v1/...`)
- Return structured errors with a consistent schema
- Never expose raw LLM responses without sanitization if user-facing

---

## Phase 3 — Quality Checklist

Run through this before marking any task done.

### Code Quality
- [ ] Functions have a single responsibility
- [ ] No magic strings or numbers — use constants or config
- [ ] No credentials or secrets in code or logs
- [ ] Type annotations on all public functions
- [ ] Docstrings on non-trivial functions (purpose + params + returns)

### AI-Specific Quality
- [ ] Prompt templates are versioned and testable in isolation
- [ ] Retrieved context is logged (with scores) at debug level
- [ ] LLM calls are wrapped with error handling and timeouts
- [ ] Embedding model and vector store collection name are in config, not hardcoded
- [ ] Token counts are bounded — no silent truncation

### Security (OWASP-aligned)
- [ ] User input is not injected directly into prompts without sanitization (prompt injection risk)
- [ ] API keys loaded from environment variables / secrets manager only
- [ ] Vector store queries are parameterised — no raw string concatenation
- [ ] PII is scrubbed or masked before sending to external LLM APIs

---

## Phase 4 — Testing

See [Workflow Checklists](./references/workflow-checklists.md#testing) for full test templates.

| Layer | What to Test | Tool |
|---|---|---|
| Ingestion | Chunking output, embedding shape, metadata fields | `pytest` |
| Retrieval | Top-k correctness, threshold filtering | `pytest` + mock vector store |
| LLM integration | Response schema, timeout handling, retry logic | `pytest` + `respx`/`httpx` mocks |
| Agent graph | Node transitions, edge conditions, state mutations | `pytest` + LangGraph test harness |
| API | Request validation, error responses, auth | `pytest` + `httpx.AsyncClient` |
| End-to-end | Full query → response round trip | integration test suite |

### Test Principles
- Mock external LLM and vector store calls in unit tests
- Use fixture factories for chat history and document objects
- Assert on structure (schema) AND content (key phrases) for LLM outputs
- Measure retrieval precision@k and recall@k for regression testing

---

## Phase 5 — Review

When reviewing a PR or self-reviewing before merge:

1. **Correctness**: Does it do what the design specified?
2. **Observability**: Are there logs at the right levels? Are traces/spans added for LLM calls?
3. **Configurability**: Are model names, thresholds, and collection names in config?
4. **Failure modes**: What happens when the vector store is empty? When the LLM returns null?
5. **Performance**: Is there N+1 embedding? Unnecessary repeated LLM calls?
6. **Backward compatibility**: Does this change affect existing vector store schemas or API contracts?

---

## This Project's Stack Quick Reference

| Concern | Tool/Library |
|---|---|
| Agent orchestration | LangGraph |
| LLM provider | HuggingFace (`huggingface_hub`, `transformers`) |
| Vector store | Qdrant |
| Frontend / chat UI | Chainlit |
| API backend | FastAPI |
| Data validation | Pydantic v2 |
| Config management | YAML + `pydantic-settings` |
| Packaging | `pyproject.toml` (uv/pip) |
| Containerisation | Docker + `docker-compose` |
| CI/CD | GitHub Actions (`.github/workflows/`) |

For broader tool selection and alternatives, see [Tech Stack Reference](./references/tech-stack.md).
