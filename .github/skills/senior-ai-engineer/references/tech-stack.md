# AI Engineering Tech Stack Reference

A curated decision guide for choosing the right tool at each layer of an AI system.

---

## Orchestration / Agent Frameworks

| Tool | When to Choose |
|---|---|
| **LangGraph** | Stateful multi-step agents; cyclic graphs; complex branching logic; production grade |
| **LangChain** | Rapid prototyping; large ecosystem of pre-built chains and tools |
| **LlamaIndex** | Document-heavy RAG; advanced indexing strategies; query engines |
| **AutoGen** | Multi-agent conversation loops; code execution agents |
| **CrewAI** | Role-based multi-agent crews with sequential/hierarchical processes |
| **Haystack** | Pipelines for search + QA; strong document processing |

**This project uses**: LangGraph

---

## LLM Providers

| Provider | When to Choose |
|---|---|
| **HuggingFace Hub / Inference API** | Open weights; self-hosted; cost control; custom fine-tunes |
| **OpenAI** | GPT-4o / GPT-4o-mini; best general capability; function calling |
| **Anthropic Claude** | Long context; instruction following; safety-critical apps |
| **Google Gemini** | Multimodal; long context; Google Cloud integration |
| **Ollama** | Local inference; privacy; offline environments |
| **vLLM / TGI** | Self-hosted, high-throughput serving of open models |

**This project uses**: HuggingFace

### LLM Client Libraries
| Library | Notes |
|---|---|
| `langchain-openai` | OpenAI + Azure OpenAI via LangChain |
| `langchain-anthropic` | Anthropic via LangChain |
| `huggingface_hub` | Inference API, model downloads |
| `transformers` | Local model loading and inference |
| `litellm` | Unified interface across 100+ providers |

---

## Embedding Models

| Model | Dimensions | Notes |
|---|---|---|
| `BAAI/bge-large-en-v1.5` | 1024 | Strong general English retrieval |
| `BAAI/bge-m3` | 1024 | Multilingual, dense + sparse hybrid |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, lightweight |
| `text-embedding-3-small` (OpenAI) | 1536 | Cost-effective, strong |
| `text-embedding-3-large` (OpenAI) | 3072 | Highest OpenAI quality |
| `nomic-embed-text` | 768 | Open, competitive with OpenAI small |

**Rule of thumb**: Pick one model and never mix embedding models in the same collection.

---

## Vector Stores

| Store | When to Choose |
|---|---|
| **Qdrant** | Production; filtering + payload; self-hosted or cloud; fast |
| **Chroma** | Local dev; simple setup; no infra |
| **Pinecone** | Fully managed; serverless; large scale |
| **Weaviate** | Hybrid search (dense + BM25); graph features |
| **pgvector** | Already using Postgres; lower operational overhead |
| **FAISS** | In-memory, CPU/GPU, no infra; research / batch workflows |
| **Milvus** | High-scale; cloud-native; strong filtering |

**This project uses**: Qdrant

### Qdrant Patterns
```python
# Always store metadata as payload
points = [PointStruct(id=uuid, vector=embedding, payload={
    "source": filename,
    "doc_type": "technical",
    "chunk_index": i,
    "text": chunk_text,
})]

# Filter retrieval by metadata
from qdrant_client.models import Filter, FieldCondition, MatchValue
query_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="technical"))])
```

---

## Chunking Strategies {#chunking}

| Strategy | Best For |
|---|---|
| **Fixed-size with overlap** | General documents; simple baseline |
| **Recursive character text splitter** | Mixed content; respects paragraph/sentence boundaries |
| **Semantic chunking** | Coherent topic-based chunks; higher quality retrieval |
| **Markdown-aware splitter** | `.md` docs; preserves headers as chunk boundaries |
| **Code splitter** | Source code; respects function/class boundaries |
| **Parent-child chunking** | Retrieval by small chunk, context from parent chunk |

**Rule**: Always store the `chunk_index` and `parent_doc_id` in metadata for traceability.

---

## Retrieval Strategies

| Strategy | When to Use |
|---|---|
| **Dense (ANN)** | Semantic similarity; standard baseline |
| **Sparse (BM25/TF-IDF)** | Keyword matching; exact terms matter |
| **Hybrid (dense + sparse, RRF)** | Best of both; production default |
| **HyDE** | Hypothetical Document Embeddings; improves recall for vague queries |
| **Self-query retriever** | Queries with metadata filters derived from natural language |
| **Multi-query retriever** | Generate N query variants, union results |
| **MMR (Maximal Marginal Relevance)** | Diversity in results; avoid redundant chunks |

---

## Rerankers

| Model | Notes |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, good quality |
| `BAAI/bge-reranker-large` | High quality |
| Cohere Rerank API | Managed, strong multilingual |
| `FlashRank` | CPU-friendly, lightweight |

Use rerankers when: top-k > 10, documents are heterogeneous, or precision matters more than latency.

---

## Prompt Engineering

### Prompt Template Structure
```
[System role and persona]
[Task description]
[Constraints / format instructions]
[Retrieved context — clearly delimited]
[Few-shot examples if needed]
[User query]
```

### Key Principles
- Use XML-style delimiters for context: `<context>...</context>`, `<question>...</question>`
- State output format explicitly (JSON schema, bullet list, etc.)
- Add "If you don't know, say you don't know" to reduce hallucination
- Keep system prompts versioned alongside code

### Prompt Injection Defence
- Never interpolate raw user input directly into system prompts
- Wrap user content in a dedicated section with clear boundaries
- Validate/sanitize user input before prompt construction

---

## Observability & Evaluation

| Tool | Purpose |
|---|---|
| **Langfuse** | Open-source, self-hostable tracing + prompt versioning + evals; native LangGraph callback; best all-in-one for self-hosted setups |
| **LangSmith** | LangChain/LangGraph tracing, eval datasets, prompt playground |
| **Phoenix (Arize)** | LLM observability, retrieval traces, embedding visualisation |
| **Weave (W&B)** | Experiment tracking, evaluation, model versioning |
| **RAGAS** | RAG evaluation metrics (faithfulness, answer relevancy, context precision) |
| **DeepEval** | Unit test LLM outputs; hallucination, toxicity, correctness |
| **TruLens** | RAG triad evaluation (groundedness, relevance, coherence) |

### When to Choose Langfuse vs. Alternatives
| Need | Best Tool |
|---|---|
| Self-hosted tracing + prompt management in one tool | **Langfuse** |
| Deepest LangGraph/LangChain native integration (managed) | **LangSmith** |
| Retrieval trace visualisation + embedding explorer | **Phoenix** |
| Experiment tracking + model versioning | **Weave (W&B)** |

### Langfuse Integration Pattern (this project)
```python
# Attach at graph invocation — traces all nodes, LLM calls, and token usage automatically
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler()  # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
graph.invoke(state, config={"callbacks": [langfuse_handler]})
```
Required env vars: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (use `https://cloud.langfuse.com` for managed or your self-hosted URL).

### Key RAG Metrics
| Metric | What It Measures |
|---|---|
| Context Precision | Are retrieved chunks relevant to the query? |
| Context Recall | Are all relevant chunks being retrieved? |
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | Does the answer address the question? |

---

## API & Backend

| Tool | Purpose |
|---|---|
| **FastAPI** | Async REST API; auto OpenAPI docs; Pydantic integration |
| **Pydantic v2** | Data validation; structured LLM output; settings management |
| **uvicorn** | ASGI server for FastAPI |
| **httpx** | Async HTTP client; use for external LLM/API calls |
| **tenacity** | Retry logic with exponential back-off |

**This project uses**: FastAPI + Pydantic + uvicorn

---

## Frontend / Chat UI

| Tool | Purpose |
|---|---|
| **Chainlit** | LLM-native chat UI; streaming; message elements; auth |
| **Gradio** | Quick demos; multi-modal; Hugging Face Spaces |
| **Streamlit** | General data apps; easy deployment |
| **Next.js + Vercel AI SDK** | Production web apps; full control; streaming |

**This project uses**: Chainlit

---

## Infrastructure & Deployment

| Concern | Tool |
|---|---|
| Containerisation | Docker + docker-compose |
| Orchestration | Kubernetes (k8s) / Docker Swarm |
| CI/CD | GitHub Actions |
| Secrets management | Environment variables + `.env` files (dev) / Vault / AWS Secrets Manager (prod) |
| Config management | YAML + `pydantic-settings` |
| Dependency management | `uv` or `pip` with `pyproject.toml` |

### Production Checklist
- [ ] All secrets in env vars / secrets manager — never in code or image
- [ ] Health check endpoints on all services
- [ ] Readiness and liveness probes in k8s manifests
- [ ] Resource limits set on all containers
- [ ] Vector store collection exists before service starts (migration script)
- [ ] LLM API key validated at startup
