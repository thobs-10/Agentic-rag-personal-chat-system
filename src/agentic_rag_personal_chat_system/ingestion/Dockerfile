FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# --- Dependency layer (cached unless pyproject.toml / uv.lock change) ---
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra ingestion --no-dev --no-install-project

# --- Source layer ---
COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra ingestion --no-dev

RUN mkdir -p /app/data/technical /app/data/personal

CMD ["python", "-m", "agentic_rag_personal_chat_system.ingestion.src.component.run_ingestion"]
