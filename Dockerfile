FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/
COPY src/ /app/src/
COPY README.md /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "src.agentic_rag_personal_chat_system.backend.src.api:app", "--host", "0.0.0.0", "--port", "8000"]
