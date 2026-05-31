.PHONY: setup test test-ingestion test-backend lint clean run docker-all docker-backend docker-frontend docker-ingest docker-db docker-stop docker-clean docker-logs docker-logs-backend docker-logs-frontend docker-rebuild

setup:
	pip install -e .

test:
	python -m pytest -n auto -v src/agentic_rag_personal_chat_system/ingestion/tests src/agentic_rag_personal_chat_system/backend/tests

test-ingestion:
	python -m pytest -n auto -v src/agentic_rag_personal_chat_system/ingestion/tests

test-backend:
	python -m pytest -n auto -v src/agentic_rag_personal_chat_system/backend/tests

lint:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +

run:
	python src/main.py

# Docker Compose Commands
docker-all:
	docker-compose up --build

docker-backend:
	docker-compose up --build backend qdrant ollama

docker-frontend:
	docker-compose up --build frontend

docker-ingest:
	docker-compose --profile ingestion up --build -d qdrant ingestion

docker-db:
	docker-compose up qdrant

docker-ollama:
	docker-compose up  -d ollama

docker-pull-ollama-model:
	docker exec -it ollama ollama pull llama3.2:1b

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v

docker-logs:
	docker-compose logs -f

docker-remove-orphaned-networks:
	docker network prune -f

docker-logs-backend:
	docker-compose logs -f backend

docker-logs-frontend:
	docker-compose logs -f frontend

docker-rebuild:
	docker-compose build --no-cache
