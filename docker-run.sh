#!/bin/bash

# Agentic RAG System Docker Manager
# Usage: ./docker-run.sh [command]

set -e

COMPOSE_FILE="docker-compose.yaml"

case "${1:-help}" in
    "all")
        echo "Starting all services..."
        docker-compose -f $COMPOSE_FILE up --build
        ;;
    "backend")
        echo "Starting backend and database..."
        docker-compose -f $COMPOSE_FILE up --build backend qdrant
        ;;
    "frontend")
        echo "Starting frontend..."
        docker-compose -f $COMPOSE_FILE up --build frontend
        ;;
    "ingest")
        echo "Running ingestion service..."
        docker-compose -f $COMPOSE_FILE --profile ingestion up --build ingestion
        ;;
    "db")
        echo "Starting database only..."
        docker-compose -f $COMPOSE_FILE up qdrant
        ;;
    "stop")
        echo "Stopping all services..."
        docker-compose -f $COMPOSE_FILE down
        ;;
    "clean")
        echo "Cleaning up containers and volumes..."
        docker-compose -f $COMPOSE_FILE down -v
        ;;
    "logs")
        echo "Showing logs..."
        docker-compose -f $COMPOSE_FILE logs -f
        ;;
    "rebuild")
        echo "Rebuilding all images..."
        docker-compose -f $COMPOSE_FILE build --no-cache
        ;;
    "status")
        echo "Service status:"
        docker-compose -f $COMPOSE_FILE ps
        ;;
    "help"|*)
        echo "Agentic RAG System Docker Manager"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  all      - Start all services (backend, frontend, qdrant)"
        echo "  backend  - Start backend and database only"
        echo "  frontend - Start frontend only"
        echo "  ingest   - Run data ingestion (one-time job)"
        echo "  db       - Start database only"
        echo "  stop     - Stop all services"
        echo "  clean    - Stop and remove volumes"
        echo "  logs     - Show all service logs"
        echo "  rebuild  - Rebuild all Docker images"
        echo "  status   - Show service status"
        echo "  help     - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 all      # Start everything"
        echo "  $0 backend  # Development mode"
        echo "  $0 stop     # Stop all"
        ;;
esac