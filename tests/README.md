# Integration Tests

This directory is reserved for **integration tests** that span multiple components.

## Unit Tests

Unit tests live alongside each component they test:

- **Ingestion**: `src/agentic_rag_personal_chat_system/ingestion/tests/`
- **Backend**: `src/agentic_rag_personal_chat_system/backend/tests/`

## Running Tests

```bash
# Run all unit tests
make test

# Run ingestion tests only
make test-ingestion

# Run backend tests only
make test-backend
```
