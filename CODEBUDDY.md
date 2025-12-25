# CODEBUDDY.md This file provides guidance to CodeBuddy Code when working with code in this repository.

## Project Overview

DB-GPT is an open-source project for interacting with data using LLMs. It supports both local model deployment and API proxy models (OpenAI, DeepSeek, SiliconFlow, Ollama, etc.).

## Development Commands

### Environment Setup

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies for proxy model (minimal)
uv sync --all-packages \
  --extra "base" \
  --extra "proxy_openai" \
  --extra "rag" \
  --extra "storage_chromadb" \
  --extra "dbgpts"

# For China region, add: --index-url=https://pypi.tuna.tsinghua.edu.cn/simple
```

### Running the Application

```bash
# Start backend webserver (port 5670)
uv run dbgpt start webserver --config configs/dbgpt-proxy-siliconflow.toml

# Alternative: run directly
uv run python packages/dbgpt-app/src/dbgpt_app/dbgpt_server.py --config /absolute/path/to/config.toml

# Start frontend dev server (port 3000)
cd web && npm install && npm run dev
```

### Code Quality

```bash
# Format code
make fmt

# Check formatting without changes
make fmt-check

# Run tests
make test

# Run doctests
make test-doc

# Type checking
make mypy

# Pre-commit checks (fmt-check + test + test-doc + mypy)
make pre-commit
```

### Build & Publish

```bash
make build          # Build all packages
make publish        # Publish to PyPI
make publish-test   # Publish to TestPyPI
```

## Architecture

### Monorepo Structure (packages/)

The project uses uv workspace with multiple packages:

- **dbgpt-core** (`dbgpt`): Core library with fundamental abstractions
  - `agent/` - Agent framework
  - `core/` - AWEL (Agentic Workflow Expression Language), prompts, operators
  - `model/` - Model abstractions, cluster management, worker system
  - `rag/` - RAG framework (retrieval, knowledge, embedding)
  - `storage/` - Storage abstractions (metadata, vector, graph, cache)
  - `datasource/` - Database connectors (RDBMS base)
  - `cli/` - Command line interface

- **dbgpt-ext** (`dbgpt_ext`): Extensions and integrations
  - Additional datasource connectors (MySQL, PostgreSQL, Oracle, etc.)
  - Vector store implementations (Milvus, Chroma, Weaviate, etc.)
  - RAG extensions

- **dbgpt-serve** (`dbgpt_serve`): Service layer with REST APIs
  - `agent/` - Agent service and app management
  - `rag/` - Knowledge space and document management
  - `conversation/` - Chat history management
  - `datasource/` - Datasource management
  - `flow/` - AWEL flow management
  - `model/` - Model service
  - `file/` - File storage service

- **dbgpt-app** (`dbgpt_app`): Application entry point
  - `dbgpt_server.py` - Main webserver entry
  - `scene/` - Chat scenes (chat_excel, chat_with_db, chat_knowledge, etc.)
  - `knowledge/` - Knowledge management APIs
  - `openapi/` - REST API endpoints

- **dbgpt-client**: Python client library
- **dbgpt-sandbox**: Code execution sandbox

### Key Concepts

1. **AWEL (Agentic Workflow Expression Language)**: DAG-based workflow system in `dbgpt.core.awel`
2. **Component System**: Dependency injection via `dbgpt.component.SystemApp`
3. **Model Cluster**: Distributed model serving with controller/worker architecture
4. **Configuration**: TOML-based config files in `configs/` directory

### Data Directories

- `pilot/` - Runtime data (SQLite DB, model cache, knowledge data)
- `pilot/meta_data/` - Alembic migrations and metadata DB
- `logs/` - Application logs

### Web Frontend

Located in `web/` - Next.js application with Ant Design components.

## Configuration

Configuration files are in `configs/` directory (TOML format):
- `dbgpt-proxy-openai.toml` - OpenAI proxy
- `dbgpt-proxy-siliconflow.toml` - SiliconFlow proxy  
- `dbgpt-proxy-ollama.toml` - Ollama proxy
- `dbgpt-local-*.toml` - Local model configs

Key config sections:
- `[system]` - Language, API keys
- `[service.web]` - Server host/port, database
- `[models]` - LLM, embedding, reranker configurations
- `[rag.storage]` - Vector store settings

## Important Notes

- Python 3.10+ required
- Use absolute paths for config files when running directly
- The `ROOT_PATH` in `dbgpt.configs.model_config` points to project root
- Database migrations handled by Alembic (auto-runs on startup in dev mode)
