[![CI](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/ci.yml) [![Docs](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/docs.yml/badge.svg)](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/docs.yml) [![Publish](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/publish.yml/badge.svg)](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/publish.yml)
# Agentic Context Engineering Toolkit

Research-oriented framework for Agentic Context Engineering. It captures, ranks, and reuses "context deltas" from LLM interactions so agents adapt without retraining, following the methodology described in [Agentic Context Engineering Framework](https://www.arxiv.org/abs/2510.04618).

## Features
- LLM provider agnostic (OpenAI, Anthropic, LiteLLM, Ollama, custom wrappers)
- Storage backend agnostic (memory, SQLite, Postgres/pgvector, extensible interfaces)
- Token budget management, retrieval & ranking, reflection, and curation pipelines
- Ready for Python 3.12 with strict typing, async workflows, and modern tooling

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project Layout
```
.
  acet/               # Library source (packages added per phase)
  benchmarks/         # Performance and benchmark suites
  docs/               # Documentation site sources
  examples/           # Usage examples and sample apps
  tests/              # Unit, integration, and benchmark tests
```

## Development Workflow
1. Create/activate the local virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run format and lint checks: `black .` and `ruff check`.
4. Run type checks: `mypy --strict .`.
5. Run tests: `pytest --cov=acet`.





