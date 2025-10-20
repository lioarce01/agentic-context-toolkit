[![CI](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/ci.yml) [![Docs](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/docs.yml/badge.svg)](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/docs.yml) [![Publish](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/publish.yml/badge.svg)](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/publish.yml) [![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/lioarce01/agentic-context-toolkit/actions/workflows/ci.yml)
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

## Performance Snapshot
- **Delta retrieval (250 active deltas)**: ~2 ms mean latency (`tests/benchmarks/test_delta_retrieval.py`)
- **SQLite save/query (300 staged deltas)**: ~23 ms mean latency (`tests/benchmarks/test_storage_throughput.py`)
- **Curator dedup (300 proposed insights, 30% duplicates)**: ~140 ms mean latency (`tests/benchmarks/test_curator_throughput.py`)

All benchmarks are reproducible via the CLI harnesses under `benchmarks/`. For example:
```bash
python benchmarks/delta_retrieval.py --iterations 30 --plot benchmarks/artifacts/delta_latency.png
python benchmarks/storage_throughput.py --backend all --iterations 30 --plot benchmarks/artifacts/storage_latency.png
python benchmarks/curator_throughput.py --proposals 300 --duplicate-ratio 0.3 --iterations 20 --plot benchmarks/artifacts/curator_latency.png
```
Adjust the parameters or swap in your production embeddings/backends to profile your deployment.





