# Examples

Hands-on recipes that demonstrate common ACT deployment scenarios.

## Simple RAG Pipeline

`examples/simple_rag.py` shows how to:

- Configure an inexpensive echo-style LLM provider.
- Ingest a small offline dataset to seed context deltas.
- Serve online queries while the engine reflects/curates in the background.

Run it with:

```bash
python examples/simple_rag.py
```

## Multi-LLM Strategy

`examples/multi_llm_example.py` demonstrates how to route between multiple providers (OpenAI, Anthropic, LiteLLM/Ollama) while reusing the same ACT engine and storage backend.

It highlights:

- Creating provider instances with shared defaults.
- Using ACT’s ranker and curator across heterogeneous models.
- Collecting telemetry (token budgets, curated deltas) for each provider.

```bash
python examples/multi_llm_example.py
```

Both scripts are self-contained and rely only on the dependencies listed in `requirements.txt`. Review them as a starting point for production integrations.
