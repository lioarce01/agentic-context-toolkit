# Examples

Hands-on recipes that demonstrate common ACET deployment scenarios.

## Simple RAG Pipeline

`examples/simple_rag.py` shows how to:

- Bootstrap an engine with a toy LLM provider and in-memory storage.
- Prime the context library with a handful of labelled examples.
- Answer a live query while reflection + curation refine the playbook.

Run it with:

```bash
python examples/simple_rag.py
```

Read through the script for guidance on substituting a real LLM provider, persistent storage, and stronger embedding model.

## Multi-LLM Strategy

`examples/multi_llm_example.py` demonstrates how to route between multiple providers (OpenAI, Anthropic, LiteLLM/Ollama) while reusing the same ACET engine and storage backend.

It highlights:

- Creating provider instances with shared defaults.
- Reusing ACET's ranker and curator across heterogeneous models.
- Collecting telemetry (token budgets, curated deltas) for each provider.

```bash
python examples/multi_llm_example.py
```

Both scripts are self-contained and rely only on the dependencies listed in `requirements.txt`. Use them as a starting point before wiring ACET into LangChain, LangGraph, or your own orchestration layer.
