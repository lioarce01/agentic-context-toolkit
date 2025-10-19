# API Reference

This section summarises the primary modules in the Agentic Context Toolkit. Refer to the docstrings for full signatures; everything ships with type hints and strict mypy annotations.

## Core Models & Interfaces

| Module | Description |
| --- | --- |
| `acet.core.models` | Data structures such as `ContextDelta`, `ReflectionReport`, and `ACTConfig`. |
| `acet.core.interfaces` | Abstract base classes for generators, reflectors, curators, storage backends, and embedding providers. |
| `acet.core.budget` | `TokenBudgetManager` for counting tokens and packing deltas. |

### `ContextDelta`

```python
ContextDelta(
    topic: str,
    guideline: str,
    conditions: list[str] = ...,
    evidence: list[str] = ...,
    tags: list[str] = ...,
    score: float = 0.0,
    recency: float = 1.0,
    usage_count: int = 0,
    confidence: float = 0.0,
)
```

Represents a single actionable insight. Deltas flow through the lifecycle `STAGED → ACTIVE → ARCHIVED`.

## Engine

| Object | Description |
| --- | --- |
| `acet.engine.ACETEngine` | Orchestrates generator → reflector → curator → storage → ranker. |
| `ACETEngine.run_offline_adaptation(...)` | Bulk-train from labelled transcripts. |
| `ACETEngine.run_online_adaptation(...)` | Handle a single live request. |
| `ACETEngine.ingest_interaction(...)` | Persist an externally generated interaction (useful for LangChain memory). |

## Retrieval & Ranking

| Module | Description |
| --- | --- |
| `acet.retrieval.ranker.DeltaRanker` | Combines embeddings, recency, usage, and risk. |
| `acet.retrieval.dedup.DeltaDeduplicator` | Detect and merge near-duplicate deltas. |

## Storage

All storage backends implement `acet.core.interfaces.StorageBackend`:

- `acet.storage.memory.MemoryBackend`
- `acet.storage.sqlite.SQLiteBackend`
- `acet.storage.postgres.PostgresBackend`

## LLM Providers

`acet.llm.base.BaseLLMProvider` defines the required interface. Built-in implementations include:

- `acet.llm.providers.OpenAIProvider`
- `acet.llm.providers.AnthropicProvider`
- `acet.llm.providers.LiteLLMProvider`

Each provider records model usage metadata and exposes `count_tokens` for budgeting.

## Generators, Reflectors, Curators

| Module | Description |
| --- | --- |
| `acet.generators.llm.LLMGenerator` | Injects context bullets into a system prompt and calls the chosen LLM provider. |
| `acet.reflectors.llm` | Implements JSON-mode reflection loops (placeholder until fully implemented). |
| `acet.curators.standard.StandardCurator` | Scores candidate deltas, deduplicates them, and stages accepted ones. |

## Integrations & Agents

- `acet.integrations.ACTMemory` – LangChain-compatible memory component.  
- `acet.agents.react.ReActAgent` – Async ReAct loop with tool support.  

See {doc}`../guides/integrations` for usage guidance.
