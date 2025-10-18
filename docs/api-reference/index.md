# API Reference

This section summarises the primary modules in the Agentic Context Toolkit. Refer to the docstrings for full signatures; everything ships with type hints and strict mypy annotations.

## Core Models & Interfaces

| Module | Description |
| --- | --- |
| `act.core.models` | Data structures such as `ContextDelta`, `ReflectionReport`, and `ACTConfig`. |
| `act.core.interfaces` | Abstract base classes for generators, reflectors, curators, storage backends, and embedding providers. |
| `act.core.budget` | `TokenBudgetManager` for counting tokens and packing deltas. |

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
| `act.engine.ACTEngine` | Orchestrates generator → reflector → curator → storage → ranker. |
| `ACTEngine.run_offline_adaptation(...)` | Bulk-train from labelled transcripts. |
| `ACTEngine.run_online_adaptation(...)` | Handle a single live request. |
| `ACTEngine.ingest_interaction(...)` | Persist an externally generated interaction (useful for LangChain memory). |

## Retrieval & Ranking

| Module | Description |
| --- | --- |
| `act.retrieval.ranker.DeltaRanker` | Combines embeddings, recency, usage, and risk. |
| `act.retrieval.dedup.DeltaDeduplicator` | Detect and merge near-duplicate deltas. |

## Storage

All storage backends implement `act.core.interfaces.StorageBackend`:

- `act.storage.memory.MemoryBackend`
- `act.storage.sqlite.SQLiteBackend`
- `act.storage.postgres.PostgresBackend`

## LLM Providers

`act.llm.base.BaseLLMProvider` defines the required interface. Built-in implementations include:

- `act.llm.providers.OpenAIProvider`
- `act.llm.providers.AnthropicProvider`
- `act.llm.providers.LiteLLMProvider`

Each provider records model usage metadata and exposes `count_tokens` for budgeting.

## Generators, Reflectors, Curators

| Module | Description |
| --- | --- |
| `act.generators.llm.LLMGenerator` | Injects context bullets into a system prompt and calls the chosen LLM provider. |
| `act.reflectors.llm` | Implements JSON-mode reflection loops (placeholder until fully implemented). |
| `act.curators.standard.StandardCurator` | Scores candidate deltas, deduplicates them, and stages accepted ones. |

## Integrations & Agents

- `act.integrations.ACTMemory` – LangChain-compatible memory component.  
- `act.agents.react.ReActAgent` – Async ReAct loop with tool support.  

See {doc}`../guides/integrations` for usage guidance.
