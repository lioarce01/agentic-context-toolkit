# Configuration

ACT is modular by design. You can swap LLM providers, storage backends, embedding sources, and curation policies without touching the orchestration loop. This guide explains the core configuration objects and how to assemble a working pipeline.

## Core Settings (`ACTConfig`)

`ACTConfig` centralises all knobs that affect token budgets, batching, ranking, and reflection behaviour.

```python
from act.core.models import ACTConfig

config = ACTConfig(
    token_budget=800,
    batch_size=2,
    max_epochs=3,
    similarity_weight=0.5,
    recency_weight=0.2,
    usage_weight=0.2,
    risk_penalty_weight=0.1,
    reflection_sample_rate=0.4,
)
```

Key fields:

- **`token_budget`** – Maximum tokens reserved for context delta bullets.  
- **`batch_size`** – How many candidate deltas are processed in each curation batch.  
- **`similarity_*` weights** – Blend semantic, recency, usage, and risk signals in the ranker.  
- **`reflection_sample_rate`** – Probability that online generations trigger a reflection pass.

## Token Budget Management

Use `act.core.budget.TokenBudgetManager` to understand how many deltas fit into a model-specific context window.

```python
from act.core.budget import TokenBudgetManager

manager = TokenBudgetManager(model="gpt-4o", budget=1200)
bullets, tokens_used = manager.pack_deltas(deltas)
```

The budget manager automatically falls back to the `cl100k_base` encoding when a specific tokenizer is unavailable. Keep an eye on the `tokens_used` metric in your logs; it indicates how much context the generator actually consumed.

## Storage Backends

ACT ships with three built-in implementations of `StorageBackend`:

| Backend | Module | When to use |
| --- | --- | --- |
| In-memory | `act.storage.memory.MemoryBackend` | Fast local experiments; resets on restart. |
| SQLite | `act.storage.sqlite.SQLiteBackend` | Zero-config persistence on a single machine. |
| Postgres + pgvector | `act.storage.postgres.PostgresBackend` | Production deployments that need horizontal scale and vector search. |

Choose a backend that matches your deployment constraints. All backends expose the same async CRUD interface.

## Embeddings and Ranking

`act.retrieval.DeltaRanker` combines semantic similarity with usage, recency, and risk signals. To plug-in a custom embedding strategy, implement `act.core.interfaces.EmbeddingProvider`:

```python
from act.core.interfaces import EmbeddingProvider

class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.client = SentenceTransformer(model_name)

    async def embed(self, text: str) -> list[float]:
        return self.client.encode(text).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.client.encode(t).tolist() for t in texts]

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        return cosine_similarity([emb1], [emb2])[0][0]
```

Once an embedding provider exists you can initialise the ranker:

```python
from act.retrieval import DeltaRanker

ranker = DeltaRanker(embedding_provider=SentenceTransformerProvider())
```

## Logging

Every major component uses `structlog`. Configure it early in your application entrypoint:

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
```

With this configuration, ACT emits structured JSON logs that capture context token counts, curated delta IDs, reflection stats, and error diagnostics.

Continue to {doc}`offline-online-adaptation` to learn how these pieces cooperate inside the adaptation loops.
