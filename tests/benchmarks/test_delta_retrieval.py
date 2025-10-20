"""Benchmark for the core delta retrieval + ranking loop."""

from __future__ import annotations

import asyncio
import math
import random
from typing import TYPE_CHECKING, Any, List

import pytest

from acet.core.interfaces import EmbeddingProvider
from acet.core.models import ContextDelta, DeltaStatus
from acet.retrieval import DeltaRanker

if TYPE_CHECKING:
    BenchmarkFixture = Any


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for benchmarking."""

    async def embed(self, text: str) -> List[float]:
        return self._vectorize(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(text) for text in texts]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        if not emb1 or not emb2:
            return 0.0
        # Simple cosine similarity for the synthetic vectors.
        dot = sum(a * b for a, b in zip(emb1, emb2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def _vectorize(text: str) -> List[float]:
        # Encode characters deterministically so benchmarks remain stable.
        return [float(ord(char) % 97) / 50.0 for char in text[:32]]


def _make_delta(index: int) -> ContextDelta:
    random.seed(index)
    return ContextDelta(
        topic=f"topic-{index % 5}",
        guideline=f"Guideline #{index}: respond to case {index}",
        conditions=[f"condition-{index % 3}"],
        evidence=[f"doc-{index}-{i}" for i in range(2)],
        tags=[f"tag-{index % 7}"],
        status=DeltaStatus.ACTIVE,
        usage_count=random.randint(0, 50),
        recency=random.random(),
        confidence=0.6 + (index % 4) * 0.05,
        risk_level=random.choice(["low", "medium"]),
    )


@pytest.mark.benchmark(group="delta-retrieval")
def test_delta_ranker_latency_under_budget(benchmark: BenchmarkFixture) -> None:
    """Ensure ranking stays comfortably within the 100 ms retrieval target."""
    embedding_provider = StubEmbeddingProvider()
    ranker = DeltaRanker(embedding_provider=embedding_provider)

    # Simulate a moderately sized active delta set.
    active_deltas = [_make_delta(i) for i in range(250)]
    query = "How should I handle refund escalations for enterprise customers?"

    async def _run_rank() -> None:
        await ranker.rank(query, active_deltas, top_k=25)

    def _invoke() -> None:
        # Reset embeddings to force ranking work on every iteration.
        for delta in active_deltas:
            delta.embedding = None
        asyncio.run(_run_rank())

    benchmark(_invoke)
    assert benchmark.stats.stats.mean < 0.1, "Delta ranking exceeded 100 ms budget"
