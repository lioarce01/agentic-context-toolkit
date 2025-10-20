"""Benchmark curator throughput under heavy proposed insight loads."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, List

import pytest

from acet.core.interfaces import EmbeddingProvider
from acet.core.models import ContextDelta, ReflectionReport
from acet.curators.standard import StandardCurator

if TYPE_CHECKING:
    BenchmarkFixture = Any


class StubEmbeddingProvider(EmbeddingProvider):
    """Lightweight embedding provider returning deterministic vectors."""

    async def embed(self, text: str) -> List[float]:
        return [float(ord(ch) % 101) / 50.0 for ch in text[:32]]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        if not emb1 or not emb2:
            return 0.0
        dot = sum(a * b for a, b in zip(emb1, emb2, strict=False))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(dot / (norm1 * norm2))


def _build_existing(count: int) -> List[ContextDelta]:
    return [
        ContextDelta(
            id=str(uuid.uuid4()),
            topic=f"existing-topic-{index % 5}",
            guideline=f"Existing guideline #{index}",
            evidence=[f"doc-{index}"],
            tags=[f"tag-{index % 3}"],
            confidence=0.8,
        )
        for index in range(count)
    ]


def _build_report(proposals: int, duplicate_ratio: float = 0.3) -> ReflectionReport:
    proposed = []
    for index in range(proposals):
        if index < proposals * duplicate_ratio:
            topic = f"existing-topic-{index % 5}"
            guideline = f"Existing guideline #{index % 10}"
        else:
            topic = f"new-topic-{index % 7}"
            guideline = f"New guideline #{index}"

        proposed.append(
            ReflectionReport.ProposedInsight(
                topic=topic,
                guideline=guideline,
                conditions=[f"condition-{index % 3}"],
                evidence=[f"evidence-{index % 4}"],
                tags=[f"tag-{index % 5}"],
                confidence=0.7,
            )
        )

    return ReflectionReport(
        question="benchmark",
        answer="benchmark",
        proposed_insights=proposed,
    )


@pytest.mark.benchmark(group="curator-throughput")
def test_curator_dedup_throughput(benchmark: BenchmarkFixture) -> None:
    """Ensure curator handles heavy batches within acceptable latency."""
    existing = _build_existing(150)
    report = _build_report(300)

    embedding = StubEmbeddingProvider()
    curator = StandardCurator(embedding_provider=embedding, min_confidence=0.0)

    async def _run() -> None:
        await curator.curate(report, existing)

    def _iteration() -> None:
        asyncio.run(_run())

    benchmark(_iteration)
    assert benchmark.stats.stats.mean < 0.18, "Curator throughput exceeded 180 ms budget"
