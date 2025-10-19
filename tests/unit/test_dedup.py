"""Unit tests for DeltaDeduplicator behaviour."""

from __future__ import annotations

import pytest

from acet.core.interfaces import EmbeddingProvider
from acet.core.models import ContextDelta
from acet.retrieval.dedup import DeltaDeduplicator


class _StubEmbedder(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        difference = abs(emb1[0] - emb2[0])
        return max(0.0, 1.0 - difference / 10.0)


@pytest.mark.asyncio
async def test_is_duplicate_detects_near_match() -> None:
    embedder = _StubEmbedder()
    dedup = DeltaDeduplicator(embedder, threshold=0.9)

    existing = ContextDelta(topic="safety", guideline="Keep users safe.")
    candidate = ContextDelta(topic="safety", guideline="Keep users safe!")

    is_dup, matched = await dedup.is_duplicate(candidate, [existing])

    assert is_dup is True
    assert matched is existing
    assert candidate.embedding is not None
    assert existing.embedding is not None


@pytest.mark.asyncio
async def test_merge_duplicates_accumulates_metadata() -> None:
    embedder = _StubEmbedder()
    dedup = DeltaDeduplicator(embedder)

    existing = ContextDelta(
        topic="tone",
        guideline="Respond positively.",
        usage_count=2,
        helpful_count=1,
        evidence=["example_a"],
        tags=["positive"],
        confidence=0.6,
    )
    candidate = ContextDelta(
        topic="tone",
        guideline="Respond positively!",
        usage_count=0,
        helpful_count=0,
        evidence=["example_b"],
        tags=["positive", "friendly"],
        confidence=0.9,
    )

    previous_version = existing.version
    previous_updated = existing.updated_at

    merged = await dedup.merge_duplicates(candidate, existing)

    assert merged.usage_count == 3
    assert set(merged.evidence) == {"example_a", "example_b"}
    assert set(merged.tags) == {"positive", "friendly"}
    assert merged.version == previous_version + 1
    assert merged.helpful_count == 1
    assert merged.updated_at > previous_updated
    assert pytest.approx(merged.confidence, rel=1e-3) == 0.7
