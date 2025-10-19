"""Unit tests for DeltaRanker composite scoring."""

from __future__ import annotations

import pytest

from acet.core.interfaces import EmbeddingProvider
from acet.core.models import ContextDelta
from acet.retrieval.ranker import DeltaRanker


class _RankerEmbedder(EmbeddingProvider):
    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    async def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = [float(len(text))]
        return self._cache[text]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        return 1.0 / (1.0 + abs(emb1[0] - emb2[0]))


@pytest.mark.asyncio
async def test_ranker_prioritizes_high_similarity_and_low_risk() -> None:
    embedder = _RankerEmbedder()
    ranker = DeltaRanker(embedder)

    query = "Follow safety policies exactly."
    high_match = ContextDelta(
        topic="safety",
        guideline=query,
        usage_count=5,
        recency=0.9,
        risk_level="low",
    )
    weak_match = ContextDelta(
        topic="safety",
        guideline="Unrelated guideline text.",
        usage_count=50,
        recency=1.0,
        risk_level="low",
    )
    risky = ContextDelta(
        topic="safety",
        guideline=query,
        usage_count=5,
        recency=0.9,
        risk_level="high",
    )

    ranked = await ranker.rank(query, [weak_match, risky, high_match])

    # High similarity should beat weaker matches even if other signals are stronger.
    assert ranked[0][0] is high_match
    scores = {id(delta): score for delta, score in ranked}
    assert scores[id(high_match)] > scores[id(weak_match)]
    # Identical deltas that differ only by risk should reflect the penalty.
    assert scores[id(high_match)] > scores[id(risky)]
