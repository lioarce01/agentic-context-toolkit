"""Unit tests for the StandardCurator pipeline."""

from __future__ import annotations

import pytest

from acet.core.interfaces import EmbeddingProvider
from acet.core.models import ContextDelta, ReflectionReport
from acet.curators.standard import StandardCurator


class _CuratorEmbedder(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        difference = abs(emb1[0] - emb2[0])
        return max(0.0, 1.0 - difference / 5.0)


@pytest.mark.asyncio
async def test_curator_filters_low_confidence_and_scores_delta() -> None:
    curator = StandardCurator(_CuratorEmbedder(), min_confidence=0.6)
    report = ReflectionReport(
        question="How was the answer?",
        answer="Great.",
        proposed_insights=[
            ReflectionReport.ProposedInsight(
                topic="tone",
                guideline="Respond with empathy.",
                conditions=["user expresses frustration"],
                evidence=["transcript-42"],
                confidence=0.8,
            ),
            ReflectionReport.ProposedInsight(
                topic="tone",
                guideline="Add emoji everywhere.",
                confidence=0.4,
            ),
        ],
    )

    curated = await curator.curate(report, existing_deltas=[])

    assert len(curated) == 1
    delta = curated[0]
    assert delta.topic == "tone"
    assert str(delta.status) == "staged"
    assert pytest.approx(delta.score, rel=1e-3) == min(0.8 * 1.1 * 1.05, 1.0)


@pytest.mark.asyncio
async def test_curator_deduplicates_against_existing_deltas() -> None:
    curator = StandardCurator(_CuratorEmbedder(), min_confidence=0.6, dedup_threshold=0.9)
    existing = [
        ContextDelta(
            topic="formatting",
            guideline="Summarize key points first.",
            evidence=["guideline-v1"],
            confidence=0.7,
        )
    ]
    previous_version = existing[0].version

    report = ReflectionReport(
        question="How was the answer?",
        answer="It was fine.",
        proposed_insights=[
            ReflectionReport.ProposedInsight(
                topic="formatting",
                guideline="Summarize key points first!",
                evidence=["guideline-v2"],
                confidence=0.9,
            )
        ],
    )

    curated = await curator.curate(report, existing_deltas=existing)

    assert curated == []  # merged into existing delta
    assert existing[0].version == previous_version + 1
    assert set(existing[0].evidence) == {"guideline-v1", "guideline-v2"}
