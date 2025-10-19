"""Unit tests covering ACETEngine adaptation paths."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from acet.core.interfaces import Curator, EmbeddingProvider, Generator, Reflector
from acet.core.models import ACETConfig, ContextDelta, DeltaStatus, ReflectionReport
from acet.curators.standard import StandardCurator  # noqa: F401  # imported for type hints
from acet.engine import ACETEngine
from acet.retrieval.ranker import DeltaRanker
from acet.storage.memory import MemoryBackend


class _StubGenerator(Generator):
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def generate(self, query: str, context: List[str], **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"query": query, "context": context})
        return {"answer": "response", "evidence": ["doc"], "metadata": {}}


class _StubReflector(Reflector):
    async def reflect(
        self,
        query: str,
        answer: str,
        evidence: List[str],
        context: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectionReport:
        return ReflectionReport(
            question=query,
            answer=answer,
            evidence_refs=evidence,
            proposed_insights=[
                ReflectionReport.ProposedInsight(
                    topic="policy",
                    guideline="Follow the policy.",
                    confidence=0.9,
                )
            ],
        )

    async def refine(self, report: ReflectionReport, iterations: int = 3) -> ReflectionReport:
        return report


class _StubCurator(Curator):
    async def curate(
        self,
        report: ReflectionReport,
        existing_deltas: List[ContextDelta],
    ) -> List[ContextDelta]:
        return [
            ContextDelta(
                topic="policy",
                guideline="Follow the policy.",
                confidence=0.9,
            )
        ]

    def score_delta(self, delta: ContextDelta) -> float:
        return delta.confidence

    def deduplicate(
        self,
        candidate: ContextDelta,
        existing: List[ContextDelta],
        threshold: float = 0.90,
    ) -> bool:
        return False


class _StubEmbedder(EmbeddingProvider):
    async def embed(self, text: str) -> List[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        return 1.0 / (1.0 + abs(emb1[0] - emb2[0]))


@pytest.mark.asyncio
async def test_run_offline_adaptation_creates_and_activates_deltas() -> None:
    generator = _StubGenerator()
    reflector = _StubReflector()
    curator = _StubCurator()
    storage = MemoryBackend()
    ranker = DeltaRanker(_StubEmbedder())
    config = ACETConfig(reflection_sample_rate=1.0, max_epochs=1)
    engine = ACETEngine(generator, reflector, curator, storage, ranker, config=config)

    training_data = [
        {"query": f"Sample {i}", "ground_truth": "Follow rules."}
        for i in range(25)
    ]
    training_data.append({"note": "missing query"})

    stats = await engine.run_offline_adaptation(training_data=training_data)

    assert stats["deltas_created"] == 25
    assert stats["deltas_activated"] == 25
    assert stats["reflections_ran"] == 25
    active = await storage.query_deltas()
    assert len(active) == 25
    assert generator.calls[0]["context"] == []


@pytest.mark.asyncio
async def test_run_online_and_ingest_updates_usage() -> None:
    generator = _StubGenerator()
    reflector = _StubReflector()
    curator = _StubCurator()
    storage = MemoryBackend()
    ranker = DeltaRanker(_StubEmbedder())
    engine = ACETEngine(generator, reflector, curator, storage, ranker, config=ACETConfig())

    active_delta = ContextDelta(topic="policy", guideline="Use policy.", status=DeltaStatus.ACTIVE)
    await storage.save_delta(active_delta)

    result = await engine.run_online_adaptation("Need guidance?", update_context=True)

    assert result["injected_context"]
    assert result["metadata"]["active_deltas_count"] == 1

    # Ingest interaction should update usage counts.
    await engine.ingest_interaction(
        query="Need guidance?",
        answer="Here",
        evidence=["doc"],
        context_deltas=[active_delta],
    )

    updated = await storage.get_delta(active_delta.id)
    assert updated is not None
    assert updated.usage_count == 1


def test_should_reflect_thresholds() -> None:
    engine = ACETEngine(
        generator=_StubGenerator(),
        reflector=_StubReflector(),
        curator=_StubCurator(),
        storage=MemoryBackend(),
        ranker=DeltaRanker(_StubEmbedder()),
        config=ACETConfig(reflection_sample_rate=0.0),
    )
    assert engine._should_reflect() is False

    engine.config.reflection_sample_rate = 1.0
    assert engine._should_reflect() is True
