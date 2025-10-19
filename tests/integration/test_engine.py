"""Integration tests for ACETEngine orchestration."""

from __future__ import annotations

import pytest

from acet.core.interfaces import EmbeddingProvider, Generator, Reflector
from acet.core.models import DeltaStatus, ReflectionReport
from acet.curators.standard import StandardCurator
from acet.engine import ACETEngine
from acet.retrieval.ranker import DeltaRanker
from acet.storage.memory import MemoryBackend


class _EngineEmbedder(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        return 1.0 / (1.0 + abs(emb1[0] - emb2[0]))


class _StubGenerator(Generator):
    async def generate(self, query: str, context: list[str], **_: object) -> dict[str, object]:
        return {
            "answer": "Stubbed answer.",
            "evidence": ["stub-document"],
            "metadata": {"used_context": context},
        }


class _StubReflector(Reflector):
    async def reflect(
        self,
        query: str,
        answer: str,
        evidence: list[str],
        context: list[str],
        ground_truth: str | None = None,
        **_: object,
    ) -> ReflectionReport:
        return ReflectionReport(
            question=query,
            answer=answer,
            evidence_refs=evidence,
            proposed_insights=[
                ReflectionReport.ProposedInsight(
                    topic="safety",
                    guideline="Always verify facts.",
                    evidence=["stub-document"],
                    confidence=0.85,
                )
            ],
        )

    async def refine(self, report: ReflectionReport, iterations: int = 3) -> ReflectionReport:
        return report


@pytest.mark.asyncio
async def test_engine_online_adaptation_persists_curated_delta() -> None:
    embedder = _EngineEmbedder()
    engine = ACETEngine(
        generator=_StubGenerator(),
        reflector=_StubReflector(),
        curator=StandardCurator(embedder),
        storage=MemoryBackend(),
        ranker=DeltaRanker(embedder),
    )

    outcome = await engine.run_online_adaptation("How should we respond?", update_context=True)

    assert outcome["answer"] == "Stubbed answer."
    staged = await engine.storage.query_deltas(status=DeltaStatus.STAGED)
    assert len(staged) == 1
    assert staged[0].guideline == "Always verify facts."
    assert staged[0].confidence == pytest.approx(0.85)
