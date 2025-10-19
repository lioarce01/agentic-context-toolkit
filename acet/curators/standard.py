"""Standard curator implementation."""

from __future__ import annotations

from typing import List

import structlog

from acet.core.interfaces import Curator, EmbeddingProvider
from acet.core.models import ContextDelta, DeltaStatus, ReflectionReport
from acet.retrieval.dedup import DeltaDeduplicator

logger = structlog.get_logger(__name__)


class StandardCurator(Curator):
    """Converts reflection insights into curated context deltas."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        min_confidence: float = 0.55,
        dedup_threshold: float = 0.90,
    ) -> None:
        self.embedder = embedding_provider
        self.min_confidence = min_confidence
        self.deduplicator = DeltaDeduplicator(embedding_provider, threshold=dedup_threshold)

    async def curate(
        self,
        report: ReflectionReport,
        existing_deltas: List[ContextDelta],
    ) -> List[ContextDelta]:
        curated: List[ContextDelta] = []
        for insight in report.proposed_insights:
            confidence = insight.confidence
            if confidence < self.min_confidence:
                logger.debug("Skipping low-confidence insight", confidence=confidence)
                continue

            candidate = ContextDelta(
                topic=insight.topic,
                guideline=insight.guideline,
                conditions=list(insight.conditions),
                evidence=list(insight.evidence),
                tags=list(insight.tags),
                confidence=confidence,
                status=DeltaStatus.STAGED,
            )

            is_dup, similar = await self.deduplicator.is_duplicate(
                candidate, existing_deltas + curated
            )

            if is_dup and similar is not None:
                logger.info("Merging duplicate delta", candidate_id=candidate.id, existing_id=similar.id)
                updated = await self.deduplicator.merge_duplicates(candidate, similar)
                self._replace_delta(existing_deltas, similar.id, updated)
                self._replace_delta(curated, similar.id, updated)
            else:
                candidate.score = self.score_delta(candidate)
                curated.append(candidate)
                logger.info(
                    "Curated new delta",
                    delta_id=candidate.id,
                    topic=candidate.topic,
                    score=candidate.score,
                )

        return curated

    def score_delta(self, delta: ContextDelta) -> float:
        score = delta.confidence
        if delta.evidence:
            score *= 1.1
        if delta.conditions:
            score *= 1.05
        return min(score, 1.0)

    def deduplicate(
        self,
        candidate: ContextDelta,
        existing: List[ContextDelta],
        threshold: float = 0.90,
    ) -> bool:
        if candidate.embedding is None:
            return False
        for delta in existing:
            if delta.embedding is None:
                continue
            similarity = self.embedder.similarity(candidate.embedding, delta.embedding)
            if similarity >= threshold:
                return True
        return False

    @staticmethod
    def _replace_delta(pool: List[ContextDelta], delta_id: str, updated: ContextDelta) -> None:
        for index, delta in enumerate(pool):
            if delta.id == delta_id:
                pool[index] = updated
                break
