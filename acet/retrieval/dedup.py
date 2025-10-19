"""Utilities for semantic deduplication of context deltas."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from acet.core.interfaces import EmbeddingProvider
from acet.core.models import ContextDelta


class DeltaDeduplicator:
    """Semantic deduplication helper for context deltas."""

    def __init__(self, embedding_provider: EmbeddingProvider, threshold: float = 0.90) -> None:
        self.embedder = embedding_provider
        self.threshold = threshold

    async def is_duplicate(
        self,
        candidate: ContextDelta,
        existing: List[ContextDelta],
    ) -> Tuple[bool, Optional[ContextDelta]]:
        """Return whether the candidate is a duplicate and its closest match."""
        if not existing:
            return False, None

        if candidate.embedding is None:
            candidate.embedding = await self.embedder.embed(candidate.guideline)

        best_similarity = -1.0
        best_delta: Optional[ContextDelta] = None

        for delta in existing:
            if delta.embedding is None:
                delta.embedding = await self.embedder.embed(delta.guideline)
            similarity = self.embedder.similarity(candidate.embedding, delta.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_delta = delta

        return best_similarity >= self.threshold, best_delta

    async def merge_duplicates(
        self,
        candidate: ContextDelta,
        existing: ContextDelta,
    ) -> ContextDelta:
        """Merge duplicate deltas together, updating metadata appropriately."""
        existing_usage = existing.usage_count
        candidate_usage = max(candidate.usage_count, 1)

        existing.usage_count = existing_usage + candidate_usage
        existing.helpful_count += candidate.helpful_count
        existing.harmful_count += candidate.harmful_count

        existing.evidence = list({*existing.evidence, *candidate.evidence})
        existing.tags = list({*existing.tags, *candidate.tags})
        existing.conditions = list({*existing.conditions, *candidate.conditions})

        total_usage = existing_usage + candidate_usage
        if total_usage > 0:
            confidence = (
                (existing.confidence * existing_usage)
                + (candidate.confidence * candidate_usage)
            ) / total_usage
            existing.confidence = max(0.0, min(1.0, confidence))

        existing.version += 1
        existing.updated_at = datetime.utcnow()

        return existing
