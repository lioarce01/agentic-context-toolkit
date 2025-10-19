
"""Context delta ranking utilities."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from acet.core.interfaces import EmbeddingProvider
from acet.core.models import ContextDelta


class DeltaRanker:
    """Ranks context deltas according to similarity, recency, usage, and risk."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        similarity_weight: float = 0.45,
        recency_weight: float = 0.25,
        usage_weight: float = 0.20,
        risk_penalty_weight: float = 0.10,
    ) -> None:
        self.embedder = embedding_provider
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.usage_weight = usage_weight
        self.risk_penalty_weight = risk_penalty_weight

    async def rank(
        self,
        query: str,
        candidates: Sequence[ContextDelta],
        top_k: int = 20,
    ) -> List[Tuple[ContextDelta, float]]:
        """Return the top-k deltas sorted by relevance score."""
        if not candidates:
            return []

        query_embedding = await self.embedder.embed(query)
        scored: List[Tuple[ContextDelta, float]] = []
        for delta in candidates:
            if delta.embedding is None:
                delta.embedding = await self.embedder.embed(delta.guideline)
            if delta.embedding is None:
                # Embedding provider failed; treat as low relevance.
                score = 0.0
            else:
                score = self._compute_score(query_embedding, delta.embedding, delta)
            scored.append((delta, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None and top_k > 0:
            return scored[:top_k]
        return scored

    def _compute_score(
        self,
        query_embedding: List[float],
        delta_embedding: List[float],
        delta: ContextDelta,
    ) -> float:
        """Composite scoring function used for ranking."""
        similarity = self.embedder.similarity(query_embedding, delta_embedding)
        recency = self._clamp(delta.recency)
        usage = self._usage_term(delta.usage_count)
        risk_penalty = self._risk_penalty(delta.risk_level)

        score = (
            self.similarity_weight * similarity
            + self.recency_weight * recency
            + self.usage_weight * usage
            - self.risk_penalty_weight * risk_penalty
        )
        return self._clamp(score)

    def _usage_term(self, usage_count: int) -> float:
        """Normalize usage counts with diminishing returns."""
        if usage_count <= 0:
            return 0.0
        return min(1.0, math.log1p(usage_count) / 10.0)

    def _risk_penalty(self, risk_level: str) -> float:
        """Translate risk level into a penalty amount."""
        mapping = {"low": 0.0, "medium": 0.2, "high": 0.5}
        return mapping.get(risk_level, 0.0)

    @staticmethod
    def _clamp(value: Optional[float]) -> float:
        """Clamp values into the [0, 1] interval."""
        if value is None:
            return 0.0
        return max(0.0, min(1.0, value))
