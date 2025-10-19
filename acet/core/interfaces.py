"""Abstract interfaces that define ACET toolkit components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import ContextDelta, DeltaStatus, ReflectionReport


class Generator(ABC):
    """Generates task responses while incorporating contextual deltas."""

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: List[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Produce an answer for the given query with injected context."""


class Reflector(ABC):
    """Analyzes generation outputs and proposes new insights."""

    @abstractmethod
    async def reflect(
        self,
        query: str,
        answer: str,
        evidence: List[str],
        context: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectionReport:
        """Produce a structured reflection report for a generation attempt."""

    @abstractmethod
    async def refine(
        self,
        report: ReflectionReport,
        iterations: int = 3,
    ) -> ReflectionReport:
        """Iteratively improve a reflection report."""


class Curator(ABC):
    """Converts reflection insights into curated context deltas."""

    @abstractmethod
    async def curate(
        self,
        report: ReflectionReport,
        existing_deltas: List[ContextDelta],
    ) -> List[ContextDelta]:
        """Create or update deltas based on reflection output."""

    @abstractmethod
    def score_delta(self, delta: ContextDelta) -> float:
        """Compute the composite score for a delta."""

    @abstractmethod
    def deduplicate(
        self,
        candidate: ContextDelta,
        existing: List[ContextDelta],
        threshold: float = 0.90,
    ) -> bool:
        """Return True if the candidate delta is a duplicate of existing ones."""


class StorageBackend(ABC):
    """Persistence layer abstraction for context deltas."""

    @abstractmethod
    async def save_delta(self, delta: ContextDelta) -> None:
        """Persist a single delta."""

    @abstractmethod
    async def save_deltas(self, deltas: List[ContextDelta]) -> None:
        """Persist a batch of deltas."""

    @abstractmethod
    async def get_delta(self, delta_id: str) -> Optional[ContextDelta]:
        """Retrieve a delta by identifier."""

    @abstractmethod
    async def query_deltas(
        self,
        status: Optional[DeltaStatus] = None,
        tags: Optional[List[str]] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ContextDelta]:
        """Query deltas using optional filters."""

    @abstractmethod
    async def update_delta(self, delta: ContextDelta) -> None:
        """Update a persisted delta."""

    @abstractmethod
    async def delete_delta(self, delta_id: str) -> None:
        """Delete a delta by identifier."""

    @abstractmethod
    async def activate_staged(self) -> int:
        """Promote staged deltas to active and return the number activated."""


class EmbeddingProvider(ABC):
    """Embedding generation abstraction used for retrieval tasks."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text snippet."""

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""

    @abstractmethod
    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute similarity between two embeddings."""

