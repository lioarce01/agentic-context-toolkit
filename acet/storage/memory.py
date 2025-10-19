"""In-memory storage backend implementation."""

from __future__ import annotations

from typing import Dict, List, Optional

from acet.core.interfaces import StorageBackend
from acet.core.models import ContextDelta, DeltaStatus


class MemoryBackend(StorageBackend):
    """Simple in-memory storage suitable for testing and prototyping."""

    def __init__(self) -> None:
        self._deltas: Dict[str, ContextDelta] = {}

    async def save_delta(self, delta: ContextDelta) -> None:
        self._deltas[delta.id] = delta

    async def save_deltas(self, deltas: List[ContextDelta]) -> None:
        for delta in deltas:
            self._deltas[delta.id] = delta

    async def get_delta(self, delta_id: str) -> Optional[ContextDelta]:
        return self._deltas.get(delta_id)

    async def query_deltas(
        self,
        status: Optional[DeltaStatus] = None,
        tags: Optional[List[str]] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ContextDelta]:
        results = list(self._deltas.values())

        if status is not None:
            results = [delta for delta in results if delta.status == status]

        if tags:
            results = [
                delta
                for delta in results
                if any(tag in delta.tags for tag in tags)
            ]

        if topic is not None:
            results = [delta for delta in results if delta.topic == topic]

        if limit is not None:
            results = results[:limit]

        return results

    async def update_delta(self, delta: ContextDelta) -> None:
        if delta.id in self._deltas:
            self._deltas[delta.id] = delta

    async def delete_delta(self, delta_id: str) -> None:
        self._deltas.pop(delta_id, None)

    async def activate_staged(self) -> int:
        activated = 0
        for delta in self._deltas.values():
            if delta.status == DeltaStatus.STAGED:
                delta.status = DeltaStatus.ACTIVE
                activated += 1
        return activated
