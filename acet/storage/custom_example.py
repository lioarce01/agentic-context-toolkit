"""Example custom storage backend implementation."""

from __future__ import annotations

import importlib
from typing import Any, List, Optional

try:
    redis_asyncio: Any | None = importlib.import_module("redis.asyncio")
except ImportError as exc:  # pragma: no cover - optional dependency
    redis_asyncio = None
    REDIS_IMPORT_ERROR: ImportError | None = exc
else:
    REDIS_IMPORT_ERROR = None

from acet.core.interfaces import StorageBackend
from acet.core.models import ContextDelta, DeltaStatus


class RedisBackend(StorageBackend):
    """Demonstrates how to extend StorageBackend using Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        if redis_asyncio is None:
            raise ImportError(
                "redis is required for RedisBackend. Install with `pip install redis`."
            ) from REDIS_IMPORT_ERROR
        self.client = redis_asyncio.from_url(redis_url, decode_responses=True)
        self.prefix = "ace:delta:"

    async def save_delta(self, delta: ContextDelta) -> None:
        await self.client.set(self._key(delta.id), delta.model_dump_json())

    async def save_deltas(self, deltas: List[ContextDelta]) -> None:
        if not deltas:
            return
        pipeline = self.client.pipeline()
        for delta in deltas:
            pipeline.set(self._key(delta.id), delta.model_dump_json())
        await pipeline.execute()

    async def get_delta(self, delta_id: str) -> Optional[ContextDelta]:
        payload = await self.client.get(self._key(delta_id))
        return ContextDelta.model_validate_json(payload) if payload else None

    async def query_deltas(
        self,
        status: Optional[DeltaStatus] = None,
        tags: Optional[List[str]] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ContextDelta]:
        pattern = f"{self.prefix}*"
        keys = [key async for key in self.client.scan_iter(pattern)]
        results: List[ContextDelta] = []
        for key in keys:
            payload = await self.client.get(key)
            if not payload:
                continue
            delta = ContextDelta.model_validate_json(payload)
            if status is not None and delta.status != status:
                continue
            if topic is not None and delta.topic != topic:
                continue
            if tags and not any(tag in delta.tags for tag in tags):
                continue
            results.append(delta)
            if limit is not None and len(results) >= limit:
                break
        return results

    async def update_delta(self, delta: ContextDelta) -> None:
        await self.save_delta(delta)

    async def delete_delta(self, delta_id: str) -> None:
        await self.client.delete(self._key(delta_id))

    async def activate_staged(self) -> int:
        deltas = await self.query_deltas(status=DeltaStatus.STAGED)
        for delta in deltas:
            delta.status = DeltaStatus.ACTIVE
            await self.save_delta(delta)
        return len(deltas)

    def _key(self, delta_id: str) -> str:
        return f"{self.prefix}{delta_id}"
