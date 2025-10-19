"""Unit tests for the in-memory storage backend."""

from __future__ import annotations

import pytest

from acet.core.models import ContextDelta, DeltaStatus
from acet.storage.memory import MemoryBackend


@pytest.mark.asyncio
async def test_memory_backend_crud_and_filters() -> None:
    backend = MemoryBackend()

    delta = ContextDelta(topic="safety", guideline="Follow policies.")
    await backend.save_delta(delta)

    fetched = await backend.get_delta(delta.id)
    assert fetched is not None
    assert fetched.guideline == "Follow policies."

    # Update and filter by tag/topic.
    delta.tags.append("policy")
    await backend.update_delta(delta)

    results = await backend.query_deltas(tags=["policy"])
    assert len(results) == 1
    assert results[0].tags == ["policy"]

    limited = await backend.query_deltas(limit=1)
    assert len(limited) == 1

    none_topic = await backend.query_deltas(topic="other")
    assert none_topic == []

    # Activation should flip status.
    activated = await backend.activate_staged()
    assert activated == 1
    active = await backend.query_deltas(status=DeltaStatus.ACTIVE)
    assert active and active[0].status is DeltaStatus.ACTIVE

    await backend.delete_delta(delta.id)
    assert await backend.get_delta(delta.id) is None
