"""Integration coverage for the asynchronous SQLite backend."""

from __future__ import annotations

import pytest

from acet.core.models import ContextDelta, DeltaStatus
from acet.storage.sqlite import SQLiteBackend


@pytest.mark.asyncio
async def test_sqlite_backend_crud_flow(tmp_path) -> None:
    db_path = tmp_path / "test_deltas.db"
    backend = SQLiteBackend(str(db_path))

    delta = ContextDelta(topic="safety", guideline="Comply with safety policies.")
    await backend.save_delta(delta)

    stored = await backend.get_delta(delta.id)
    assert stored is not None
    assert stored.guideline == delta.guideline
    assert stored.status == DeltaStatus.STAGED

    stored.guideline = "Enforce safety guardrails."
    await backend.update_delta(stored)

    updated = await backend.get_delta(delta.id)
    assert updated is not None
    assert updated.guideline == "Enforce safety guardrails."

    fetched = await backend.query_deltas(status=DeltaStatus.STAGED)
    assert len(fetched) == 1
    assert fetched[0].id == delta.id

    activated = await backend.activate_staged()
    assert activated == 1

    active_delta = await backend.get_delta(delta.id)
    assert active_delta is not None
    assert active_delta.status == DeltaStatus.ACTIVE

    await backend.delete_delta(delta.id)
    assert await backend.get_delta(delta.id) is None
