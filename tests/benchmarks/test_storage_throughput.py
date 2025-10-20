"""Benchmark persistence throughput across storage backends."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import pytest

from acet.core.interfaces import StorageBackend
from acet.core.models import ContextDelta, DeltaStatus
from acet.storage.memory import MemoryBackend
from acet.storage.sqlite import SQLiteBackend

if TYPE_CHECKING:
    BenchmarkFixture = Any


def _generate_deltas(count: int) -> List[ContextDelta]:
    return [
        ContextDelta(
            id=str(uuid.uuid4()),
            topic=f"topic-{index % 5}",
            guideline=f"Guideline #{index}",
            conditions=[f"condition-{index % 3}"],
            evidence=[f"doc-{index}-{i}" for i in range(2)],
            tags=[f"tag-{index % 7}"],
            status=DeltaStatus.STAGED,
            confidence=0.7,
        )
        for index in range(count)
    ]


async def _exercise_backend(backend: StorageBackend, deltas: List[ContextDelta]) -> None:
    await backend.save_deltas(deltas)
    saved = await backend.query_deltas()
    assert len(saved) == len(deltas)

    await backend.activate_staged()
    active = await backend.query_deltas(status=DeltaStatus.ACTIVE)
    assert len(active) == len(deltas)


@pytest.mark.parametrize("backend_name", ["memory", "sqlite"], ids=["memory", "sqlite"])
@pytest.mark.benchmark(group="storage-throughput")
def test_storage_save_query_throughput(
    benchmark: BenchmarkFixture,
    backend_name: str,
    tmp_path: Path,
) -> None:
    """Ensure storage backends keep save/query latency within budget."""
    deltas = _generate_deltas(300)

    def _iteration() -> None:
        backend: StorageBackend
        if backend_name == "memory":
            backend = MemoryBackend()
        else:
            db_file = tmp_path / f"deltas-{uuid.uuid4().hex}.db"
            backend = SQLiteBackend(db_path=str(db_file))

        asyncio.run(_exercise_backend(backend, deltas))

    benchmark(_iteration)
    assert benchmark.stats.stats.mean < 0.25, f"{backend_name} backend exceeded 250 ms budget"
