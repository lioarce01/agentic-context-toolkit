"""Unit tests for the PostgresBackend logic using stubbed SQLAlchemy sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from acet.core.models import ContextDelta, DeltaStatus
from acet.storage import postgres


class _StubSession:
    def __init__(self) -> None:
        self.records: dict[str, postgres.DeltaRecord] = {}
        self.commits: int = 0

    async def merge(self, record: postgres.DeltaRecord) -> None:
        self.records[record.id] = record

    async def commit(self) -> None:
        self.commits += 1

    async def get(self, model: Any, delta_id: str) -> Optional[postgres.DeltaRecord]:
        return self.records.get(delta_id)

    async def scalars(self, stmt: Any) -> Any:
        class _Scalars:
            def __init__(self, values: List[postgres.DeltaRecord]) -> None:
                self._values = values

            def all(self) -> List[postgres.DeltaRecord]:
                return list(self._values)

        return _Scalars(list(self.records.values()))

    async def delete(self, record: postgres.DeltaRecord) -> None:
        self.records.pop(record.id, None)

    async def execute(self, stmt: Any) -> Any:
        activated: List[str] = []
        for record in self.records.values():
            if record.status == DeltaStatus.STAGED.value:
                record.status = DeltaStatus.ACTIVE.value
                activated.append(record.id)

        class _Result:
            def __init__(self, ids: List[str]) -> None:
                self._ids = ids

            def scalars(self) -> Any:
                class _Scalars:
                    def __init__(self, ids: List[str]) -> None:
                        self._ids = ids

                    def all(self) -> List[str]:
                        return list(self._ids)

                return _Scalars(self._ids)

        return _Result(activated)


@dataclass
class _StubSessionContext:
    session: _StubSession

    async def __aenter__(self) -> _StubSession:
        return self.session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _StubSessionmaker:
    def __init__(self, session: _StubSession) -> None:
        self._session = session

    def __call__(self) -> _StubSessionContext:
        return _StubSessionContext(self._session)


@pytest.mark.asyncio
async def test_postgres_backend_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _StubSession()

    class _StubEngine:
        def __init__(self) -> None:
            self.metadata_created = False

        def begin(self) -> Any:
            engine = self

            class _Conn:
                async def __aenter__(self_inner) -> _Conn:
                    return self_inner

                async def __aexit__(self_inner, exc_type: Any, exc: Any, tb: Any) -> None:
                    return None

                async def run_sync(self_inner, func: Any) -> None:
                    engine.metadata_created = True

            return _Conn()

    monkeypatch.setattr(postgres, "Vector", object())
    monkeypatch.setattr(postgres, "PGVECTOR_IMPORT_ERROR", None)
    engine_stub = _StubEngine()
    monkeypatch.setattr(postgres, "create_async_engine", lambda *args, **kwargs: engine_stub)
    monkeypatch.setattr(
        postgres,
        "async_sessionmaker",
        lambda *args, **kwargs: _StubSessionmaker(session),
    )

    backend = postgres.PostgresBackend("postgresql+asyncpg://user:pass@localhost/db")

    staged = ContextDelta(topic="policy", guideline="Follow rules.")
    active = ContextDelta(topic="tone", guideline="Be kind.", status=DeltaStatus.ACTIVE)

    await backend.initialize()
    assert engine_stub.metadata_created is True

    await backend.save_deltas([staged, active])
    assert set(session.records.keys()) == {staged.id, active.id}

    fetched = await backend.get_delta(staged.id)
    assert fetched is not None
    assert fetched.guideline == "Follow rules."

    queried = await backend.query_deltas(
        status=DeltaStatus.ACTIVE, tags=["policy"], topic="policy", limit=1
    )
    assert len(queried) >= 1

    activated = await backend.activate_staged()
    assert activated == 1
    assert session.records[staged.id].status == DeltaStatus.ACTIVE.value

    active.guideline = "Be very kind."
    await backend.update_delta(active)
    assert session.records[active.id].guideline == "Be very kind."

    await backend.delete_delta(active.id)
    assert active.id not in session.records

    assert session.commits >= 3


def test_postgres_backend_requires_pgvector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(postgres, "Vector", None)
    monkeypatch.setattr(postgres, "PGVECTOR_IMPORT_ERROR", ImportError("missing"))
    with pytest.raises(ImportError):
        postgres.PostgresBackend("postgresql+asyncpg://")
