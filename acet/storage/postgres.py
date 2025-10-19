"""PostgreSQL + pgvector storage backend."""

from __future__ import annotations

from typing import Any, List, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, select, update
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from acet.core.interfaces import StorageBackend
from acet.core.models import ContextDelta, DeltaStatus

Vector: Any | None
PGVECTOR_IMPORT_ERROR: ImportError | None

try:
    from pgvector.sqlalchemy import Vector as PGVector
except ImportError as exc:  # pragma: no cover - optional dependency
    PGVECTOR_IMPORT_ERROR = exc
    PGVector = None
else:
    PGVECTOR_IMPORT_ERROR = None

Vector = PGVector


class Base(DeclarativeBase):
    pass


class DeltaRecord(Base):
    __tablename__ = "deltas"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    guideline: Mapped[str] = mapped_column(Text, nullable=False)
    conditions: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    evidence: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String, default=DeltaStatus.STAGED.value)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    recency: Mapped[float] = mapped_column(Float, default=1.0)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    helpful_count: Mapped[int] = mapped_column(Integer, default=0)
    harmful_count: Mapped[int] = mapped_column(Integer, default=0)
    author: Mapped[str] = mapped_column(String, default="reflector")
    risk_level: Mapped[str] = mapped_column(String, default="low")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector if Vector is not None else ARRAY(Float),
        nullable=True,
    )


class PostgresBackend(StorageBackend):
    """PostgreSQL backend with pgvector support."""

    def __init__(self, connection_string: str) -> None:
        if Vector is None:
            raise ImportError(
                "pgvector is required for PostgresBackend. Install with "
                "`pip install pgvector psycopg2-binary`."
            ) from PGVECTOR_IMPORT_ERROR
        self.engine = create_async_engine(connection_string)
        self.Session: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Initialize database schema. Must be called before use."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_delta(self, delta: ContextDelta) -> None:
        record = self._to_record(delta)
        async with self.Session() as session:
            await session.merge(record)
            await session.commit()

    async def save_deltas(self, deltas: List[ContextDelta]) -> None:
        async with self.Session() as session:
            for delta in deltas:
                await session.merge(self._to_record(delta))
            await session.commit()

    async def get_delta(self, delta_id: str) -> Optional[ContextDelta]:
        async with self.Session() as session:
            result = await session.get(DeltaRecord, delta_id)
            return self._from_record(result) if result else None

    async def query_deltas(
        self,
        status: Optional[DeltaStatus] = None,
        tags: Optional[List[str]] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ContextDelta]:
        async with self.Session() as session:
            stmt = select(DeltaRecord)
            if status is not None:
                stmt = stmt.where(DeltaRecord.status == status.value)
            if topic is not None:
                stmt = stmt.where(DeltaRecord.topic == topic)
            if tags:
                stmt = stmt.where(DeltaRecord.tags.overlap(tags))
            if limit is not None:
                stmt = stmt.limit(limit)

            rows = (await session.scalars(stmt)).all()
            return [self._from_record(row) for row in rows]

    async def update_delta(self, delta: ContextDelta) -> None:
        await self.save_delta(delta)

    async def delete_delta(self, delta_id: str) -> None:
        async with self.Session() as session:
            record = await session.get(DeltaRecord, delta_id)
            if record is not None:
                await session.delete(record)
                await session.commit()

    async def activate_staged(self) -> int:
        async with self.Session() as session:
            stmt = (
                update(DeltaRecord)
                .where(DeltaRecord.status == DeltaStatus.STAGED.value)
                .values(status=DeltaStatus.ACTIVE.value)
                .returning(DeltaRecord.id)
            )
            result = await session.execute(stmt)
            await session.commit()
            activated_ids = result.scalars().all()
            return len(activated_ids)

    def _to_record(self, delta: ContextDelta) -> DeltaRecord:
        data = delta.model_dump()
        return DeltaRecord(
            id=data["id"],
            topic=data["topic"],
            guideline=data["guideline"],
            conditions=data["conditions"],
            evidence=data["evidence"],
            tags=data["tags"],
            version=data["version"],
            status=getattr(delta.status, "value", delta.status),
            score=data["score"],
            recency=data["recency"],
            usage_count=data["usage_count"],
            helpful_count=data["helpful_count"],
            harmful_count=data["harmful_count"],
            author=data["author"],
            risk_level=data["risk_level"],
            confidence=data["confidence"],
            created_at=delta.created_at,
            updated_at=delta.updated_at,
            embedding=delta.embedding,
        )

    def _from_record(self, record: DeltaRecord) -> ContextDelta:
        return ContextDelta(
            id=record.id,
            topic=record.topic,
            guideline=record.guideline,
            conditions=record.conditions or [],
            evidence=record.evidence or [],
            tags=record.tags or [],
            version=record.version,
            status=DeltaStatus(record.status),
            score=record.score,
            recency=record.recency,
            usage_count=record.usage_count,
            helpful_count=record.helpful_count,
            harmful_count=record.harmful_count,
            author=record.author,
            risk_level=record.risk_level,
            confidence=record.confidence,
            created_at=record.created_at,
            updated_at=record.updated_at,
            embedding=list(record.embedding) if record.embedding is not None else None,
        )
