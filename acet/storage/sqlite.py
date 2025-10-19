"""Asynchronous SQLite-backed storage implementation."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, cast

import aiosqlite

from acet.core.interfaces import StorageBackend
from acet.core.models import ContextDelta, DeltaStatus


class SQLiteBackend(StorageBackend):
    """SQLite-based storage backend for ACET context deltas."""

    def __init__(self, db_path: str = "ACET_deltas.db") -> None:
        self.db_path = Path(db_path)
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def save_delta(self, delta: ContextDelta) -> None:
        payload = self._serialize_delta(delta)
        columns = ", ".join(payload.keys())
        placeholders = ", ".join("?" for _ in payload)
        values = list(payload.values())

        statement = f"""
            INSERT INTO deltas ({columns})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
                topic=excluded.topic,
                guideline=excluded.guideline,
                conditions=excluded.conditions,
                evidence=excluded.evidence,
                tags=excluded.tags,
                version=excluded.version,
                status=excluded.status,
                score=excluded.score,
                recency=excluded.recency,
                usage_count=excluded.usage_count,
                helpful_count=excluded.helpful_count,
                harmful_count=excluded.harmful_count,
                author=excluded.author,
                risk_level=excluded.risk_level,
                confidence=excluded.confidence,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at,
                embedding=excluded.embedding
        """

        async with self._connect() as conn:
            await conn.execute(statement, values)
            await conn.commit()

    async def save_deltas(self, deltas: List[ContextDelta]) -> None:
        if not deltas:
            return

        payloads = [self._serialize_delta(delta) for delta in deltas]
        columns = ", ".join(payloads[0].keys())
        placeholders = ", ".join("?" for _ in payloads[0])
        values = [list(payload.values()) for payload in payloads]

        statement = f"""
            INSERT INTO deltas ({columns})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
                topic=excluded.topic,
                guideline=excluded.guideline,
                conditions=excluded.conditions,
                evidence=excluded.evidence,
                tags=excluded.tags,
                version=excluded.version,
                status=excluded.status,
                score=excluded.score,
                recency=excluded.recency,
                usage_count=excluded.usage_count,
                helpful_count=excluded.helpful_count,
                harmful_count=excluded.harmful_count,
                author=excluded.author,
                risk_level=excluded.risk_level,
                confidence=excluded.confidence,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at,
                embedding=excluded.embedding
        """

        async with self._connect() as conn:
            await conn.executemany(statement, values)
            await conn.commit()

    async def get_delta(self, delta_id: str) -> Optional[ContextDelta]:
        async with self._connect() as conn:
            cursor = await conn.execute("SELECT * FROM deltas WHERE id = ?", (delta_id,))
            row = await cursor.fetchone()
        return self._deserialize_row(cast(Mapping[str, Any], row)) if row else None

    async def query_deltas(
        self,
        status: Optional[DeltaStatus] = None,
        tags: Optional[List[str]] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ContextDelta]:
        clauses: List[str] = []
        parameters: List[Any] = []

        if status is not None:
            clauses.append("status = ?")
            parameters.append(status.value)

        if topic is not None:
            clauses.append("topic = ?")
            parameters.append(topic)

        if tags:
            tag_filters = " OR ".join("tags LIKE ?" for _ in tags)
            clauses.append(f"({tag_filters})")
            parameters.extend([f"%{tag}%" for tag in tags])

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        query = f"SELECT * FROM deltas {where_clause} ORDER BY updated_at DESC {limit_clause}"

        async with self._connect() as conn:
            cursor = await conn.execute(query, tuple(parameters))
            rows = await cursor.fetchall()

        return [self._deserialize_row(cast(Mapping[str, Any], row)) for row in rows]

    async def update_delta(self, delta: ContextDelta) -> None:
        await self.save_delta(delta)

    async def delete_delta(self, delta_id: str) -> None:
        async with self._connect() as conn:
            await conn.execute("DELETE FROM deltas WHERE id = ?", (delta_id,))
            await conn.commit()

    async def activate_staged(self) -> int:
        async with self._connect() as conn:
            cursor = await conn.execute(
                "UPDATE deltas SET status = ? WHERE status = ?",
                (DeltaStatus.ACTIVE.value, DeltaStatus.STAGED.value),
            )
            await conn.commit()
            return cursor.rowcount or 0

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            async with aiosqlite.connect(self.db_path) as conn:
                await conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS deltas (
                        id TEXT PRIMARY KEY,
                        topic TEXT NOT NULL,
                        guideline TEXT NOT NULL,
                        conditions TEXT,
                        evidence TEXT,
                        tags TEXT,
                        version INTEGER DEFAULT 1,
                        status TEXT DEFAULT 'staged',
                        score REAL DEFAULT 0.0,
                        recency REAL DEFAULT 1.0,
                        usage_count INTEGER DEFAULT 0,
                        helpful_count INTEGER DEFAULT 0,
                        harmful_count INTEGER DEFAULT 0,
                        author TEXT,
                        risk_level TEXT DEFAULT 'low',
                        confidence REAL DEFAULT 0.0,
                        created_at TEXT,
                        updated_at TEXT,
                        embedding BLOB
                    );
                    CREATE INDEX IF NOT EXISTS idx_status ON deltas(status);
                    CREATE INDEX IF NOT EXISTS idx_topic ON deltas(topic);
                    CREATE INDEX IF NOT EXISTS idx_tags ON deltas(tags);
                    """
                )
                await conn.commit()

            self._initialized = True

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        await self._ensure_initialized()
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        try:
            yield conn
        finally:
            await conn.close()

    def _serialize_delta(self, delta: ContextDelta) -> Dict[str, Any]:
        data = delta.model_dump()
        return {
            "id": data["id"],
            "topic": data["topic"],
            "guideline": data["guideline"],
            "conditions": json.dumps(data["conditions"]),
            "evidence": json.dumps(data["evidence"]),
            "tags": json.dumps(data["tags"]),
            "version": data["version"],
            "status": data["status"],
            "score": data["score"],
            "recency": data["recency"],
            "usage_count": data["usage_count"],
            "helpful_count": data["helpful_count"],
            "harmful_count": data["harmful_count"],
            "author": data["author"],
            "risk_level": data["risk_level"],
            "confidence": data["confidence"],
            "created_at": delta.created_at.isoformat(),
            "updated_at": delta.updated_at.isoformat(),
            "embedding": json.dumps(delta.embedding) if delta.embedding is not None else None,
        }

    def _deserialize_row(self, row: Mapping[str, Any]) -> ContextDelta:
        embedding_json = row["embedding"]
        embedding = json.loads(embedding_json) if embedding_json else None

        return ContextDelta(
            id=row["id"],
            topic=row["topic"],
            guideline=row["guideline"],
            conditions=json.loads(row["conditions"]) if row["conditions"] else [],
            evidence=json.loads(row["evidence"]) if row["evidence"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            version=row["version"],
            status=DeltaStatus(row["status"]),
            score=row["score"],
            recency=row["recency"],
            usage_count=row["usage_count"],
            helpful_count=row["helpful_count"],
            harmful_count=row["harmful_count"],
            author=row["author"] or "reflector",
            risk_level=row["risk_level"] or "low",
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            embedding=embedding,
        )
