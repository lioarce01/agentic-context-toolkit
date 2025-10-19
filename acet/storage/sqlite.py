"""SQLite-backed storage implementation."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

from acet.core.interfaces import StorageBackend
from acet.core.models import ContextDelta, DeltaStatus


class SQLiteBackend(StorageBackend):
    """SQLite-based storage backend for ACET context deltas."""

    def __init__(self, db_path: str = "ACET_deltas.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
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
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON deltas(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON deltas(topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags ON deltas(tags)")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    async def save_delta(self, delta: ContextDelta) -> None:
        payload = self._serialize_delta(delta)
        placeholders = ", ".join("?" for _ in payload)
        columns = ", ".join(payload.keys())
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
        with self._connect() as conn:
            conn.execute(statement, values)

    async def save_deltas(self, deltas: List[ContextDelta]) -> None:
        if not deltas:
            return

        payloads = [self._serialize_delta(delta) for delta in deltas]
        columns = ", ".join(payloads[0].keys())
        placeholders = ", ".join("?" for _ in payloads[0])

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

        values = [list(payload.values()) for payload in payloads]
        with self._connect() as conn:
            conn.executemany(statement, values)

    async def get_delta(self, delta_id: str) -> Optional[ContextDelta]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM deltas WHERE id = ?", (delta_id,)
            ).fetchone()
        return self._deserialize_row(row) if row else None

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

        with self._connect() as conn:
            rows = conn.execute(query, tuple(parameters)).fetchall()
        return [self._deserialize_row(row) for row in rows]

    async def update_delta(self, delta: ContextDelta) -> None:
        await self.save_delta(delta)

    async def delete_delta(self, delta_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM deltas WHERE id = ?", (delta_id,))

    async def activate_staged(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE deltas SET status = ? WHERE status = ?",
                (DeltaStatus.ACTIVE.value, DeltaStatus.STAGED.value),
            )
            return cursor.rowcount

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
            "status": delta.status.value,
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

    def _deserialize_row(self, row: Sequence[Any]) -> ContextDelta:
        (
            delta_id,
            topic,
            guideline,
            conditions,
            evidence,
            tags,
            version,
            status,
            score,
            recency,
            usage_count,
            helpful_count,
            harmful_count,
            author,
            risk_level,
            confidence,
            created_at,
            updated_at,
            embedding_json,
        ) = row

        embedding = json.loads(embedding_json) if embedding_json else None

        return ContextDelta(
            id=delta_id,
            topic=topic,
            guideline=guideline,
            conditions=json.loads(conditions) if conditions else [],
            evidence=json.loads(evidence) if evidence else [],
            tags=json.loads(tags) if tags else [],
            version=version,
            status=DeltaStatus(status),
            score=score,
            recency=recency,
            usage_count=usage_count,
            helpful_count=helpful_count,
            harmful_count=harmful_count,
            author=author or "reflector",
            risk_level=risk_level or "low",
            confidence=confidence,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            embedding=embedding,
        )

