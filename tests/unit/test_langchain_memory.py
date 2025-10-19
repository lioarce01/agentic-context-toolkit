"""Unit tests for the LangChain ACETMemory adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from acet.core.models import ContextDelta, DeltaStatus
from acet.integrations.langchain import ACETMemory


@dataclass
class _StubStorage:
    deltas: List[ContextDelta]

    async def query_deltas(
        self,
        status: Optional[DeltaStatus] = None,
        tags: Optional[List[str]] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ContextDelta]:
        results = self.deltas
        if status is not None:
            results = [delta for delta in results if delta.status == status]
        return results


class _StubRanker:
    async def rank(self, query: str, deltas: List[ContextDelta], top_k: int = 10):
        return [(delta, 1.0) for delta in deltas[:top_k]]


class _StubBudget:
    def pack_deltas(self, deltas: List[ContextDelta], budget: Optional[int] = None):
        bullets = [f"- {delta.guideline}" for delta in deltas]
        return bullets, len(bullets) * 5


class _StubEngine:
    def __init__(self, deltas: List[ContextDelta]) -> None:
        self.storage = _StubStorage(deltas)
        self.ranker = _StubRanker()
        self.budget_manager = _StubBudget()
        self.ingested: Dict[str, Any] = {}

    async def ingest_interaction(
        self,
        query: str,
        answer: str,
        *,
        evidence: List[str],
        context_deltas: List[ContextDelta],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ingested = {
            "query": query,
            "answer": answer,
            "evidence": evidence,
            "context_ids": [delta.id for delta in context_deltas],
            "ground_truth": ground_truth,
        }
        return {"created_deltas": []}


@pytest.mark.asyncio
async def test_memory_loads_and_saves_context() -> None:
    delta = ContextDelta(topic="policy", guideline="Follow the rules.", status=DeltaStatus.ACTIVE)
    engine = _StubEngine([delta])
    memory = ACETMemory(
        engine,
        context_key="context",
        input_keys=["input"],
        ground_truth_key="ground_truth",
    )

    loaded = await memory.aload_memory_variables({"input": "How to comply?"})
    assert "- Follow the rules." in loaded["context"]

    await memory.asave_context(
        inputs={"input": "How to comply?", "ground_truth": "reference"},
        outputs={"response": "Here is the answer.", "evidence": ["doc1"]},
    )

    assert engine.ingested["query"] == "How to comply?"
    assert engine.ingested["evidence"] == ["doc1"]
    assert engine.ingested["ground_truth"] == "reference"
    assert engine.ingested["context_ids"] == [delta.id]

    # When no query provided context should reset.
    loaded_empty = await memory.aload_memory_variables({"input": ""})
    assert loaded_empty["context"] == ""


def test_memory_sync_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    delta = ContextDelta(topic="policy", guideline="Rule.", status=DeltaStatus.ACTIVE)
    engine = _StubEngine([delta])
    memory = ACETMemory(
        engine,
        context_key="context",
        input_keys=["input"],
        update_context=False,
        ground_truth_key="ground_truth",
    )

    # Synchronous load should proxy to async implementation via asyncio.run.
    loaded = memory.load_memory_variables({"input": "Hi"})
    assert "- Rule." in loaded["context"]

    # With update_context disabled, save_context should short-circuit and reset.
    memory.save_context({"input": "ignored"}, {"response": "ignored"})
    assert engine.ingested == {}


@pytest.mark.asyncio
async def test_memory_load_sync_raises_with_event_loop() -> None:
    engine = _StubEngine([])
    memory = ACETMemory(engine, context_key="context", input_keys=["input"])

    coro = memory.aload_memory_variables({"input": "hi"})
    try:
        with pytest.raises(RuntimeError):
            memory._run_sync(coro)
    finally:
        coro.close()

@pytest.mark.asyncio
async def test_memory_aload_handles_missing_query_and_no_active() -> None:
    engine = _StubEngine([])
    memory = ACETMemory(engine, context_key="context", input_keys=["input"])

    empty = await memory.aload_memory_variables({"input": ""})
    assert empty["context"] == ""

    none_active = await memory.aload_memory_variables({"input": "hello"})
    assert none_active["context"] == ""


@pytest.mark.asyncio
async def test_memory_asave_skips_when_missing_answer() -> None:
    delta = ContextDelta(topic="policy", guideline="Rule.", status=DeltaStatus.ACTIVE)
    engine = _StubEngine([delta])
    memory = ACETMemory(engine, context_key="context", input_keys=["input"])
    memory._last_context_deltas = [delta]

    await memory.asave_context({"input": "question"}, {"response": ""})

    assert engine.ingested == {}
    assert memory._last_context_deltas == []


def test_memory_extract_helpers() -> None:
    engine = _StubEngine([])
    memory = ACETMemory(
        engine,
        context_key="ctx",
        input_keys=["query", "prompt"],
        output_key="result",
        ground_truth_key="truth",
    )

    assert memory._extract_query({"prompt": " value "}) == " value "
    assert memory._extract_query({"prompt": "   "}) == ""
    assert memory._extract_output({"result": {"content": "A"}}) == "A"
    assert memory._extract_output({"result": {"response": "B"}}) == "B"
    assert memory._extract_output({"result": {"text": "C"}}) == "C"
    assert memory._extract_output({"result": 42}) == ""
    assert memory._extract_evidence({"evidence": ["doc"]}) == ["doc"]
    assert memory._extract_evidence({"evidence": ["doc", 1]}) == []
    assert memory._extract_ground_truth({"truth": "GT"}) == "GT"
    assert memory._extract_ground_truth({"truth": 123}) is None

    memory._last_context_deltas = [ContextDelta(topic="t", guideline="g")]
    memory.clear()
    assert memory._last_context_deltas == []

    no_truth_memory = ACETMemory(engine, context_key="ctx", input_keys=["query"])
    assert no_truth_memory._extract_ground_truth({"unused": "value"}) is None
