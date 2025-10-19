"""Unit tests for the LangChain ACETMemory adapter."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from acet.core.interfaces import Curator, EmbeddingProvider, Generator, Reflector
from acet.core.models import ACETConfig, ContextDelta, DeltaStatus, ReflectionReport
from acet.engine import ACETEngine
from acet.integrations.langchain import ACETMemory
from acet.retrieval import DeltaRanker
from acet.storage.memory import MemoryBackend


class _NullGenerator(Generator):
    async def generate(self, query: str, context: List[str], **kwargs: Any) -> Dict[str, Any]:
        return {"answer": "", "evidence": [], "metadata": {}}


class _NullReflector(Reflector):
    async def reflect(
        self,
        query: str,
        answer: str,
        evidence: List[str],
        context: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectionReport:
        return ReflectionReport(question=query, answer=answer)

    async def refine(self, report: ReflectionReport, iterations: int = 3) -> ReflectionReport:
        return report


class _NullCurator(Curator):
    async def curate(self, report: ReflectionReport, existing_deltas: List[ContextDelta]) -> List[ContextDelta]:
        return []

    def score_delta(self, delta: ContextDelta) -> float:
        return delta.confidence

    def deduplicate(self, candidate: ContextDelta, existing: List[ContextDelta], threshold: float = 0.90) -> bool:
        return False


class _StubEmbedding(EmbeddingProvider):
    async def embed(self, text: str) -> List[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        return 1.0 / (1.0 + abs(emb1[0] - emb2[0]))


class RecordingEngine(ACETEngine):
    def __init__(self) -> None:
        super().__init__(
            generator=_NullGenerator(),
            reflector=_NullReflector(),
            curator=_NullCurator(),
            storage=MemoryBackend(),
            ranker=DeltaRanker(_StubEmbedding()),
            config=ACETConfig(reflection_sample_rate=0.0),
        )
        self.last_ingest: Dict[str, Any] | None = None

    async def ingest_interaction(
        self,
        query: str,
        answer: str,
        *,
        evidence: Optional[List[str]] = None,
        context_deltas: Optional[List[ContextDelta]] = None,
        ground_truth: Optional[str] = None,
        update_usage: bool = True,
    ) -> Dict[str, Any]:
        evidence_list = evidence or []
        deltas = context_deltas or []
        self.last_ingest = {
            "query": query,
            "answer": answer,
            "evidence": evidence_list,
            "context_ids": [delta.id for delta in deltas],
            "ground_truth": ground_truth,
        }
        return {"report": None, "created_deltas": [], "context_tokens": 0}


async def _store(engine: RecordingEngine, *deltas: ContextDelta) -> None:
    for delta in deltas:
        await engine.storage.save_delta(delta)


@pytest.mark.asyncio
async def test_memory_loads_and_saves_context() -> None:
    delta = ContextDelta(topic="policy", guideline="Follow the rules.", status=DeltaStatus.ACTIVE)
    engine = RecordingEngine()
    await _store(engine, delta)
    memory = ACETMemory(
        engine,
        context_key="context",
        input_keys=["input"],
        ground_truth_key="ground_truth",
    )

    loaded = await memory.aload_memory_variables({"input": "How to comply?"})
    assert "Follow the rules." in loaded["context"]

    await memory.asave_context(
        inputs={"input": "How to comply?", "ground_truth": "reference"},
        outputs={"response": "Here is the answer.", "evidence": ["doc1"]},
    )

    assert engine.last_ingest == {
        "query": "How to comply?",
        "answer": "Here is the answer.",
        "evidence": ["doc1"],
        "context_ids": [delta.id],
        "ground_truth": "reference",
    }

    loaded_empty = await memory.aload_memory_variables({"input": ""})
    assert loaded_empty["context"] == ""


def test_memory_sync_wrappers() -> None:
    delta = ContextDelta(topic="policy", guideline="Rule.", status=DeltaStatus.ACTIVE)
    engine = RecordingEngine()
    asyncio.run(_store(engine, delta))
    memory = ACETMemory(
        engine,
        context_key="context",
        input_keys=["input"],
        update_context=False,
        ground_truth_key="ground_truth",
    )

    loaded = memory.load_memory_variables({"input": "Hi"})
    assert "Rule." in loaded["context"]

    memory.save_context({"input": "ignored"}, {"response": "ignored"})
    assert engine.last_ingest is None


@pytest.mark.asyncio
async def test_memory_load_sync_raises_with_event_loop() -> None:
    engine = RecordingEngine()
    memory = ACETMemory(engine, context_key="context", input_keys=["input"])

    coro = memory.aload_memory_variables({"input": "hi"})
    try:
        with pytest.raises(RuntimeError):
            memory._run_sync(coro)
    finally:
        coro.close()


@pytest.mark.asyncio
async def test_memory_aload_handles_missing_query_and_no_active() -> None:
    engine = RecordingEngine()
    memory = ACETMemory(engine, context_key="context", input_keys=["input"])

    empty = await memory.aload_memory_variables({"input": ""})
    assert empty["context"] == ""

    none_active = await memory.aload_memory_variables({"input": "hello"})
    assert none_active["context"] == ""


@pytest.mark.asyncio
async def test_memory_asave_skips_when_missing_answer() -> None:
    delta = ContextDelta(topic="policy", guideline="Rule.", status=DeltaStatus.ACTIVE)
    engine = RecordingEngine()
    await _store(engine, delta)
    memory = ACETMemory(engine, context_key="context", input_keys=["input"])
    memory._last_context_deltas = [delta]

    await memory.asave_context({"input": "question"}, {"response": ""})

    assert engine.last_ingest is None
    assert memory._last_context_deltas == []


def test_memory_extract_helpers() -> None:
    engine = RecordingEngine()
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
