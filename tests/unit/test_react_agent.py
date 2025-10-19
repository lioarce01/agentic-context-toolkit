"""Unit tests for the ReActAgent orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pytest

from acet.agents.react import ReActAgent, Tool
from acet.core.interfaces import Curator, EmbeddingProvider, Generator, Reflector
from acet.core.models import ACETConfig, ContextDelta, DeltaStatus, ReflectionReport
from acet.engine import ACETEngine
from acet.llm.base import BaseLLMProvider, LLMResponse, Message
from acet.retrieval import DeltaRanker
from acet.storage.memory import MemoryBackend


class _StubLLM(BaseLLMProvider):
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: List[List[Message]] = []

    async def complete(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        self.calls.append(messages)
        content = self._responses.pop(0)
        return LLMResponse(content=content, model="stub-model", usage={"prompt_tokens": 5})

    def count_tokens(self, text: str) -> int:
        return len(text)

    @property
    def model_name(self) -> str:
        return "stub-model"


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


async def _activate(engine: RecordingEngine, deltas: List[ContextDelta]) -> None:
    for delta in deltas:
        await engine.storage.save_delta(delta)


@pytest.mark.asyncio
async def test_react_agent_executes_tool_and_ingests_interaction() -> None:
    responses = [
        (
            "Thought: need tool\n"
            "Action: lookup\n"
            "Action Input: policy\n"
            "Final Answer:"
        ),
        "Final Answer: Always comply with policy.",
    ]
    llm = _StubLLM(responses)
    delta = ContextDelta(topic="policy", guideline="Follow policy.", status=DeltaStatus.ACTIVE)
    engine = RecordingEngine()
    await _activate(engine, [delta])

    tool_calls: List[str] = []

    async def lookup_tool(text: str) -> str:
        tool_calls.append(text)
        return "policy document"

    agent = ReActAgent(
        engine=engine,
        llm=llm,
        tools=[Tool(name="lookup", description="Lookup info", coroutine=lookup_tool)],
        max_steps=2,
    )

    result = await agent.run("How to comply?", metadata={"source": "unit"})

    assert result["answer"] == "Always comply with policy."
    assert tool_calls == ["policy"]
    assert engine.last_ingest == {
        "query": "How to comply?",
        "answer": "Always comply with policy.",
        "evidence": ["policy document"],
        "context_ids": [delta.id],
        "ground_truth": None,
    }
    assert result["metadata"]["source"] == "unit"


@pytest.mark.asyncio
async def test_react_agent_handles_unknown_tool_and_immediate_answer() -> None:
    llm = _StubLLM(["Thought: done\nAction: unknown\nAction Input: ?\nFinal Answer:"])
    delta = ContextDelta(topic="policy", guideline="Be accurate.", status=DeltaStatus.ACTIVE)
    engine = RecordingEngine()
    await _activate(engine, [delta])

    agent = ReActAgent(engine=engine, llm=llm, tools=[], max_steps=1)

    result = await agent.run("Question?")

    assert engine.last_ingest is not None
    assert "Unknown tool" in engine.last_ingest["evidence"][0]
    assert result["answer"].startswith("Unable to produce")


@pytest.mark.asyncio
async def test_react_agent_uses_raw_content_when_unstructured() -> None:
    llm = _StubLLM(["This is the complete answer."])
    engine = RecordingEngine()

    agent = ReActAgent(engine=engine, llm=llm, tools=[], max_steps=1)

    result = await agent.run("Question?")

    assert engine.last_ingest is not None
    assert result["answer"] == "This is the complete answer."
