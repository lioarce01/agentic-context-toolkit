"""Unit tests for the ReActAgent orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pytest

from acet.agents.react import ReActAgent, Tool
from acet.core.models import ContextDelta, DeltaStatus
from acet.llm.base import BaseLLMProvider, LLMResponse, Message


class _StubLLM(BaseLLMProvider):
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: List[List[Message]] = []

    async def complete(self, messages: List[Message], **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        self.calls.append(messages)
        content = self._responses.pop(0)
        return LLMResponse(content=content, model="stub-model", usage={"prompt_tokens": 5})

    def count_tokens(self, text: str) -> int:
        return len(text)

    @property
    def model_name(self) -> str:
        return "stub-model"


class _StubRanker:
    async def rank(self, query: str, deltas: List[ContextDelta], top_k: int = 10):
        return [(delta, 1.0) for delta in deltas[:top_k]]


class _StubBudget:
    def pack_deltas(self, deltas: List[ContextDelta], budget: Optional[int] = None):
        bullets = [f"- {delta.guideline}" for delta in deltas]
        return bullets, len(bullets)


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
        if status is None:
            return self.deltas
        return [delta for delta in self.deltas if delta.status == status]


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
        }
        return {"created_deltas": []}


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
    engine = _StubEngine([delta])

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
    assert engine.ingested["query"] == "How to comply?"
    assert engine.ingested["evidence"] == ["policy document"]
    assert engine.ingested["context_ids"] == [delta.id]
    assert result["metadata"]["source"] == "unit"


@pytest.mark.asyncio
async def test_react_agent_handles_unknown_tool_and_immediate_answer() -> None:
    llm = _StubLLM(["Thought: done\nAction: unknown\nAction Input: ?\nFinal Answer:"])
    delta = ContextDelta(topic="policy", guideline="Be accurate.", status=DeltaStatus.ACTIVE)
    engine = _StubEngine([delta])

    agent = ReActAgent(engine=engine, llm=llm, tools=[], max_steps=1)

    result = await agent.run("Question?")

    assert "Unknown tool" in engine.ingested["evidence"][0]
    assert result["answer"].startswith("Unable to produce")


@pytest.mark.asyncio
async def test_react_agent_uses_raw_content_when_unstructured() -> None:
    llm = _StubLLM(["This is the complete answer."])
    engine = _StubEngine([])
    agent = ReActAgent(engine=engine, llm=llm, tools=[], max_steps=1)

    result = await agent.run("Question?")

    assert result["answer"] == "This is the complete answer."
