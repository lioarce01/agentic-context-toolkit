"""Unit tests for the LLMReflector."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List

import pytest

from acet.core.models import ReflectionReport
from acet.llm.base import BaseLLMProvider, LLMResponse, Message
from acet.reflectors.llm import LLMReflector

ResponseLike = LLMResponse | Exception


class _StubLLM(BaseLLMProvider):
    def __init__(self, responses: List[ResponseLike]) -> None:
        self.responses: Deque[ResponseLike] = deque(responses)
        self.calls: List[Dict[str, Any]] = []

    async def complete(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        outcome = self.responses.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def count_tokens(self, text: str) -> int:
        return len(text)

    @property
    def model_name(self) -> str:
        return "stub-model"


def _response(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="stub", usage={})


@pytest.mark.asyncio
async def test_reflect_parses_json_response() -> None:
    llm = _StubLLM([_response('{"issues": [], "proposed_insights": []}')])
    reflector = LLMReflector(llm)

    report = await reflector.reflect(
        query="Q",
        answer="A",
        evidence=["doc"],
        context=["- delta"],
    )

    assert report.question == "Q"
    assert report.evidence_refs == ["doc"]
    call_kwargs = llm.calls[0]["kwargs"]
    assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_reflect_retries_without_response_format_on_error() -> None:
    llm = _StubLLM(
        [
            RuntimeError("bad json mode"),
            _response("```json\n{\"proposed_insights\": []}\n```"),
        ]
    )
    reflector = LLMReflector(llm)

    report = await reflector.reflect(query="Q", answer="A", evidence=[], context=[])

    assert report.proposed_insights == []
    assert len(llm.calls) == 2
    assert "response_format" not in llm.calls[1]["kwargs"]


@pytest.mark.asyncio
async def test_refine_stops_on_invalid_json() -> None:
    llm = _StubLLM(
        [
            _response('{"issues": [{"type": "gap", "explanation": "Fix", "severity": 3}]}'),
            _response("not json"),
        ]
    )
    reflector = LLMReflector(llm)

    base_report = ReflectionReport(question="Q", answer="A")

    refined = await reflector.refine(base_report, iterations=2)

    assert refined.issues[0].type == "gap"
    assert len(llm.calls) >= 2
