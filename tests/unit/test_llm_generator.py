"""Unit tests for the LLMGenerator."""

from __future__ import annotations

from typing import Any, List

import pytest

from acet.generators.llm import LLMGenerator
from acet.llm.base import BaseLLMProvider, LLMResponse, Message


class _StubLLM(BaseLLMProvider):
    def __init__(self) -> None:
        self.captured_messages: List[Message] = []
        self._model = "stub-model"

    async def complete(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        self.captured_messages = messages
        return LLMResponse(content="answer", model=self._model, usage={"prompt_tokens": 5})

    def count_tokens(self, text: str) -> int:
        return len(text)

    @property
    def model_name(self) -> str:
        return self._model


@pytest.mark.asyncio
async def test_llm_generator_injects_context_and_metadata() -> None:
    llm = _StubLLM()
    generator = LLMGenerator(llm_provider=llm, system_prompt="System prompt.")
    context = ["- Guideline A", "- Guideline B"]

    result = await generator.generate("What is the policy?", context=context)

    assert llm.captured_messages[0].role == "system"
    assert "Context Playbook" in llm.captured_messages[0].content
    assert "- Guideline A" in llm.captured_messages[0].content
    assert result["answer"] == "answer"
    assert result["metadata"]["model"] == "stub-model"
    assert result["metadata"]["usage"]["prompt_tokens"] == 5
    assert result["metadata"]["context_bullets"] == len(context)


@pytest.mark.asyncio
async def test_llm_generator_without_context_uses_default_system_prompt() -> None:
    llm = _StubLLM()
    generator = LLMGenerator(llm_provider=llm, system_prompt="Default prompt.")

    await generator.generate("What is the policy?", context=[])

    system_message = llm.captured_messages[0]
    assert system_message.role == "system"
    assert system_message.content == "Default prompt."
