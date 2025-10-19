"""Unit tests for the AnthropicProvider wrapper."""

from __future__ import annotations

import types
from typing import Any, Dict

import pytest

from acet.llm.base import LLMResponse, Message
from acet.llm.providers.anthropic import AnthropicProvider


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.model = "claude-test"
        self.usage = _FakeUsage(4, 6)
        self.content = [types.SimpleNamespace(text=text)]


class _FakeMessagesClient:
    def __init__(self) -> None:
        self.captured_kwargs: Dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.captured_kwargs = kwargs
        return _FakeResponse("anthropic reply")


class _FakeAnthropic:
    def __init__(self, **_: Any) -> None:
        self.messages = _FakeMessagesClient()


@pytest.fixture(autouse=True)
def _patch_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.SimpleNamespace(AsyncAnthropic=_FakeAnthropic)
    monkeypatch.setattr("acet.llm.providers.anthropic.anthropic_module", fake_module)
    monkeypatch.setattr("acet.llm.providers.anthropic.ANTHROPIC_IMPORT_ERROR", None)


@pytest.mark.asyncio
async def test_anthropic_provider_complete_handles_system_messages() -> None:
    provider = AnthropicProvider(model="claude-test")
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Say hi"),
    ]

    response = await provider.complete(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "anthropic reply"
    assert response.usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    assert response.model == "claude-test"


def test_anthropic_count_tokens_approximation() -> None:
    provider = AnthropicProvider(model="claude-test")
    assert provider.count_tokens("abcd") == 1
