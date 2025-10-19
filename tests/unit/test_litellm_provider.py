"""Unit tests for the LiteLLMProvider wrapper."""

from __future__ import annotations

import types
from typing import Any, Dict

import pytest

from acet.llm.base import LLMResponse, Message
from acet.llm.providers.litellm_provider import LiteLLMProvider


class _FakeChoice:
    def __init__(self, content: str, finish: str = "stop") -> None:
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = finish


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.choices = [_FakeChoice(text)]
        self.model = "router-model"
        self.usage = types.SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)


class _FakeLiteLLM:
    def __init__(self) -> None:
        self.captured_kwargs: Dict[str, Any] = {}
        self.token_counter = lambda model, text: len(text)  # type: ignore[arg-type]

    async def acompletion(self, **kwargs: Any) -> Any:
        self.captured_kwargs = kwargs
        return _FakeResponse("lite llm reply")


@pytest.fixture
def _patched_litellm(monkeypatch: pytest.MonkeyPatch) -> _FakeLiteLLM:
    fake = _FakeLiteLLM()
    monkeypatch.setattr("acet.llm.providers.litellm_provider.litellm_module", fake)
    monkeypatch.setattr("acet.llm.providers.litellm_provider.LITELLM_IMPORT_ERROR", None)
    return fake


@pytest.mark.asyncio
async def test_litellm_provider_complete_uses_router(_patched_litellm: _FakeLiteLLM) -> None:
    provider = LiteLLMProvider(model="router-model")
    response = await provider.complete([Message(role="user", content="hello")])

    assert isinstance(response, LLMResponse)
    assert response.content == "lite llm reply"
    assert response.usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert _patched_litellm.captured_kwargs["model"] == "router-model"


def test_litellm_count_tokens_falls_back_without_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM()
    fake.token_counter = None  # type: ignore[assignment]
    monkeypatch.setattr("acet.llm.providers.litellm_provider.litellm_module", fake)
    monkeypatch.setattr("acet.llm.providers.litellm_provider.LITELLM_IMPORT_ERROR", None)

    provider = LiteLLMProvider(model="router-model")
    assert provider.count_tokens("abcdefgh") == 2  # len // 4
