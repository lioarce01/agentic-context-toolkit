"""Unit tests for the OpenAIProvider behaviour."""

from __future__ import annotations

import types
from typing import Any

import pytest

from acet.llm.base import LLMResponse, Message
from acet.llm.providers.openai import OpenAIProvider


class _FakeChoice:
    def __init__(self, content: str, finish: str = "stop") -> None:
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = finish
        self.delta = types.SimpleNamespace(content=[types.SimpleNamespace(text=content)])


class _FakeCompletion:
    def __init__(self, text: str = "hello world") -> None:
        self.choices = [_FakeChoice(text)]
        self.model = "gpt-4o"
        self.usage = types.SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)


class _FakeStreamChunk:
    def __init__(self, text: str) -> None:
        self.choices = [_FakeChoice(text)]


class _FakeChatClient:
    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.captured_kwargs = kwargs
        if kwargs.get("stream"):
            async def generator() -> Any:
                for token in ["he", "llo"]:
                    yield _FakeStreamChunk(token)
            return generator()
        return _FakeCompletion("hello world")


class _FakeAsyncOpenAI:
    def __init__(self, **_: Any) -> None:
        self.chat = types.SimpleNamespace(completions=_FakeChatClient())


class _FakeTiktoken:
    @staticmethod
    def encoding_for_model(model: str) -> Any:
        return types.SimpleNamespace(encode=lambda text: list(text))

    @staticmethod
    def get_encoding(_: str) -> Any:
        return types.SimpleNamespace(encode=lambda text: list(text))


@pytest.fixture(autouse=True)
def _patch_openai_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_openai = types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI)
    monkeypatch.setattr("acet.llm.providers.openai.openai_module", fake_openai)
    monkeypatch.setattr("acet.llm.providers.openai.OPENAI_IMPORT_ERROR", None)
    monkeypatch.setattr("acet.llm.providers.openai.tiktoken_module", _FakeTiktoken)


@pytest.mark.asyncio
async def test_openai_provider_complete_builds_response() -> None:
    provider = OpenAIProvider(model="gpt-test")
    response = await provider.complete([Message(role="user", content="hi")])

    assert isinstance(response, LLMResponse)
    assert response.content == "hello world"
    assert response.usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert response.model == "gpt-4o"


@pytest.mark.asyncio
async def test_openai_provider_complete_stream_yields_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="gpt-stream")
    iterator = await provider.complete_stream([Message(role="user", content="hi")])
    tokens = [token async for token in iterator]

    assert tokens == ["he", "llo"]


def test_openai_count_tokens_uses_encoder() -> None:
    provider = OpenAIProvider(model="gpt-mock")
    assert provider.count_tokens("abc") == 3


def test_openai_first_choice_raises_when_missing() -> None:
    completion = types.SimpleNamespace(choices=[])
    with pytest.raises(ValueError):
        OpenAIProvider._first_choice(completion)


def test_openai_extract_message_and_delta_tokens() -> None:
    content_items = [types.SimpleNamespace(text="part1"), types.SimpleNamespace(text="part2")]
    message = types.SimpleNamespace(content=content_items)
    delta = types.SimpleNamespace(content="token")

    text = OpenAIProvider._extract_message_text(message)
    tokens = list(OpenAIProvider._extract_delta_tokens(delta))

    assert text == "part1part2"
    assert tokens == ["token"]


def test_openai_build_usage_defaults_to_zero() -> None:
    completion = types.SimpleNamespace(choices=[1], usage=None)
    usage = OpenAIProvider._build_usage(completion)
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
