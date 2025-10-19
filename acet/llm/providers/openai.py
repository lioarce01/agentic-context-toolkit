"""OpenAI chat provider implementation."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterable, List, Optional, cast

from acet.llm.base import BaseLLMProvider, LLMResponse, Message

try:  # pragma: no cover - optional dependency
    openai_module: Any | None = importlib.import_module("openai")
except ImportError as exc:  # pragma: no cover - optional dependency
    OPENAI_IMPORT_ERROR: ImportError | None = exc
    openai_module = None
else:
    OPENAI_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    tiktoken_module: Any | None = importlib.import_module("tiktoken")
except ImportError:  # pragma: no cover - optional dependency
    tiktoken_module = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import AsyncOpenAI as OpenAIClient
    from tiktoken.core import Encoding as TiktokenEncoding
else:  # pragma: no cover - runtime fallback
    OpenAIClient = Any
    TiktokenEncoding = Any


class OpenAIProvider(BaseLLMProvider):
    """LLM provider that delegates to OpenAI's async client."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **default_kwargs: Any,
    ) -> None:
        if openai_module is None:
            raise ImportError(
                "openai is required for OpenAIProvider. Install with `pip install openai`."
            ) from OPENAI_IMPORT_ERROR

        self._model = model
        client_cls = cast(type[OpenAIClient], openai_module.AsyncOpenAI)
        self._client: OpenAIClient = client_cls(api_key=api_key, base_url=base_url)
        self._default_kwargs = default_kwargs
        self._encoder: Optional[TiktokenEncoding] = self._resolve_encoder(model)

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        payload: List[Dict[str, Any]] = [message.model_dump() for message in messages]
        completion: Any = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, payload),
            **{**self._default_kwargs, **kwargs},
        )

        choice = self._first_choice(completion)
        return LLMResponse(
            content=self._extract_message_text(choice.message),
            usage=self._build_usage(completion),
            model=completion.model or self._model,
            metadata={"finish_reason": getattr(choice, "finish_reason", None)},
        )

    async def complete_stream(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload: List[Dict[str, Any]] = [message.model_dump() for message in messages]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, payload),
            stream=True,
            **{**self._default_kwargs, **kwargs},
        )

        async def iterator() -> AsyncIterator[str]:
            stream = cast(AsyncIterator[Any], response)
            async for chunk in stream:
                delta = self._first_choice(chunk).delta
                for token in self._extract_delta_tokens(delta):
                    yield token

        return iterator()

    def count_tokens(self, text: str) -> int:
        if self._encoder is None:
            return max(1, len(text) // 4)
        return len(self._encoder.encode(text))

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _first_choice(completion: Any) -> Any:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise ValueError("OpenAI response contained no choices.")
        return choices[0]

    @staticmethod
    def _build_usage(completion: Any) -> Dict[str, int]:
        usage = getattr(completion, "usage", None)
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return "".join(
            part.text
            for part in content or []
            if getattr(part, "text", None)
        )

    @staticmethod
    def _extract_delta_tokens(delta: Any) -> Iterable[str]:
        content = getattr(delta, "content", None)
        if content is None:
            return []
        if isinstance(content, str):
            return [content]
        return [
            part.text
            for part in content
            if getattr(part, "text", None)
        ]

    @staticmethod
    def _resolve_encoder(model: str) -> Optional[TiktokenEncoding]:
        if tiktoken_module is None:
            return None
        try:
            return cast(TiktokenEncoding, tiktoken_module.encoding_for_model(model))
        except KeyError:
            return cast(TiktokenEncoding, tiktoken_module.get_encoding("cl100k_base"))
