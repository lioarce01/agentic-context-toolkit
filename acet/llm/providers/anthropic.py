"""Anthropic provider implementation."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple, cast

from acet.llm.base import BaseLLMProvider, LLMResponse, Message

try:  # pragma: no cover - optional dependency
    anthropic_module: Any | None = importlib.import_module("anthropic")
except ImportError as exc:  # pragma: no cover - optional dependency
    ANTHROPIC_IMPORT_ERROR: ImportError | None = exc
    anthropic_module = None
else:
    ANTHROPIC_IMPORT_ERROR = None


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider wrapper."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **default_kwargs: Dict[str, object],
    ) -> None:
        if anthropic_module is None:
            raise ImportError(
                "anthropic is required for AnthropicProvider. Install with `pip install anthropic`."
            ) from ANTHROPIC_IMPORT_ERROR

        self._model = model
        client_cls = cast(type[Any], anthropic_module.AsyncAnthropic)
        self._client: Any = client_cls(api_key=api_key)
        self._default_kwargs = default_kwargs

    async def complete(
        self,
        messages: List[Message],
        **kwargs: object,
    ) -> LLMResponse:
        system, content_messages = self._separate_system(messages)
        params = {**self._default_kwargs, **kwargs}

        response = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=content_messages,
            **params,
        )

        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": (
                (getattr(usage, "input_tokens", 0) or 0)
                + (getattr(usage, "output_tokens", 0) or 0)
            ),
        }

        content_items = getattr(response, "content", []) or []
        content = ""
        if content_items:
            first_item = content_items[0]
            content = getattr(first_item, "text", "") or ""

        return LLMResponse(
            content=content,
            usage=usage_dict,
            model=response.model or self._model,
        )

    def count_tokens(self, text: str) -> int:
        # Claude tokenizer is proprietary; approximation of 4 characters per token.
        return max(1, len(text) // 4)

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _separate_system(messages: List[Message]) -> Tuple[Optional[str], List[Dict[str, str]]]:
        system: Optional[str] = None
        converted: List[Dict[str, str]] = []
        for message in messages:
            if message.role == "system" and system is None:
                system = message.content
                continue
            converted.append({"role": message.role, "content": message.content})
        return system, converted
