"""Anthropic provider implementation."""

from __future__ import annotations

from typing import Dict, List, Optional

from ace.llm.base import BaseLLMProvider, LLMResponse, Message

try:  # pragma: no cover - optional dependency
    from anthropic import AsyncAnthropic
except ImportError as exc:  # pragma: no cover - optional dependency
    AsyncAnthropic = None  # type: ignore[assignment]
    ANTHROPIC_IMPORT_ERROR = exc
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
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic is required for AnthropicProvider. Install with `pip install anthropic`."
            ) from ANTHROPIC_IMPORT_ERROR

        self._model = model
        self._client = AsyncAnthropic(api_key=api_key)
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

        usage = response.usage or {}
        usage_dict = {
            "prompt_tokens": usage.input_tokens or 0,
            "completion_tokens": usage.output_tokens or 0,
            "total_tokens": (usage.input_tokens or 0) + (usage.output_tokens or 0),
        }

        content = response.content[0].text if response.content else ""

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
    def _separate_system(messages: List[Message]):
        system = None
        converted = []
        for message in messages:
            if message.role == "system" and system is None:
                system = message.content
                continue
            converted.append({"role": message.role, "content": message.content})
        return system, converted
