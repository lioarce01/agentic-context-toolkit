"""LiteLLM provider implementation."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from acet.llm.base import BaseLLMProvider, LLMResponse, Message

try:  # pragma: no cover - optional dependency
    litellm_module: Any | None = importlib.import_module("litellm")
except ImportError as exc:  # pragma: no cover - optional dependency
    LITELLM_IMPORT_ERROR: ImportError | None = exc
    litellm_module = None
else:
    LITELLM_IMPORT_ERROR = None


class LiteLLMProvider(BaseLLMProvider):
    """Universal LLM provider leveraging LiteLLM routing."""

    def __init__(self, model: str, **default_kwargs: Dict[str, object]) -> None:
        if litellm_module is None:
            raise ImportError(
                "litellm is required for LiteLLMProvider. Install with `pip install litellm`."
            ) from LITELLM_IMPORT_ERROR

        self._model = model
        self._litellm: Any = litellm_module
        self._default_kwargs = default_kwargs

    async def complete(
        self,
        messages: List[Message],
        **kwargs: object,
    ) -> LLMResponse:
        payload = [message.model_dump() for message in messages]
        params = {**self._default_kwargs, **kwargs}
        response = await self._litellm.acompletion(
            model=self._model,
            messages=payload,
            **params,
        )

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

        return LLMResponse(
            content=choice.message.content or "",
            usage=usage_dict,
            model=response.model or self._model,
            metadata={"finish_reason": choice.finish_reason},
        )

    def count_tokens(self, text: str) -> int:
        counter = getattr(self._litellm, "token_counter", None)
        if counter is None:
            return max(1, len(text) // 4)
        return int(counter(model=self._model, text=text))

    @property
    def model_name(self) -> str:
        return self._model
