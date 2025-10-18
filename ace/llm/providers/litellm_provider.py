"""LiteLLM provider implementation."""

from __future__ import annotations

from typing import Dict, List

from ace.llm.base import BaseLLMProvider, LLMResponse, Message

try:  # pragma: no cover - optional dependency
    import litellm
except ImportError as exc:  # pragma: no cover - optional dependency
    litellm = None
    LITELLM_IMPORT_ERROR = exc
else:
    LITELLM_IMPORT_ERROR = None


class LiteLLMProvider(BaseLLMProvider):
    """Universal LLM provider leveraging LiteLLM routing."""

    def __init__(self, model: str, **default_kwargs: Dict[str, object]) -> None:
        if litellm is None:
            raise ImportError(
                "litellm is required for LiteLLMProvider. Install with `pip install litellm`."
            ) from LITELLM_IMPORT_ERROR

        self._model = model
        self._default_kwargs = default_kwargs

    async def complete(
        self,
        messages: List[Message],
        **kwargs: object,
    ) -> LLMResponse:
        payload = [message.model_dump() for message in messages]
        params = {**self._default_kwargs, **kwargs}
        response = await litellm.acompletion(model=self._model, messages=payload, **params)

        choice = response.choices[0]
        usage = response.usage or {}
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens or 0,
            "completion_tokens": usage.completion_tokens or 0,
            "total_tokens": usage.total_tokens or 0,
        }

        return LLMResponse(
            content=choice.message.content or "",
            usage=usage_dict,
            model=response.model or self._model,
            metadata={"finish_reason": choice.finish_reason},
        )

    def count_tokens(self, text: str) -> int:
        return litellm.token_counter(model=self._model, text=text)

    @property
    def model_name(self) -> str:
        return self._model
