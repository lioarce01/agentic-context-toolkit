"""Shared abstractions for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Normalized representation of a chat message."""

    role: str = Field(description="Role within the conversation (system, user, assistant, tool).")
    content: str = Field(description="Message content.")


class LLMResponse(BaseModel):
    """Response structure returned by LLM providers."""

    content: str
    usage: Dict[str, int] = Field(default_factory=dict)
    model: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a completion for the supplied messages."""

    async def complete_stream(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield streamed completion tokens."""
        raise NotImplementedError("Streaming is not supported by this provider.")

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for the given text."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the underlying model."""
