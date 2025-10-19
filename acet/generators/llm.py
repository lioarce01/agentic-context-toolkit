"""Generator implementations for ACET."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from acet.core.budget import TokenBudgetManager
from acet.core.interfaces import Generator
from acet.llm.base import BaseLLMProvider, Message


class LLMGenerator(Generator):
    """Generator that delegates completions to an injected LLM provider."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        system_prompt: str = "You are a helpful assistant.",
        budget_manager: Optional[TokenBudgetManager] = None,
    ) -> None:
        self.llm = llm_provider
        self.system_prompt = system_prompt
        self.budget_manager = budget_manager or TokenBudgetManager(model=llm_provider.model_name)

    async def generate(
        self,
        query: str,
        context: List[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        messages = self._build_messages(query, context)
        response = await self.llm.complete(messages, **kwargs)

        return {
            "answer": response.content,
            "evidence": [],
            "reasoning": "",
            "metadata": {
                "model": response.model,
                "usage": response.usage,
                "context_bullets": len(context),
            },
        }

    def _build_messages(self, query: str, context: List[str]) -> List[Message]:
        if context:
            context_block = "\n".join(context)
            system_content = (
                f"{self.system_prompt}\n\n### Context Playbook\n"
                f"The following insights should guide your answer:\n{context_block}"
            )
        else:
            system_content = self.system_prompt

        return [
            Message(role="system", content=system_content),
            Message(role="user", content=query),
        ]

