"""Route between multiple LLM providers while sharing a single ACET pipeline."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import structlog

from acet import ACETEngine, ACTConfig, LLMGenerator, StandardCurator
from acet.core.interfaces import EmbeddingProvider, Reflector
from acet.core.models import ReflectionReport
from acet.llm.base import BaseLLMProvider, LLMResponse, Message
from acet.llm.providers import AnthropicProvider, LiteLLMProvider, OpenAIProvider
from acet.retrieval import DeltaRanker
from acet.storage.memory import MemoryBackend


class LengthEmbeddingProvider(EmbeddingProvider):
    async def embed(self, text: str) -> List[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        return 1.0 / (1.0 + abs(emb1[0] - emb2[0]))


class NoOpReflector(Reflector):
    """Placeholder reflector that never proposes new deltas."""

    async def reflect(
        self,
        query: str,
        answer: str,
        evidence: List[str],
        context: List[str],
        ground_truth: Optional[str] = None,
        **_: Any,
    ) -> ReflectionReport:
        return ReflectionReport(question=query, answer=answer)

    async def refine(self, report: ReflectionReport, iterations: int = 3) -> ReflectionReport:
        return report


class EchoProvider(BaseLLMProvider):
    """Fallback provider used when API keys are not available."""

    def __init__(self, label: str) -> None:
        self._label = label

    async def complete(self, messages: List[Message], **_: Any) -> LLMResponse:
        last_user = next(msg.content for msg in reversed(messages) if msg.role == "user")
        return LLMResponse(content=f"[{self._label}] {last_user}", model=self._label, usage={"total_tokens": 0})

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def model_name(self) -> str:
        return f"echo-{self._label}"


def maybe_openai() -> BaseLLMProvider:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return OpenAIProvider(model="gpt-4o-mini", api_key=key)
    return EchoProvider("openai")


def maybe_anthropic() -> BaseLLMProvider:
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return AnthropicProvider(model="claude-3-haiku-20240307", api_key=key)
    return EchoProvider("anthropic")


def maybe_litellm() -> BaseLLMProvider:
    # LiteLLM can proxy many vendors (including local models). We default to an echo provider.
    if os.getenv("LITELLM_API_KEY"):
        return LiteLLMProvider(model="ollama/llama3")
    return EchoProvider("litellm")


def build_engine(provider: BaseLLMProvider) -> ACETEngine:
    embedding_provider = LengthEmbeddingProvider()
    return ACETEngine(
        generator=LLMGenerator(provider),
        reflector=NoOpReflector(),
        curator=StandardCurator(embedding_provider=embedding_provider),
        storage=MemoryBackend(),
        ranker=DeltaRanker(embedding_provider),
        config=ACTConfig(reflection_sample_rate=0.0),  # disable reflection for demo purposes
    )


async def run_demo() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    providers: Dict[str, BaseLLMProvider] = {
        "openai": maybe_openai(),
        "anthropic": maybe_anthropic(),
        "litellm": maybe_litellm(),
    }

    queries = [
        "Draft a friendly welcome message for new subscribers.",
        "Outline the steps for processing a refund.",
        "Summarise the escalation workflow for critical incidents.",
    ]

    for name, provider in providers.items():
        engine = build_engine(provider)
        print(f"\n=== Provider: {name} ({provider.model_name}) ===")
        for query in queries:
            result = await engine.run_online_adaptation(query=query, update_context=False)
            print(f"Q: {query}\nA: {result['answer']}\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
