"""Minimal retrieval-augmented workflow powered by ACET."""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import structlog

from acet import ACETConfig, ACETEngine, LLMGenerator, StandardCurator
from acet.core.interfaces import EmbeddingProvider, Reflector
from acet.core.models import ReflectionReport
from acet.llm.base import BaseLLMProvider, LLMResponse, Message
from acet.retrieval import DeltaRanker
from acet.storage.memory import MemoryBackend


class EchoProvider(BaseLLMProvider):
    """Toy LLM provider that echoes the user prompt."""

    async def complete(self, messages: List[Message], **_: Any) -> LLMResponse:
        last_user = next(msg.content for msg in reversed(messages) if msg.role == "user")
        return LLMResponse(content=f"[echo] {last_user}", model="echo-model", usage={"total_tokens": 0})

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def model_name(self) -> str:
        return "echo-model"


class SimpleEmbeddingProvider(EmbeddingProvider):
    """Length-based embedding provider suitable for demos."""

    async def embed(self, text: str) -> List[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        return 1.0 / (1.0 + abs(emb1[0] - emb2[0]))


class SimpleReflector(Reflector):
    """Reflector that proposes deltas based on keyword matching."""

    async def reflect(
        self,
        query: str,
        answer: str,
        evidence: List[str],
        context: List[str],
        ground_truth: Optional[str] = None,
        **_: Any,
    ) -> ReflectionReport:
        proposed: List[ReflectionReport.ProposedInsight] = []
        if "refund" in query.lower():
            proposed.append(
                ReflectionReport.ProposedInsight(
                    topic="refunds",
                    guideline="Always clarify the 30-day refund policy with receipt requirement.",
                    evidence=["support playbook"],
                    tags=["policy"],
                    confidence=0.75,
                )
            )
        return ReflectionReport(
            question=query,
            answer=answer,
            evidence_refs=evidence,
            issues=[],
            proposed_insights=proposed,
        )

    async def refine(self, report: ReflectionReport, iterations: int = 3) -> ReflectionReport:
        return report


async def main() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    embedding_provider = SimpleEmbeddingProvider()

    engine = ACETEngine(
        generator=LLMGenerator(EchoProvider()),
        reflector=SimpleReflector(),
        curator=StandardCurator(embedding_provider=embedding_provider),
        storage=MemoryBackend(),
        ranker=DeltaRanker(embedding_provider),
        config=ACETConfig(token_budget=400, reflection_sample_rate=1.0),
    )

    # Offline bootstrapping
    training_data = [
        {"query": "How do I reset my password?", "ground_truth": "Direct to reset portal."},
        {"query": "Customer asking about refund timeline", "ground_truth": "Explain 30-day policy."},
    ]
    stats = await engine.run_offline_adaptation(training_data, epochs=1)
    structlog.get_logger(__name__).info("offline_complete", stats=stats)

    # Online query
    online_result = await engine.run_online_adaptation(
        query="What should I say about the refund window?",
        update_context=True,
    )

    print("Answer:", online_result["answer"])
    print("Injected context:", online_result["injected_context"])
    print("Created deltas:", [delta.guideline for delta in online_result["created_deltas"]])


if __name__ == "__main__":
    asyncio.run(main())
