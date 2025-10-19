# Getting Started

This quickstart shows how to install ACET, run the bundled demo, and assemble your own engine configuration.

## Prerequisites

- Python 3.12 or newer
- `pip` + virtualenv (recommended)

## Install ACET locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The default `requirements.txt` installs the core ACET library plus development tooling (pytest, mypy, ruff, black, sphinx).

## Run the sample workflow

The quickest way to see ACET in action is the simple retrieval-augmented loop:

```bash
python examples/simple_rag.py
```

This script boots an in-memory engine, primes it with two labelled queries, and serves a live question while reflection/curation run in the background. Check the console output for curated delta IDs and injected context bullets.

## Wire up your own engine

Below is a minimal example that keeps everything in memory. Replace the placeholder components with production-ready implementations (LLM provider, persistent storage, real embedding model) when you integrate ACET into your stack.

```python
import asyncio
from typing import Any, List, Optional

from acet import ACETConfig, ACETEngine, LLMGenerator, StandardCurator
from acet.core.interfaces import EmbeddingProvider, Reflector
from acet.core.models import ReflectionReport
from acet.llm.base import BaseLLMProvider, LLMResponse, Message
from acet.retrieval import DeltaRanker
from acet.storage.memory import MemoryBackend


class EchoProvider(BaseLLMProvider):
    async def complete(self, messages: List[Message], **_: Any) -> LLMResponse:
        last_user = next(msg.content for msg in reversed(messages) if msg.role == "user")
        return LLMResponse(content=f"[echo] {last_user}", model="echo", usage={"total_tokens": 0})

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def model_name(self) -> str:
        return "echo-model"


class SimpleEmbeddingProvider(EmbeddingProvider):
    async def embed(self, text: str) -> List[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        return 1.0 / (1.0 + abs(emb1[0] - emb2[0]))


class SimpleReflector(Reflector):
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


async def main() -> None:
    embedding = SimpleEmbeddingProvider()

    engine = ACETEngine(
        generator=LLMGenerator(EchoProvider()),
        reflector=SimpleReflector(),
        curator=StandardCurator(embedding_provider=embedding),
        storage=MemoryBackend(),
        ranker=DeltaRanker(embedding),
        config=ACETConfig(token_budget=400, reflection_sample_rate=1.0),
    )

    result = await engine.run_online_adaptation(
        query="What should I mention about refunds?",
        update_context=True,
    )

    print(result["answer"])
    print(result["injected_context"])


if __name__ == "__main__":
    asyncio.run(main())
```

Next steps:

- Swap in `acet.llm.providers.*` to call a real model endpoint.
- Move from `MemoryBackend` to `SQLiteBackend` or `PostgresBackend` for persistence.
- Explore {doc}`../guides/offline-online-adaptation` to combine offline bootstrapping with online learning.
