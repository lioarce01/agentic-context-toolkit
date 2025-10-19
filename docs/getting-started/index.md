# Getting Started

This quickstart walks through setting up the Agentic Context Toolkit (ACT) locally and running a minimal adaptation loop with the in-memory storage backend.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

ACT requires Python 3.12+, and ships with strict linting/type-checking tools. The default `requirements.txt` installs core dependencies together with development tooling (pytest, mypy, ruff, black, sphinx, myst-parser).

## Minimal Usage Example

```python
import asyncio

from acet import (
    ACTConfig,
    ACETEngine,
    LLMGenerator,
)
from acet.llm.base import BaseLLMProvider, LLMResponse, Message
from acet.storage.memory import MemoryBackend


class EchoProvider(BaseLLMProvider):
    """Minimal provider that echoes the user prompt."""

    async def complete(self, messages: list[Message], **_: object) -> LLMResponse:
        last_user = next(msg.content for msg in reversed(messages) if msg.role == "user")
        return LLMResponse(content=f"ECHO: {last_user}", model="echo", usage={"total_tokens": 0})

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def model_name(self) -> str:
        return "echo-model"


async def main() -> None:
    generator = LLMGenerator(EchoProvider())
    storage = MemoryBackend()
    engine = ACETEngine(
        generator=generator,
        reflector=...,  # supply an acet.reflectors implementation
        curator=...,    # supply an acet.curators implementation
        storage=storage,
        ranker=...,     # supply an acet.retrieval.DeltaRanker
        config=ACTConfig(),
    )

    result = await engine.run_online_adaptation("How do I greet a customer?")
    print(result["answer"])


asyncio.run(main())
```

The example above uses the `MemoryBackend` and a toy `EchoProvider`. In a real deployment you would:

- Configure an actual LLM provider (`acet.llm.providers.OpenAIProvider`, `LiteLLMProvider`, etc.).
- Choose a persistent backend (`acet.storage.sqlite.SQLiteBackend`, `acet.storage.postgres.PostgresBackend`).
- Wire up the standard curator, reflector, and ranker components described in the Guides.

See {doc}`../examples/index` for complete runnable scripts.
