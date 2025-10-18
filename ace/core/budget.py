"""Token budget management utilities."""

from __future__ import annotations

import importlib
from typing import Any, List, Optional, Tuple

from .models import ContextDelta

try:
    tiktoken_module = importlib.import_module("tiktoken")
except ImportError as exc:
    raise ImportError(
        "tiktoken is required for TokenBudgetManager; install with `pip install tiktoken`."
    ) from exc

Encoding = Any


class TokenBudgetManager:
    """Keeps track of the token budget available for context injection."""

    def __init__(self, model: Optional[str] = None, budget: int = 800) -> None:
        """
        Args:
            model: Identifier for the tokenizer to use. When omitted, falls back to
                a broadly compatible encoding.
            budget: Maximum number of tokens available for context injection.
        """
        self.model = model
        self.budget = budget
        self.encoder: Encoding = self._resolve_encoder(model)

    def _resolve_encoder(self, model: Optional[str]) -> Encoding:
        """Return a tokenizer encoding compatible with the given model."""
        try:
            if model:
                return tiktoken_module.encoding_for_model(model)
        except KeyError:
            pass

        # Default to a widely supported encoding when no specific model is supplied.
        return tiktoken_module.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens consumed by the given text."""
        return len(self.encoder.encode(text))

    def format_delta(self, delta: ContextDelta) -> str:
        """Render a delta as a bullet suitable for prompt injection."""
        conditions = ", ".join(delta.conditions) if delta.conditions else "always"
        evidence = ", ".join(delta.evidence[:3]) if delta.evidence else ""

        bullet = f"- If {conditions}, then {delta.guideline}"
        if evidence:
            bullet += f" [refs: {evidence}]"

        return bullet

    def pack_deltas(
        self,
        deltas: List[ContextDelta],
        budget: Optional[int] = None,
    ) -> Tuple[List[str], int]:
        """Pack deltas into the available budget and return bullets plus token usage."""
        limit = budget if budget is not None else self.budget
        bullets: List[str] = []
        tokens_used = 0

        for delta in deltas:
            bullet = self.format_delta(delta)
            bullet_tokens = self.count_tokens(bullet)

            if tokens_used + bullet_tokens > limit:
                break

            bullets.append(bullet)
            tokens_used += bullet_tokens

        return bullets, tokens_used
