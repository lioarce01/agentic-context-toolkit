"""Unit tests for ace.core.budget."""

from ace.core.budget import TokenBudgetManager
from ace.core.models import ContextDelta


def build_delta(guideline: str) -> ContextDelta:
    return ContextDelta(topic="general", guideline=guideline)


def test_format_delta_includes_conditions_and_evidence() -> None:
    delta = ContextDelta(
        topic="safety",
        guideline="Avoid disallowed content.",
        conditions=["user request is risky"],
        evidence=["policy.doc"],
    )
    manager = TokenBudgetManager()

    bullet = manager.format_delta(delta)

    assert bullet.startswith("- If user request is risky")
    assert "[refs: policy.doc]" in bullet


def test_pack_deltas_respects_budget_limit() -> None:
    manager = TokenBudgetManager(budget=10)
    deltas = [build_delta("Respond briefly."), build_delta("Add emoji to replies.")]

    bullets, tokens_used = manager.pack_deltas(deltas)

    assert len(bullets) >= 1
    assert tokens_used <= manager.budget
