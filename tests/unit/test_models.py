"""Unit tests for acet.core.models."""

from datetime import UTC, datetime

from acet.core.models import ACETConfig, ContextDelta, DeltaStatus, ReflectionReport


def test_context_delta_defaults_and_mutation() -> None:
    delta = ContextDelta(topic="greeting", guideline="Always say hello first.")

    assert delta.status is DeltaStatus.STAGED
    assert delta.version == 1
    assert delta.score == 0.0
    assert delta.recency == 1.0
    assert delta.conditions == []

    # Updating fields should refresh timestamps
    original_updated = delta.updated_at
    delta.usage_count += 1
    delta.updated_at = datetime.now(UTC)
    assert delta.updated_at > original_updated


def test_reflection_report_nested_models() -> None:
    report = ReflectionReport(
        question="Did we greet the user?",
        answer="Yes, greeting delivered.",
        issues=[
            ReflectionReport.Issue(
                type="omission",
                explanation="Forgot to ask follow-up question.",
                severity=3,
            )
        ],
        proposed_insights=[
            ReflectionReport.ProposedInsight(
                topic="follow_up",
                guideline="Ask the user if they need additional help.",
                confidence=0.7,
            )
        ],
    )

    assert report.issues[0].severity == 3
    assert report.proposed_insights[0].topic == "follow_up"


def test_acet_config_weights_sum_within_bounds() -> None:
    config = ACETConfig()
    total_weight = (
        config.similarity_weight
        + config.recency_weight
        + config.usage_weight
        + config.risk_penalty_weight
    )
    assert 0.0 < total_weight <= 1.0
