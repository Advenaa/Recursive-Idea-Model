from __future__ import annotations

from rim.eval.scoring import resolve_rubric_domain, rubric_weights, weighted_quality_score


def test_weighted_quality_score_uses_domain_rubric() -> None:
    rigor, novelty, practicality = 0.8, 0.6, 0.7
    general = weighted_quality_score(rigor, novelty, practicality, domain=None)
    finance = weighted_quality_score(rigor, novelty, practicality, domain="finance")
    consumer = weighted_quality_score(rigor, novelty, practicality, domain="consumer")

    assert round(general, 3) == 0.71
    assert finance > general
    assert consumer < general


def test_rubric_domain_alias_resolution() -> None:
    assert resolve_rubric_domain("fintech") == "finance"
    assert resolve_rubric_domain("devtools") == "developer_tools"
    assert resolve_rubric_domain("unknown-domain") == "general"
    assert rubric_weights("finance")["rigor"] == 0.52
