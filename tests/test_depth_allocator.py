from rim.core.depth_allocator import decide_next_cycle, severity_counts
from rim.core.schemas import CriticFinding


def test_severity_counts_tracks_high_and_critical() -> None:
    findings = [
        CriticFinding(
            node_id="n1",
            critic_type="logic",
            issue="a",
            severity="low",
            confidence=0.7,
            suggested_fix="fix",
            provider="codex",
        ),
        CriticFinding(
            node_id="n1",
            critic_type="evidence",
            issue="b",
            severity="high",
            confidence=0.8,
            suggested_fix="fix",
            provider="claude",
        ),
        CriticFinding(
            node_id="n1",
            critic_type="adversarial",
            issue="c",
            severity="critical",
            confidence=0.9,
            suggested_fix="fix",
            provider="claude",
        ),
    ]
    high, critical = severity_counts(findings)
    assert high == 2
    assert critical == 1


def test_depth_allocator_recurses_when_critical_findings_exist() -> None:
    decision = decide_next_cycle(
        cycle=1,
        max_cycles=3,
        confidence_score=0.92,
        residual_risk_count=0,
        high_severity_findings=1,
        critical_findings=1,
        previous_confidence=None,
        min_confidence_to_stop=0.8,
        max_residual_risks_to_stop=2,
        max_high_findings_to_stop=1,
    )
    assert decision.recurse is True
    assert decision.reason == "critical_findings_present"
    assert decision.next_cycle == 2


def test_depth_allocator_stops_when_stable() -> None:
    decision = decide_next_cycle(
        cycle=1,
        max_cycles=3,
        confidence_score=0.9,
        residual_risk_count=1,
        high_severity_findings=1,
        critical_findings=0,
        previous_confidence=0.82,
        min_confidence_to_stop=0.8,
        max_residual_risks_to_stop=2,
        max_high_findings_to_stop=1,
    )
    assert decision.recurse is False
    assert decision.reason == "stability_reached"
    assert decision.confidence_delta == 0.08


def test_depth_allocator_hard_stops_on_max_cycles() -> None:
    decision = decide_next_cycle(
        cycle=2,
        max_cycles=2,
        confidence_score=0.3,
        residual_risk_count=9,
        high_severity_findings=5,
        critical_findings=2,
        previous_confidence=0.2,
        min_confidence_to_stop=0.8,
        max_residual_risks_to_stop=2,
        max_high_findings_to_stop=1,
    )
    assert decision.recurse is False
    assert decision.reason == "max_cycles_reached"
    assert decision.next_cycle is None
