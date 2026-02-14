from rim.agents.reconciliation import reconcile_findings
from rim.core.schemas import CriticFinding


def _finding(
    *,
    node_id: str,
    critic_type: str,
    issue: str,
    severity: str = "medium",
    confidence: float = 0.8,
) -> CriticFinding:
    return CriticFinding(
        node_id=node_id,
        critic_type=critic_type,
        issue=issue,
        severity=severity,
        confidence=confidence,
        suggested_fix="fix",
        provider="codex",
    )


def test_reconcile_detects_consensus_and_disagreement() -> None:
    findings = [
        _finding(node_id="n1", critic_type="logic", issue="Missing feasibility plan", severity="high"),
        _finding(node_id="n1", critic_type="evidence", issue="missing feasibility plan", severity="high"),
        _finding(node_id="n1", critic_type="execution", issue="Missing feasibility plan", severity="critical"),
        _finding(node_id="n1", critic_type="adversarial", issue="No risk mitigation path", severity="medium"),
        _finding(node_id="n2", critic_type="logic", issue="Assumption is weak", severity="medium"),
    ]

    payload = reconcile_findings(
        findings,
        consensus_min_agents=3,
        consensus_min_confidence=0.7,
        min_unique_critics_per_node=1,
        max_single_critic_share=1.0,
    )
    assert payload["summary"]["total_findings"] == 5
    assert payload["summary"]["consensus_count"] == 1
    assert payload["summary"]["disagreement_count"] == 1
    assert payload["consensus_flaws"][0]["node_id"] == "n1"
    assert payload["consensus_flaws"][0]["support_count"] == 3
    assert payload["consensus_flaws"][0]["severity"] == "critical"
    assert payload["disagreements"][0]["node_id"] == "n1"
    assert payload["diversity_guardrails"]["summary"]["flagged_count"] == 0


def test_reconcile_empty_input_returns_empty_summary() -> None:
    payload = reconcile_findings([])
    assert payload["summary"] == {
        "total_findings": 0,
        "consensus_count": 0,
        "disagreement_count": 0,
    }
    assert payload["consensus_flaws"] == []
    assert payload["disagreements"] == []
    assert payload["diversity_guardrails"]["summary"]["nodes_evaluated"] == 0
    assert payload["diversity_guardrails"]["summary"]["flagged_count"] == 0


def test_reconcile_flags_low_diversity_nodes() -> None:
    findings = [
        _finding(node_id="n1", critic_type="logic", issue="Issue A", confidence=0.7),
        _finding(node_id="n1", critic_type="logic", issue="Issue B", confidence=0.7),
        _finding(node_id="n1", critic_type="logic", issue="Issue C", confidence=0.7),
        _finding(node_id="n2", critic_type="logic", issue="Issue D", confidence=0.7),
        _finding(node_id="n2", critic_type="execution", issue="Issue E", confidence=0.7),
        _finding(node_id="n2", critic_type="evidence", issue="Issue F", confidence=0.7),
    ]
    payload = reconcile_findings(
        findings,
        min_unique_critics_per_node=3,
        max_single_critic_share=0.6,
    )
    flags = payload["diversity_guardrails"]["flagged_nodes"]
    assert len(flags) == 1
    assert flags[0]["node_id"] == "n1"
    assert flags[0]["dominant_critic"] == "logic"
