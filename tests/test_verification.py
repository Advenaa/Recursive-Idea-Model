from rim.agents.verification import verify_synthesis
from rim.core.schemas import CriticFinding


def _finding(issue: str, severity: str = "high") -> CriticFinding:
    return CriticFinding(
        node_id="n1",
        critic_type="logic",
        issue=issue,
        severity=severity,
        confidence=0.8,
        suggested_fix="fix",
        provider="codex",
    )


def test_verify_synthesis_captures_constraint_and_risk_failures() -> None:
    synthesis = {
        "synthesized_idea": "Launch quick pilot with manual onboarding only.",
        "changes_summary": ["Reduce scope for initial validation."],
        "residual_risks": [],
        "next_experiments": ["Run 10-user pilot this week."],
        "confidence_score": 0.8,
    }
    payload = verify_synthesis(
        synthesis=synthesis,
        findings=[_finding("No compliance plan for regulated data")],
        constraints=["Must include compliance plan and audit trail"],
        min_constraint_overlap=0.7,
        min_finding_overlap=0.5,
    )
    assert payload["summary"]["total_checks"] >= 3
    assert payload["summary"]["failed_checks"] >= 2
    assert payload["summary"]["critical_failures"] >= 2


def test_verify_synthesis_passes_when_output_covers_requirements() -> None:
    synthesis = {
        "synthesized_idea": "Ship a compliance-first rollout with audit trail and access controls.",
        "changes_summary": [
            "Added compliance plan with audit trail.",
            "Added risk mitigations for regulated data handling.",
        ],
        "residual_risks": [],
        "next_experiments": ["Validate audit logs with security team."],
        "confidence_score": 0.9,
    }
    payload = verify_synthesis(
        synthesis=synthesis,
        findings=[_finding("Need regulated data compliance and audit controls", severity="critical")],
        constraints=["Include compliance plan and audit trail"],
        min_constraint_overlap=0.4,
        min_finding_overlap=0.3,
    )
    assert payload["summary"]["failed_checks"] == 0
    assert payload["summary"]["verification_score"] == 1.0
