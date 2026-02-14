from rim.agents.executable_verifier import run_executable_verification
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


def test_executable_verification_runs_prefixed_checks() -> None:
    synthesis = {
        "confidence_score": 0.9,
        "changes_summary": ["a", "b"],
        "residual_risks": [],
        "next_experiments": ["x"],
    }
    payload = run_executable_verification(
        constraints=[
            "python: confidence_score >= 0.8",
            "assert: change_count >= 2",
            "plain text constraint",
        ],
        synthesis=synthesis,
        findings=[_finding("risk")],
        max_checks=5,
    )
    assert payload["summary"]["total_checks"] == 2
    assert payload["summary"]["failed_checks"] == 0


def test_executable_verification_flags_failures_and_errors() -> None:
    synthesis = {
        "confidence_score": 0.4,
        "changes_summary": ["a"],
        "residual_risks": ["r1", "r2"],
        "next_experiments": [],
    }
    payload = run_executable_verification(
        constraints=[
            "python: confidence_score > 0.7",
            "py: unknown_name == 1",
        ],
        synthesis=synthesis,
        findings=[_finding("risk", severity="critical")],
        max_checks=5,
    )
    assert payload["summary"]["total_checks"] == 2
    assert payload["summary"]["failed_checks"] == 2
    assert payload["summary"]["execution_errors"] == 1
    assert any("Unknown variable" in str(item.get("error")) for item in payload["checks"])


def test_executable_verification_skips_when_no_prefixed_checks() -> None:
    payload = run_executable_verification(
        constraints=["must include compliance plan"],
        synthesis={},
        findings=[],
        max_checks=5,
    )
    assert payload["summary"]["total_checks"] == 0
    assert payload["summary"]["skipped"] is True


def test_executable_verification_python_exec_disabled_by_default() -> None:
    payload = run_executable_verification(
        constraints=["python_exec: passed = context['change_count'] >= 1"],
        synthesis={"changes_summary": ["a"]},
        findings=[],
        max_checks=5,
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 1
    assert payload["checks"][0]["check_type"] == "python_exec"
    assert "disabled" in str(payload["checks"][0]["error"])


def test_executable_verification_python_exec_enabled() -> None:
    payload = run_executable_verification(
        constraints=["python_exec: passed = context['change_count'] >= 1"],
        synthesis={"changes_summary": ["a"]},
        findings=[],
        max_checks=5,
        enable_python_exec=True,
        python_exec_timeout_sec=2,
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 0
    assert payload["checks"][0]["check_type"] == "python_exec"
