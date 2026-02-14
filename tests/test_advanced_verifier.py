from __future__ import annotations

import sys
from pathlib import Path

from rim.agents.advanced_verifier import run_advanced_verification
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


def test_advanced_verifier_runs_solver_simulation_and_data_checks(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        "\n".join(
            [
                '{"idea":"Include compliance audit controls for rollout"}',
                '{"idea":"Keep risk count low with staged experiments"}',
            ]
        ),
        encoding="utf-8",
    )
    synthesis = {
        "synthesized_idea": "Include compliance audit controls and staged rollout.",
        "changes_summary": ["added compliance controls"],
        "residual_risks": ["minor onboarding risk"],
        "next_experiments": ["run staged pilot"],
        "confidence_score": 0.82,
    }
    payload = run_advanced_verification(
        constraints=[
            "solver: confidence_score >= 0.8 and risk_count <= 2",
            "simulate: confidence_score >= 0.6 | trials=120 | min_pass_rate=0.7",
            f"data: compliance, audit | path={reference} | min_overlap=0.5",
        ],
        synthesis=synthesis,
        findings=[_finding("risk issue", severity="medium")],
        max_checks=5,
        simulation_seed=7,
    )
    assert payload["summary"]["total_checks"] == 3
    assert payload["summary"]["failed_checks"] == 0
    assert payload["summary"]["execution_errors"] == 0


def test_advanced_verifier_flags_failures_and_errors(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    reference.write_text('{"idea":"baseline data"}\n', encoding="utf-8")
    synthesis = {
        "synthesized_idea": "basic idea",
        "changes_summary": [],
        "residual_risks": ["r1", "r2", "r3"],
        "next_experiments": [],
        "confidence_score": 0.35,
    }
    payload = run_advanced_verification(
        constraints=[
            "solver: unknown_name > 0",
            "simulate: confidence_score >= 0.95 | trials=60 | min_pass_rate=0.9",
            f"data: compliance, audit | path={reference} | min_overlap=0.9 | mode=all",
        ],
        synthesis=synthesis,
        findings=[_finding("critical issue", severity="critical")],
        max_checks=5,
        simulation_seed=11,
    )
    assert payload["summary"]["total_checks"] == 3
    assert payload["summary"]["failed_checks"] == 3
    assert payload["summary"]["execution_errors"] >= 1
    assert any("Unknown variable" in str(item.get("error")) for item in payload["checks"])


def test_advanced_verifier_skips_when_no_prefixed_checks() -> None:
    payload = run_advanced_verification(
        constraints=["plain natural language constraint"],
        synthesis={"confidence_score": 0.5},
        findings=[],
        max_checks=5,
    )
    assert payload["summary"]["total_checks"] == 0
    assert payload["summary"]["skipped"] is True


def test_advanced_verifier_can_use_external_solver_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "external_solver.py"
    adapter.write_text(
        "\n".join(
            [
                "import json, sys",
                "request = json.loads(sys.stdin.read() or '{}')",
                "if request.get('check_type') != 'solver':",
                "    print(json.dumps({'passed': False, 'error': 'unsupported'}))",
                "    raise SystemExit(0)",
                "print(json.dumps({'passed': True, 'result': {'backend': 'external-solver'}}))",
            ]
        ),
        encoding="utf-8",
    )
    synthesis = {
        "synthesized_idea": "Idea",
        "changes_summary": [],
        "residual_risks": [],
        "next_experiments": [],
        "confidence_score": 0.4,
    }
    payload = run_advanced_verification(
        constraints=["solver: unknown_name > 0"],
        synthesis=synthesis,
        findings=[],
        max_checks=3,
        external_solver_cmd=f"{sys.executable} {adapter}",
        external_timeout_sec=3,
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 0
    assert payload["checks"][0]["adapter"] == "external"
