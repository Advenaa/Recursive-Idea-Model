from __future__ import annotations

import sys
from pathlib import Path

import rim.agents.advanced_verifier as advanced_verifier_module
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


def test_advanced_verifier_supports_http_data_reference_via_url_option(
    monkeypatch,  # noqa: ANN001
) -> None:
    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self._payload = text.encode("utf-8")

        def read(self, size: int = -1) -> bytes:
            if size < 0:
                return self._payload
            return self._payload[:size]

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

    def fake_urlopen(request, timeout=0):  # noqa: ANN001, ANN202
        _ = request
        _ = timeout
        return _FakeResponse("compliance audit controls staged rollout")

    monkeypatch.setattr(advanced_verifier_module, "urlopen", fake_urlopen)
    synthesis = {
        "synthesized_idea": "Include compliance audit controls and staged rollout.",
        "changes_summary": ["add audit controls"],
        "residual_risks": [],
        "next_experiments": ["run staged pilot"],
        "confidence_score": 0.82,
    }
    payload = run_advanced_verification(
        constraints=[
            "data: compliance,audit | url=https://docs.example.com/reference.txt | min_overlap=0.5",
        ],
        synthesis=synthesis,
        findings=[],
        max_checks=3,
        allow_http_data_reference=True,
        http_data_timeout_sec=2,
        http_data_max_bytes=4096,
        http_data_allowed_hosts=["example.com"],
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 0
    check = payload["checks"][0]
    assert check["check_type"] == "data_reference"
    assert check["result"]["source_kind"] == "url"
    assert check["result"]["source"] == "https://docs.example.com/reference.txt"


def test_advanced_verifier_rejects_http_data_reference_when_disabled() -> None:
    synthesis = {
        "synthesized_idea": "Include compliance audit controls.",
        "changes_summary": [],
        "residual_risks": [],
        "next_experiments": [],
        "confidence_score": 0.8,
    }
    payload = run_advanced_verification(
        constraints=[
            "data: compliance,audit | url=https://example.com/reference.txt | min_overlap=0.5",
        ],
        synthesis=synthesis,
        findings=[],
        max_checks=3,
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 1
    assert "disabled" in str(payload["checks"][0].get("error")).lower()


def test_advanced_verifier_rejects_http_data_reference_when_host_not_allowed(
    monkeypatch,  # noqa: ANN001
) -> None:
    called = {"value": False}

    def fake_urlopen(request, timeout=0):  # noqa: ANN001, ANN202
        _ = request
        _ = timeout
        called["value"] = True
        raise AssertionError("urlopen should not be called when host is not allowed")

    monkeypatch.setattr(advanced_verifier_module, "urlopen", fake_urlopen)
    synthesis = {
        "synthesized_idea": "Include compliance audit controls.",
        "changes_summary": [],
        "residual_risks": [],
        "next_experiments": [],
        "confidence_score": 0.8,
    }
    payload = run_advanced_verification(
        constraints=[
            "data: compliance,audit | url=https://example.com/reference.txt | min_overlap=0.5",
        ],
        synthesis=synthesis,
        findings=[],
        max_checks=3,
        allow_http_data_reference=True,
        http_data_allowed_hosts=["docs.example.com"],
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 1
    assert "allowed host list" in str(payload["checks"][0].get("error")).lower()
    assert called["value"] is False


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


def test_advanced_verifier_can_use_repo_adapter_for_multiple_check_types(tmp_path: Path) -> None:
    adapter = Path(__file__).resolve().parents[1] / "scripts" / "advanced_verify_adapter.py"
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        "\n".join(
            [
                '{"idea":"include compliance audit controls"}',
                '{"idea":"run staged pilot"}',
            ]
        ),
        encoding="utf-8",
    )
    cmd = f"{sys.executable} {adapter}"
    synthesis = {
        "synthesized_idea": "include compliance audit controls",
        "changes_summary": [],
        "residual_risks": [],
        "next_experiments": ["run staged pilot"],
        "confidence_score": 0.8,
    }
    payload = run_advanced_verification(
        constraints=[
            "solver: confidence_score >= 0.75",
            "simulate: confidence_score >= 0.6 | trials=60 | min_pass_rate=0.7",
            f"data: compliance,audit|path={reference}|min_overlap=0.5",
        ],
        synthesis=synthesis,
        findings=[],
        max_checks=5,
        external_solver_cmd=cmd,
        external_simulation_cmd=cmd,
        external_data_cmd=cmd,
        external_timeout_sec=5,
    )
    assert payload["summary"]["total_checks"] == 3
    assert payload["summary"]["failed_checks"] == 0
    assert all(item.get("adapter") == "external" for item in payload["checks"])


def test_advanced_verifier_runs_formal_solver_with_z3_or_fallback(
    monkeypatch,  # noqa: ANN001
) -> None:
    monkeypatch.delenv("RIM_ADV_VERIFY_FORMAL_ALLOW_AST_FALLBACK", raising=False)
    synthesis = {
        "synthesized_idea": "Idea",
        "changes_summary": [],
        "residual_risks": [],
        "next_experiments": [],
        "confidence_score": 0.8,
    }
    payload = run_advanced_verification(
        constraints=["formal: confidence_score >= 0.75 and risk_count <= 1"],
        synthesis=synthesis,
        findings=[],
        max_checks=3,
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 0
    check = payload["checks"][0]
    assert check["check_type"] == "formal_solver"
    backend = ((check.get("result") or {}).get("backend") if isinstance(check.get("result"), dict) else None)
    assert backend in {"z3_formal", "ast_context_fallback"}
    assert check["mode"] == "prove"


def test_advanced_verifier_formal_solver_returns_counterexample_for_failed_proof(
    monkeypatch,  # noqa: ANN001
) -> None:
    monkeypatch.delenv("RIM_ADV_VERIFY_FORMAL_ALLOW_AST_FALLBACK", raising=False)
    synthesis = {
        "synthesized_idea": "Idea",
        "changes_summary": [],
        "residual_risks": [],
        "next_experiments": [],
        "confidence_score": 0.8,
    }
    payload = run_advanced_verification(
        constraints=[
            "formal: risk_count < 0 | mode=prove | assume=confidence_score >= 0;confidence_score <= 1",
        ],
        synthesis=synthesis,
        findings=[],
        max_checks=3,
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 1
    check = payload["checks"][0]
    assert check["check_type"] == "formal_solver"
    assert check["mode"] == "prove"
    if isinstance(check.get("result"), dict) and check["result"].get("backend") == "z3_formal":
        assert "counterexample" in check["result"]


def test_advanced_verifier_formal_solver_supports_satisfiable_mode() -> None:
    synthesis = {
        "synthesized_idea": "Idea",
        "changes_summary": [],
        "residual_risks": [],
        "next_experiments": [],
        "confidence_score": 0.8,
    }
    payload = run_advanced_verification(
        constraints=[
            "formal: confidence_score >= 0.5 and risk_count >= 0 | mode=satisfiable",
        ],
        synthesis=synthesis,
        findings=[],
        max_checks=3,
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 0
    check = payload["checks"][0]
    assert check["check_type"] == "formal_solver"
    assert check["mode"] == "satisfiable"


def test_advanced_verifier_formal_solver_can_disable_ast_fallback(
    monkeypatch,  # noqa: ANN001
) -> None:
    monkeypatch.setenv("RIM_ADV_VERIFY_FORMAL_ALLOW_AST_FALLBACK", "0")
    monkeypatch.setattr(
        advanced_verifier_module,
        "_z3_from_ast",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("z3 backend unavailable")),  # noqa: ARG005
    )
    synthesis = {
        "synthesized_idea": "Idea",
        "changes_summary": [],
        "residual_risks": [],
        "next_experiments": [],
        "confidence_score": 0.8,
    }
    payload = run_advanced_verification(
        constraints=["formal: confidence_score >= 0.75"],
        synthesis=synthesis,
        findings=[],
        max_checks=3,
    )
    assert payload["summary"]["total_checks"] == 1
    assert payload["summary"]["failed_checks"] == 1
    assert payload["summary"]["execution_errors"] == 1
    assert "z3 backend unavailable" in str(payload["checks"][0].get("error"))
