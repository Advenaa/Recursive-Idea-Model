from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "advanced_verify_adapter.py"


def _run_adapter(payload: dict) -> dict:
    completed = subprocess.run(
        [sys.executable, str(_script_path())],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert completed.returncode == 0
    return json.loads((completed.stdout or "").strip())


def test_advanced_verify_adapter_solver_roundtrip() -> None:
    payload = {
        "check_type": "solver",
        "payload": "confidence_score >= 0.6 and risk_count <= 2",
        "context": {
            "confidence_score": 0.8,
            "change_count": 2,
            "risk_count": 1,
            "experiment_count": 1,
            "finding_count": 2,
            "high_finding_count": 1,
            "critical_finding_count": 0,
        },
        "synthesis": {},
    }
    result = _run_adapter(payload)
    assert result["passed"] is True
    assert result["error"] is None


def test_advanced_verify_adapter_data_roundtrip(tmp_path: Path) -> None:
    data = tmp_path / "reference.jsonl"
    data.write_text('{"idea":"include compliance audit controls"}\n', encoding="utf-8")
    payload = {
        "check_type": "data_reference",
        "payload": f"compliance,audit|path={data}|min_overlap=0.5",
        "context": {},
        "synthesis": {
            "synthesized_idea": "include compliance audit controls",
            "changes_summary": [],
            "next_experiments": [],
        },
    }
    result = _run_adapter(payload)
    assert result["passed"] is True
    assert result["error"] is None


def test_advanced_verify_adapter_handles_bad_request() -> None:
    payload = {
        "check_type": "solver",
        "payload": "unknown_name > 0",
        "context": {"confidence_score": 0.1},
        "synthesis": {},
    }
    result = _run_adapter(payload)
    assert result["passed"] is False
    assert "unknown variable" in str(result["error"]).lower()
