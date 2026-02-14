from __future__ import annotations

from rim.eval.runner import (
    compare_reports,
    evaluate_regression_gate,
    run_single_pass_baseline,
)


def test_compare_reports_computes_deltas() -> None:
    base = {
        "created_at": "2026-02-14T00:00:00Z",
        "dataset_size": 2,
        "mode": "deep",
        "average_runtime_sec": 120.0,
        "average_quality_score": 0.61,
        "runs": [
            {"id": "a", "runtime_sec": 50.0, "quality": {"quality_score": 0.55}},
            {"id": "b", "runtime_sec": 70.0, "quality": {"quality_score": 0.67}},
        ],
    }
    target = {
        "created_at": "2026-02-15T00:00:00Z",
        "dataset_size": 2,
        "mode": "deep",
        "average_runtime_sec": 100.0,
        "average_quality_score": 0.7,
        "runs": [
            {"id": "a", "runtime_sec": 40.0, "quality": {"quality_score": 0.64}},
            {"id": "b", "runtime_sec": 60.0, "quality": {"quality_score": 0.76}},
        ],
    }
    diff = compare_reports(base, target)
    assert diff["average_quality_delta"] == 0.09
    assert diff["average_runtime_delta_sec"] == -20.0
    assert diff["shared_run_count"] == 2


def test_single_pass_baseline_report(tmp_path) -> None:  # noqa: ANN001
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"id":"x1","idea":"Idea A"}',
                '{"id":"x2","idea":"Idea B"}',
            ]
        ),
        encoding="utf-8",
    )
    report = run_single_pass_baseline(dataset_path=dataset, limit=1)
    assert report["mode"] == "single_pass_baseline"
    assert report["dataset_size"] == 1
    assert report["runs"][0]["id"] == "x1"
    assert report["runs"][0]["run_id"] == "baseline-x1"
    assert report["average_quality_score"] == 0.22


def test_regression_gate_passes_within_thresholds() -> None:
    comparison = {
        "average_quality_delta": 0.05,
        "average_runtime_delta_sec": -10.0,
        "shared_run_count": 8,
    }
    gate = evaluate_regression_gate(
        comparison=comparison,
        min_quality_delta=0.02,
        max_runtime_delta_sec=20.0,
        min_shared_runs=5,
    )
    assert gate["passed"] is True
    assert all(check["passed"] for check in gate["checks"])


def test_regression_gate_fails_on_quality_drop() -> None:
    comparison = {
        "average_quality_delta": -0.01,
        "average_runtime_delta_sec": 5.0,
        "shared_run_count": 6,
    }
    gate = evaluate_regression_gate(
        comparison=comparison,
        min_quality_delta=0.0,
        max_runtime_delta_sec=10.0,
        min_shared_runs=3,
    )
    assert gate["passed"] is False
    quality_check = next(item for item in gate["checks"] if item["name"] == "quality_delta")
    assert quality_check["passed"] is False
