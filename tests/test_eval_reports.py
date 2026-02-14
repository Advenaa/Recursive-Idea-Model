from __future__ import annotations

from rim.eval.runner import compare_reports


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
