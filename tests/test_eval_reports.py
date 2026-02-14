from __future__ import annotations

import asyncio

from rim.core.schemas import AnalyzeResult, DecompositionNode
from rim.eval import runner
from rim.eval.runner import (
    build_blind_review_packet,
    compare_reports,
    evaluate_regression_gate,
    run_benchmark,
    run_duel_benchmark,
    run_single_pass_baseline,
    save_blind_review_packet,
    save_report,
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


class MixedOutcomeOrchestrator:
    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, request):  # noqa: ANN001, ANN201
        self.calls += 1
        if "fail" in request.idea.lower():
            raise RuntimeError("simulated provider error")
        return AnalyzeResult(
            run_id=f"run-{self.calls}",
            mode=request.mode,
            input_idea=request.idea,
            decomposition=[
                DecompositionNode(
                    depth=0,
                    component_text=request.idea,
                    node_type="claim",
                    confidence=0.4,
                )
            ],
            critic_findings=[],
            synthesized_idea=request.idea,
            changes_summary=[],
            residual_risks=[],
            next_experiments=[],
            confidence_score=0.5,
        )


def test_run_benchmark_continues_on_item_failure(tmp_path) -> None:  # noqa: ANN001
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"id":"ok-1","idea":"first idea"}',
                '{"id":"bad-1","idea":"this should fail"}',
            ]
        ),
        encoding="utf-8",
    )
    report = asyncio.run(
        run_benchmark(
            orchestrator=MixedOutcomeOrchestrator(),
            dataset_path=dataset,
            mode="deep",
        )
    )
    assert report["dataset_size"] == 2
    assert report["success_count"] == 1
    assert report["failure_count"] == 1
    assert report["failure_modes"] == {"RuntimeError": 1}
    assert report["average_quality_score"] == 0.22

    failed = next(item for item in report["runs"] if item["status"] == "failed")
    assert failed["id"] == "bad-1"
    assert failed["error_type"] == "RuntimeError"


def test_run_benchmark_includes_domain_metrics_and_rubric_domain(tmp_path) -> None:  # noqa: ANN001
    dataset = tmp_path / "dataset_domains.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"id":"fin-1","idea":"finance idea","domain":"finance"}',
                '{"id":"cons-1","idea":"consumer idea","domain":"consumer"}',
            ]
        ),
        encoding="utf-8",
    )
    report = asyncio.run(
        run_benchmark(
            orchestrator=MixedOutcomeOrchestrator(),
            dataset_path=dataset,
            mode="deep",
        )
    )
    assert "domain_metrics" in report
    assert report["domain_metrics"]["finance"]["dataset_size"] == 1
    assert report["domain_metrics"]["consumer"]["dataset_size"] == 1
    finance_run = next(item for item in report["runs"] if item["id"] == "fin-1")
    consumer_run = next(item for item in report["runs"] if item["id"] == "cons-1")
    assert finance_run["quality"]["rubric_domain"] == "finance"
    assert consumer_run["quality"]["rubric_domain"] == "consumer"


def test_compare_reports_ignores_failed_runs() -> None:
    base = {
        "created_at": "2026-02-14T00:00:00Z",
        "dataset_size": 2,
        "mode": "deep",
        "average_runtime_sec": 100.0,
        "average_quality_score": 0.5,
        "runs": [
            {"id": "a", "status": "completed", "runtime_sec": 40.0, "quality": {"quality_score": 0.5}},
            {"id": "b", "status": "failed", "runtime_sec": 60.0, "error": "x"},
        ],
    }
    target = {
        "created_at": "2026-02-15T00:00:00Z",
        "dataset_size": 2,
        "mode": "deep",
        "average_runtime_sec": 90.0,
        "average_quality_score": 0.55,
        "runs": [
            {"id": "a", "status": "completed", "runtime_sec": 35.0, "quality": {"quality_score": 0.55}},
            {"id": "b", "status": "completed", "runtime_sec": 55.0, "quality": {"quality_score": 0.6}},
        ],
    }
    diff = compare_reports(base, target)
    assert diff["shared_run_count"] == 1
    assert diff["run_deltas"][0]["id"] == "a"


def test_compare_reports_includes_domain_deltas() -> None:
    base = {
        "average_runtime_sec": 100.0,
        "average_quality_score": 0.5,
        "runs": [],
        "domain_metrics": {
            "finance": {"average_quality_score": 0.55, "average_runtime_sec": 90.0, "success_count": 3},
            "consumer": {"average_quality_score": 0.45, "average_runtime_sec": 110.0, "success_count": 3},
        },
    }
    target = {
        "average_runtime_sec": 95.0,
        "average_quality_score": 0.58,
        "runs": [],
        "domain_metrics": {
            "finance": {"average_quality_score": 0.62, "average_runtime_sec": 88.0, "success_count": 3},
            "consumer": {"average_quality_score": 0.5, "average_runtime_sec": 104.0, "success_count": 3},
        },
    }
    diff = compare_reports(base, target)
    assert len(diff["domain_deltas"]) == 2
    finance = next(item for item in diff["domain_deltas"] if item["domain"] == "finance")
    assert finance["quality_delta"] == 0.07
    assert finance["runtime_delta_sec"] == -2.0


def test_run_duel_benchmark_outputs_comparison_and_gate(tmp_path) -> None:  # noqa: ANN001
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id":"ok-1","idea":"first idea"}\n', encoding="utf-8")
    payload = asyncio.run(
        run_duel_benchmark(
            orchestrator=MixedOutcomeOrchestrator(),
            dataset_path=dataset,
            mode="deep",
            min_quality_delta=0.0,
            min_shared_runs=1,
        )
    )
    assert payload["baseline"]["mode"] == "single_pass_baseline"
    assert payload["target"]["mode"] == "deep"
    assert payload["comparison"]["shared_run_count"] == 1
    assert payload["gate"]["passed"] is True


def test_save_report_auto_paths_are_unique(tmp_path) -> None:  # noqa: ANN001
    report = {"created_at": "2026-02-14T00:00:00Z", "runs": []}
    original_reports_dir = runner.DEFAULT_REPORTS_DIR
    runner.DEFAULT_REPORTS_DIR = tmp_path
    try:
        path_a = save_report(report)
        path_b = save_report(report)
    finally:
        runner.DEFAULT_REPORTS_DIR = original_reports_dir

    assert path_a != path_b


def test_build_blind_review_packet_anonymizes_completed_runs() -> None:
    report = {
        "created_at": "2026-02-14T00:00:00Z",
        "dataset_path": "x.jsonl",
        "runs": [
            {
                "id": "x1",
                "idea": "Idea A",
                "domain": "finance",
                "run_id": "run-1",
                "mode": "deep",
                "status": "completed",
                "quality": {"quality_score": 0.5},
                "synthesized_idea": "Better Idea A",
                "changes_summary": ["c1"],
                "residual_risks": ["r1"],
                "next_experiments": ["e1"],
            },
            {
                "id": "x2",
                "idea": "Idea B",
                "status": "failed",
            },
        ],
    }
    packet = build_blind_review_packet(report)
    assert packet["item_count"] == 1
    item = packet["items"][0]
    assert item["blind_id"] == "candidate-001"
    assert item["idea"] == "Idea A"
    assert item["synthesized_idea"] == "Better Idea A"
    assert "run_id" not in item
    assert "quality" not in item
    assert "mode" not in item


def test_save_blind_review_packet_auto_paths_are_unique(tmp_path) -> None:  # noqa: ANN001
    packet = {"items": []}
    original_reports_dir = runner.DEFAULT_REPORTS_DIR
    runner.DEFAULT_REPORTS_DIR = tmp_path
    try:
        path_a = save_blind_review_packet(packet)
        path_b = save_blind_review_packet(packet)
    finally:
        runner.DEFAULT_REPORTS_DIR = original_reports_dir
    assert path_a != path_b
