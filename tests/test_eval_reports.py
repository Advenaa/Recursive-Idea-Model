from __future__ import annotations

import asyncio

from rim.core.schemas import AnalyzeResult, DecompositionNode
from rim.eval import runner
from rim.eval.runner import (
    build_blind_review_packet,
    calibrate_arbitration_policy,
    calibrate_depth_allocator,
    calibrate_specialist_arbitration_policy,
    calibration_env_exports,
    compare_reports,
    evaluate_regression_gate,
    run_online_depth_arbitration_learning_loop,
    run_benchmark,
    run_duel_benchmark,
    run_single_call_llm_baseline,
    run_single_pass_baseline,
    save_blind_review_packet,
    save_policy_artifact,
    save_report,
    train_arbitration_policy,
    train_online_depth_and_arbitration_policies,
    train_depth_policy,
    train_memory_policy,
    train_rl_depth_and_arbitration_policies,
    train_rl_spawn_policy,
    train_spawn_policy,
    train_specialist_arbitration_policy,
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


class FakeSingleCallSession:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.stage_policy: dict[str, list[str]] = {}

    async def invoke_json(self, stage, prompt, json_schema=None):  # noqa: ANN001, ANN201
        assert stage.startswith("baseline_llm_")
        assert "STRICT JSON only" in prompt
        return (
            {
                "synthesized_idea": "LLM baseline output",
                "changes_summary": ["change 1"],
                "residual_risks": ["risk 1"],
                "next_experiments": ["experiment 1"],
                "confidence_score": 0.61,
            },
            self.provider,
        )


class FakeSingleCallRouter:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.created_sessions = 0

    def create_session(self, run_id):  # noqa: ANN001, ANN201
        self.created_sessions += 1
        return FakeSingleCallSession(provider=self.provider)


class MixedOutcomeOrchestratorWithRouter(MixedOutcomeOrchestrator):
    def __init__(self, provider: str) -> None:
        super().__init__()
        self.router = FakeSingleCallRouter(provider=provider)


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


def test_run_single_call_llm_baseline_returns_scored_report(tmp_path) -> None:  # noqa: ANN001
    dataset = tmp_path / "dataset_single_call.jsonl"
    dataset.write_text('{"id":"x1","idea":"Idea A","domain":"finance"}\n', encoding="utf-8")
    router = FakeSingleCallRouter(provider="claude")
    report = asyncio.run(
        run_single_call_llm_baseline(
            dataset_path=dataset,
            provider="claude",
            mode="deep",
            limit=1,
            router=router,
        )
    )
    assert report["mode"] == "single_call_claude"
    assert report["provider"] == "claude"
    assert report["dataset_size"] == 1
    assert report["success_count"] == 1
    assert report["runs"][0]["run_id"] == "baseline-llm-claude-x1"
    assert report["runs"][0]["provider"] == "claude"
    assert router.created_sessions == 1


def test_run_single_call_llm_baseline_rejects_invalid_provider(tmp_path) -> None:  # noqa: ANN001
    dataset = tmp_path / "dataset_invalid_provider.jsonl"
    dataset.write_text('{"id":"x1","idea":"Idea A"}\n', encoding="utf-8")
    try:
        asyncio.run(
            run_single_call_llm_baseline(
                dataset_path=dataset,
                provider="openai",
                limit=1,
                router=FakeSingleCallRouter(provider="claude"),
            )
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "provider must be one of" in str(exc)


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


def test_run_duel_benchmark_supports_single_call_llm_baseline(tmp_path) -> None:  # noqa: ANN001
    dataset = tmp_path / "dataset_duel_single_call.jsonl"
    dataset.write_text('{"id":"ok-1","idea":"first idea"}\n', encoding="utf-8")
    orchestrator = MixedOutcomeOrchestratorWithRouter(provider="claude")
    payload = asyncio.run(
        run_duel_benchmark(
            orchestrator=orchestrator,  # type: ignore[arg-type]
            dataset_path=dataset,
            mode="deep",
            baseline_provider="claude",
            min_quality_delta=0.0,
            min_shared_runs=1,
        )
    )
    assert payload["baseline"]["mode"] == "single_call_claude"
    assert payload["baseline_provider"] == "claude"
    assert payload["comparison"]["shared_run_count"] == 1


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


def test_calibrate_depth_allocator_recommends_more_depth_for_low_quality() -> None:
    report = {
        "dataset_size": 10,
        "average_quality_score": 0.45,
        "average_runtime_sec": 30.0,
        "failure_count": 1,
    }
    calibration = calibrate_depth_allocator(
        report,
        target_quality=0.7,
        target_runtime_sec=60.0,
    )
    env = calibration["recommended_env"]
    assert env["RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE"] > 0.78
    assert env["RIM_MAX_ANALYSIS_CYCLES"] >= 2


def test_calibrate_depth_allocator_recommends_less_depth_for_runtime_pressure() -> None:
    report = {
        "dataset_size": 10,
        "average_quality_score": 0.7,
        "average_runtime_sec": 140.0,
        "failure_count": 3,
    }
    calibration = calibrate_depth_allocator(
        report,
        target_quality=0.65,
        target_runtime_sec=60.0,
    )
    env = calibration["recommended_env"]
    assert env["RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE"] < 0.78
    assert env["RIM_MAX_ANALYSIS_CYCLES"] == 1


def test_calibration_env_exports_formats_env_lines() -> None:
    calibration = {
        "recommended_env": {
            "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": 0.812,
            "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": 1,
            "RIM_MAX_ANALYSIS_CYCLES": 2,
            "RIM_SPAWN_ROLE_BOOSTS": {"security": 0.4, "finance": -0.2},
            "RIM_SPAWN_ROLE_TOOL_OVERRIDES": {"security": ["threat_model", "abuse_case_review"]},
        }
    }
    exports = calibration_env_exports(calibration)
    assert "export RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE=0.812" in exports
    assert "export RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS=1" in exports
    assert "export RIM_MAX_ANALYSIS_CYCLES=2" in exports
    assert "export RIM_SPAWN_ROLE_BOOSTS='{\"finance\":-0.2,\"security\":0.4}'" in exports
    assert (
        "export RIM_SPAWN_ROLE_TOOL_OVERRIDES="
        "'{\"security\":[\"threat_model\",\"abuse_case_review\"]}'"
    ) in exports


def test_train_depth_policy_aggregates_report_calibrations() -> None:
    reports = [
        {
            "created_at": "2026-02-14T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.52,
            "average_runtime_sec": 65.0,
            "failure_count": 1,
        },
        {
            "created_at": "2026-02-15T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.68,
            "average_runtime_sec": 48.0,
            "failure_count": 0,
        },
    ]
    payload = train_depth_policy(
        reports,
        target_quality=0.7,
        target_runtime_sec=60.0,
    )
    assert payload["report_count"] == 2
    env = payload["policy_env"]
    assert "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE" in env
    assert "RIM_MAX_ANALYSIS_CYCLES" in env
    assert payload["recommended_exports"]


def test_calibrate_specialist_arbitration_policy_recommends_more_specialist_capacity() -> None:
    report = {
        "dataset_size": 10,
        "average_quality_score": 0.5,
        "average_runtime_sec": 55.0,
        "failure_count": 1,
        "runs": [
            {
                "status": "completed",
                "telemetry": {
                    "disagreement_count": 3,
                    "diversity_flagged_count": 2,
                    "specialist_count": 1,
                },
            },
            {
                "status": "completed",
                "telemetry": {
                    "disagreement_count": 2,
                    "diversity_flagged_count": 1,
                    "specialist_count": 1,
                },
            },
        ],
    }
    calibration = calibrate_specialist_arbitration_policy(
        report,
        target_quality=0.7,
        target_runtime_sec=60.0,
    )
    env = calibration["recommended_env"]
    assert env["RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP"] == 1
    assert env["RIM_SPECIALIST_ARBITRATION_MAX_JOBS"] >= 2
    assert env["RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE"] > 0.78
    assert env["RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER"] == 1
    assert env["RIM_SPECIALIST_CONTRACT_LOOKBACK_RUNS"] >= 1
    assert env["RIM_SPECIALIST_CONTRACT_MIN_ROUNDS"] >= 1
    assert env["RIM_SPECIALIST_CONTRACT_MIN_ROLE_SAMPLES"] >= 1


def test_train_specialist_arbitration_policy_aggregates_reports() -> None:
    reports = [
        {
            "created_at": "2026-02-14T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.55,
            "average_runtime_sec": 62.0,
            "failure_count": 1,
            "runs": [
                {
                    "status": "completed",
                    "quality": {"quality_score": 0.63},
                    "telemetry": {
                        "disagreement_count": 2,
                        "diversity_flagged_count": 1,
                        "specialist_count": 1,
                    },
                }
            ],
        },
        {
            "created_at": "2026-02-15T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.7,
            "average_runtime_sec": 50.0,
            "failure_count": 0,
            "runs": [
                {
                    "status": "completed",
                    "quality": {"quality_score": 0.74},
                    "telemetry": {
                        "disagreement_count": 1,
                        "diversity_flagged_count": 0,
                        "specialist_count": 0,
                    },
                }
            ],
        },
    ]
    payload = train_specialist_arbitration_policy(
        reports,
        target_quality=0.7,
        target_runtime_sec=60.0,
    )
    assert payload["report_count"] == 2
    env = payload["policy_env"]
    assert "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP" in env
    assert "RIM_SPECIALIST_ARBITRATION_MAX_JOBS" in env
    assert "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE" in env
    assert "RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER" in env
    assert "RIM_SPECIALIST_CONTRACT_LOOKBACK_RUNS" in env
    assert "RIM_SPECIALIST_CONTRACT_MIN_ROUNDS" in env
    assert "RIM_SPECIALIST_CONTRACT_MIN_ROLE_SAMPLES" in env
    assert payload["recommended_exports"]


def test_calibrate_arbitration_policy_recommends_capacity() -> None:
    report = {
        "dataset_size": 10,
        "average_quality_score": 0.5,
        "average_runtime_sec": 56.0,
        "failure_count": 1,
        "runs": [
            {
                "status": "completed",
                "telemetry": {
                    "disagreement_count": 3,
                    "arbitration_resolved_count": 1,
                    "devils_advocate_count": 1,
                    "specialist_count": 1,
                },
            },
            {
                "status": "completed",
                "telemetry": {
                    "disagreement_count": 2,
                    "arbitration_resolved_count": 1,
                    "devils_advocate_count": 1,
                    "specialist_count": 1,
                },
            },
        ],
    }
    calibration = calibrate_arbitration_policy(
        report,
        target_quality=0.7,
        target_runtime_sec=60.0,
    )
    env = calibration["recommended_env"]
    assert env["RIM_ENABLE_DISAGREEMENT_ARBITRATION"] == 1
    assert env["RIM_ARBITRATION_MAX_JOBS"] >= 2
    assert env["RIM_DEVILS_ADVOCATE_ROUNDS"] >= 1


def test_train_arbitration_policy_aggregates_reports() -> None:
    reports = [
        {
            "created_at": "2026-02-14T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.55,
            "average_runtime_sec": 62.0,
            "failure_count": 1,
            "runs": [
                {
                    "status": "completed",
                    "quality": {"quality_score": 0.63},
                    "telemetry": {
                        "disagreement_count": 3,
                        "arbitration_resolved_count": 1,
                        "devils_advocate_count": 1,
                    },
                }
            ],
        },
        {
            "created_at": "2026-02-15T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.72,
            "average_runtime_sec": 49.0,
            "failure_count": 0,
            "runs": [
                {
                    "status": "completed",
                    "quality": {"quality_score": 0.74},
                    "telemetry": {
                        "disagreement_count": 1,
                        "arbitration_resolved_count": 1,
                        "devils_advocate_count": 0,
                    },
                }
            ],
        },
    ]
    payload = train_arbitration_policy(
        reports,
        target_quality=0.7,
        target_runtime_sec=60.0,
    )
    assert payload["report_count"] == 2
    env = payload["policy_env"]
    assert "RIM_ENABLE_DISAGREEMENT_ARBITRATION" in env
    assert "RIM_ARBITRATION_MAX_JOBS" in env
    assert "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION" in env
    assert "RIM_DEVILS_ADVOCATE_ROUNDS" in env
    assert "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE" in env
    assert payload["recommended_exports"]


def test_train_spawn_policy_aggregates_reports() -> None:
    reports = [
        {
            "created_at": "2026-02-14T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.54,
            "average_runtime_sec": 61.0,
            "failure_count": 1,
            "runs": [
                {
                    "status": "completed",
                    "quality": {"quality_score": 0.63},
                    "telemetry": {
                        "disagreement_count": 3,
                        "spawn_selected_count": 3,
                        "spawn_dynamic_count": 2,
                        "spawn_selected_roles": ["security", "dynamic_aodkinv"],
                        "spawn_role_routing": {
                            "dynamic_aodkinv": "prioritize_domain_specific_signals",
                        },
                        "spawn_role_tools": {
                            "dynamic_aodkinv": ["context_probe:aodkinv", "evidence_scan"],
                        },
                        "specialist_selected_roles": ["security", "security"],
                        "specialist_role_action_counts": {
                            "security": {"merge": 2, "escalate": 0, "drop": 0, "total": 2}
                        },
                        "specialist_role_avg_match_score": {"security": 1.8},
                        "specialist_top_action": "merge",
                    },
                }
            ],
        },
        {
            "created_at": "2026-02-15T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.71,
            "average_runtime_sec": 48.0,
            "failure_count": 0,
            "runs": [
                {
                    "status": "completed",
                    "quality": {"quality_score": 0.74},
                    "telemetry": {
                        "disagreement_count": 1,
                        "spawn_selected_count": 2,
                        "spawn_dynamic_count": 1,
                        "spawn_selected_roles": ["security", "dynamic_aodkinv"],
                        "spawn_role_routing": {
                            "dynamic_aodkinv": "prioritize_domain_specific_signals",
                        },
                        "spawn_role_tools": {
                            "dynamic_aodkinv": ["context_probe:aodkinv", "counterexample_search"],
                        },
                        "specialist_selected_roles": ["security", "security"],
                        "specialist_role_action_counts": {
                            "security": {"merge": 2, "escalate": 0, "drop": 0, "total": 2}
                        },
                        "specialist_role_avg_match_score": {"security": 1.4},
                        "specialist_top_action": "merge",
                    },
                }
            ],
        },
    ]
    payload = train_spawn_policy(
        reports,
        target_quality=0.7,
        target_runtime_sec=60.0,
    )
    assert payload["report_count"] == 2
    env = payload["policy_env"]
    assert "RIM_SPAWN_MIN_ROLE_SCORE" in env
    assert "RIM_SPAWN_MAX_SPECIALISTS_DEEP" in env
    assert "RIM_ENABLE_DYNAMIC_SPECIALISTS" in env
    assert "RIM_SPAWN_ROLE_BOOSTS" in env
    assert "security" in env["RIM_SPAWN_ROLE_BOOSTS"]
    assert "RIM_SPAWN_DYNAMIC_ROLE_CONTRACTS" in env
    assert "aodkinv" in env["RIM_SPAWN_DYNAMIC_ROLE_CONTRACTS"]
    assert payload["recommended_exports"]


def test_train_memory_policy_aggregates_reports() -> None:
    reports = [
        {
            "created_at": "2026-02-14T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.56,
            "average_runtime_sec": 63.0,
            "failure_count": 1,
            "runs": [
                {
                    "status": "completed",
                    "telemetry": {
                        "memory_fold_count": 2,
                        "memory_fold_degradation_count": 1,
                        "memory_fold_avg_novelty_ratio": 0.3,
                        "memory_fold_avg_duplicate_ratio": 0.55,
                    },
                }
            ],
        },
        {
            "created_at": "2026-02-15T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.72,
            "average_runtime_sec": 49.0,
            "failure_count": 0,
            "runs": [
                {
                    "status": "completed",
                    "telemetry": {
                        "memory_fold_count": 2,
                        "memory_fold_degradation_count": 0,
                        "memory_fold_avg_novelty_ratio": 0.5,
                        "memory_fold_avg_duplicate_ratio": 0.3,
                    },
                }
            ],
        },
    ]
    payload = train_memory_policy(
        reports,
        target_quality=0.7,
        target_runtime_sec=60.0,
    )
    assert payload["report_count"] == 2
    env = payload["policy_env"]
    assert "RIM_ENABLE_MEMORY_FOLDING" in env
    assert "RIM_MEMORY_FOLD_MAX_ENTRIES" in env
    assert "RIM_MEMORY_FOLD_NOVELTY_FLOOR" in env
    assert "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO" in env
    assert payload["recommended_exports"]


def test_train_online_depth_and_arbitration_policies_blends_prior() -> None:
    reports = [
        {
            "created_at": "2026-02-14T00:00:00Z",
            "dataset_size": 8,
            "average_quality_score": 0.52,
            "average_runtime_sec": 62.0,
            "failure_count": 1,
            "runs": [
                {
                    "status": "completed",
                    "telemetry": {
                        "disagreement_count": 2,
                        "diversity_flagged_count": 1,
                        "specialist_count": 1,
                    },
                }
            ],
        }
    ]
    payload = train_online_depth_and_arbitration_policies(
        reports,
        target_quality=0.7,
        target_runtime_sec=60.0,
        learning_rate=0.5,
        prior_depth_policy_env={
            "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": 0.7,
            "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": 3,
            "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": 2,
            "RIM_MAX_ANALYSIS_CYCLES": 1,
        },
        prior_specialist_policy_env={
            "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP": 1,
            "RIM_SPECIALIST_ARBITRATION_MAX_JOBS": 1,
            "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE": 0.7,
            "RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER": 1,
            "RIM_SPECIALIST_CONTRACT_LOOKBACK_RUNS": 16,
            "RIM_SPECIALIST_CONTRACT_MIN_ROUNDS": 3,
            "RIM_SPECIALIST_CONTRACT_MIN_ROLE_SAMPLES": 2,
        },
        prior_arbitration_policy_env={
            "RIM_ENABLE_DISAGREEMENT_ARBITRATION": 1,
            "RIM_ARBITRATION_MAX_JOBS": 1,
            "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": 1,
            "RIM_DEVILS_ADVOCATE_ROUNDS": 1,
            "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": 0.8,
        },
    )
    assert payload["report_count"] == 1
    depth_env = payload["depth_policy"]["policy_env"]
    specialist_env = payload["specialist_policy"]["policy_env"]
    arbitration_env = payload["arbitration_policy"]["policy_env"]
    assert "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE" in depth_env
    assert "RIM_SPECIALIST_ARBITRATION_MAX_JOBS" in specialist_env
    assert "RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER" in specialist_env
    assert "RIM_SPECIALIST_CONTRACT_LOOKBACK_RUNS" in specialist_env
    assert "RIM_ARBITRATION_MAX_JOBS" in arbitration_env
    assert payload["recommended_exports"]


def test_train_rl_depth_and_arbitration_policies_outputs_credit_assignment() -> None:
    reports = [
        {
            "created_at": "2026-02-14T00:00:00Z",
            "dataset_size": 2,
            "average_quality_score": 0.6,
            "average_runtime_sec": 62.0,
            "failure_count": 0,
            "runs": [
                {
                    "id": "run-a",
                    "run_id": "run-a",
                    "status": "completed",
                    "runtime_sec": 58.0,
                    "quality": {"quality_score": 0.66},
                    "telemetry": {
                        "disagreement_count": 2,
                        "diversity_flagged_count": 1,
                        "depth_max_cycles_config": 2,
                        "depth_min_confidence_config": 0.81,
                        "depth_max_residual_risks_config": 1,
                        "depth_max_high_findings_config": 1,
                        "specialist_max_jobs_config": 3,
                        "specialist_min_confidence_config": 0.82,
                        "specialist_loop_enabled_config": True,
                    },
                },
                {
                    "id": "run-b",
                    "run_id": "run-b",
                    "status": "completed",
                    "runtime_sec": 74.0,
                    "quality": {"quality_score": 0.52},
                    "telemetry": {
                        "disagreement_count": 1,
                        "diversity_flagged_count": 0,
                        "depth_max_cycles_config": 1,
                        "depth_min_confidence_config": 0.78,
                        "depth_max_residual_risks_config": 2,
                        "depth_max_high_findings_config": 1,
                        "specialist_max_jobs_config": 1,
                        "specialist_min_confidence_config": 0.78,
                        "specialist_loop_enabled_config": True,
                    },
                },
            ],
        }
    ]
    payload = train_rl_depth_and_arbitration_policies(
        reports,
        target_quality=0.65,
        target_runtime_sec=60.0,
        learning_rate=0.2,
        epochs=2,
    )
    assert payload["optimizer"] == "rl_credit_assignment_v1"
    assert payload["experience_count"] == 2
    assert "depth_policy" in payload
    assert "specialist_policy" in payload
    assert "arbitration_policy" in payload
    assert "credit_assignment" in payload
    assert payload["credit_assignment"]["depth"]["top_runs"]
    assert payload["credit_assignment"]["specialist"]["top_runs"]
    assert payload["credit_assignment"]["arbitration"]["top_runs"]


def test_train_rl_spawn_policy_outputs_credit_assignment() -> None:
    reports = [
        {
            "created_at": "2026-02-14T00:00:00Z",
            "dataset_size": 2,
            "average_quality_score": 0.6,
            "average_runtime_sec": 64.0,
            "failure_count": 0,
            "mode": "deep",
            "runs": [
                {
                    "id": "run-a",
                    "run_id": "run-a",
                    "status": "completed",
                    "mode": "deep",
                    "runtime_sec": 52.0,
                    "quality": {"quality_score": 0.69},
                    "telemetry": {
                        "disagreement_count": 2,
                        "spawn_selected_count": 3,
                        "spawn_dynamic_count": 1,
                        "spawn_selected_roles": ["security", "dynamic_bioinformatics"],
                        "spawn_dynamic_roles": ["dynamic_bioinformatics"],
                        "spawn_role_routing": {
                            "security": "prioritize_high_severity_and_compliance_constraints",
                            "dynamic_bioinformatics": "prioritize_domain_specific_signals",
                        },
                        "spawn_role_tools": {
                            "security": ["threat_model", "policy_checklist"],
                            "dynamic_bioinformatics": [
                                "context_probe:bioinformatics",
                                "evidence_scan",
                            ],
                        },
                        "spawn_min_role_score_config": 0.8,
                        "spawn_max_specialists_config": 4,
                        "spawn_max_dynamic_specialists_config": 2,
                        "spawn_dynamic_enabled_config": True,
                        "specialist_selected_roles": ["security"],
                        "specialist_role_action_counts": {
                            "security": {"merge": 2, "escalate": 0, "drop": 0, "total": 2}
                        },
                        "specialist_role_avg_match_score": {"security": 1.7},
                    },
                },
                {
                    "id": "run-b",
                    "run_id": "run-b",
                    "status": "completed",
                    "mode": "deep",
                    "runtime_sec": 74.0,
                    "quality": {"quality_score": 0.52},
                    "telemetry": {
                        "disagreement_count": 0,
                        "spawn_selected_count": 1,
                        "spawn_dynamic_count": 0,
                        "spawn_selected_roles": ["finance"],
                        "spawn_dynamic_roles": [],
                        "spawn_role_routing": {
                            "finance": "prioritize_margin_and_budget_constraints",
                        },
                        "spawn_role_tools": {
                            "finance": ["unit_economics_model"],
                        },
                        "spawn_min_role_score_config": 1.3,
                        "spawn_max_specialists_config": 2,
                        "spawn_max_dynamic_specialists_config": 1,
                        "spawn_dynamic_enabled_config": True,
                        "specialist_selected_roles": ["finance"],
                        "specialist_role_action_counts": {
                            "finance": {"merge": 1, "escalate": 0, "drop": 0, "total": 1}
                        },
                        "specialist_role_avg_match_score": {"finance": 1.1},
                    },
                },
            ],
        }
    ]
    payload = train_rl_spawn_policy(
        reports,
        target_quality=0.65,
        target_runtime_sec=60.0,
        learning_rate=0.2,
        epochs=2,
    )
    assert payload["optimizer"] == "rl_spawn_credit_assignment_v1"
    assert payload["experience_count"] == 2
    assert "spawn_policy" in payload
    assert payload["credit_assignment"]["spawn"]["top_runs"]
    env = payload["spawn_policy"]["policy_env"]
    assert "RIM_SPAWN_MIN_ROLE_SCORE" in env
    assert "RIM_SPAWN_MAX_SPECIALISTS_DEEP" in env
    assert "RIM_SPAWN_ROLE_ROUTING_OVERRIDES" in env
    assert "RIM_SPAWN_ROLE_BOOSTS" in env
    assert "RIM_SPAWN_DYNAMIC_ROLE_CONTRACTS" in env
    assert "bioinformatics" in env["RIM_SPAWN_DYNAMIC_ROLE_CONTRACTS"]
    assert "security" in env["RIM_SPAWN_ROLE_BOOSTS"]
    assert payload["recommended_exports"]


def test_save_policy_artifact_writes_expected_shape(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "depth_policy.json"
    policy = {
        "policy_env": {
            "RIM_MAX_ANALYSIS_CYCLES": 2,
            "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": 0.82,
            "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": 1,
            "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": 1,
        }
    }
    saved_path = save_policy_artifact(
        policy,
        policy_kind="depth",
        source_reports=["report-a.json"],
        output_path=path,
        learning_meta={"learning_rate": 0.35},
    )
    assert saved_path == path
    payload = runner.load_report(path)
    assert payload["policy_kind"] == "depth"
    assert payload["policy_env"]["RIM_MAX_ANALYSIS_CYCLES"] == 2
    assert payload["recommended_exports"]


def test_run_online_depth_arbitration_learning_loop_writes_policy_files(tmp_path) -> None:  # noqa: ANN001
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id":"ok-1","idea":"first idea"}\n', encoding="utf-8")
    reports_dir = tmp_path / "reports"
    depth_policy_path = tmp_path / "policies" / "depth_policy.json"
    specialist_policy_path = tmp_path / "policies" / "specialist_policy.json"
    arbitration_policy_path = tmp_path / "policies" / "arbitration_policy.json"
    spawn_policy_path = tmp_path / "policies" / "spawn_policy.json"
    memory_policy_path = tmp_path / "policies" / "memory_policy.json"

    payload = asyncio.run(
        run_online_depth_arbitration_learning_loop(
            orchestrator=MixedOutcomeOrchestrator(),
            dataset_path=dataset,
            mode="deep",
            limit=1,
            iterations=1,
            lookback_reports=3,
            target_quality=0.65,
            target_runtime_sec=60.0,
            learning_rate=0.4,
            optimizer="rl",
            rl_epochs=2,
            reports_dir=reports_dir,
            depth_policy_path=depth_policy_path,
            specialist_policy_path=specialist_policy_path,
            arbitration_policy_path=arbitration_policy_path,
            spawn_policy_path=spawn_policy_path,
            memory_policy_path=memory_policy_path,
        )
    )
    assert payload["iterations"] == 1
    assert payload["optimizer"] == "rl"
    assert depth_policy_path.exists() is True
    assert specialist_policy_path.exists() is True
    assert arbitration_policy_path.exists() is True
    assert spawn_policy_path.exists() is True
    assert memory_policy_path.exists() is True
    assert payload["arbitration_policy_path"] == str(arbitration_policy_path)
    assert payload["spawn_policy_path"] == str(spawn_policy_path)
    assert payload["memory_policy_path"] == str(memory_policy_path)
    assert payload["final"]["memory_policy_path"] == str(memory_policy_path)
    assert payload["final"]["training_report_count"] >= 1
