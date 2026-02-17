from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rim.api.job_queue import RunJobQueue
from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest
from rim.engine import build_orchestrator as build_embedded_orchestrator
from rim.eval.runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REPORTS_DIR,
    build_blind_review_packet,
    calibrate_depth_allocator,
    calibration_env_exports,
    compare_reports,
    evaluate_regression_gate,
    list_reports,
    load_report,
    run_online_depth_arbitration_learning_loop,
    run_benchmark,
    run_duel_benchmark,
    run_single_call_llm_baseline,
    run_single_pass_baseline,
    save_blind_review_packet,
    save_report,
    train_arbitration_policy,
    train_depth_policy,
    train_memory_policy,
    train_rl_depth_and_arbitration_policies,
    train_rl_memory_policy,
    train_rl_orchestration_policies,
    train_rl_spawn_policy,
    train_spawn_policy,
    train_specialist_arbitration_policy,
)
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def _build_orchestrator(
    *,
    repository: RunRepository | None = None,
    router: ProviderRouter | None = None,
) -> RimOrchestrator:
    return build_embedded_orchestrator(
        repository=repository or RunRepository(),
        router=router or ProviderRouter(),
    )


async def _cmd_analyze(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    request = AnalyzeRequest(
        idea=args.idea,
        mode=args.mode,
        domain=args.domain,
        constraints=args.constraint or [],
        desired_outcome=args.desired_outcome,
    )
    run_payload = None
    result = None

    if args.run_id:
        run_payload = orchestrator.get_run(args.run_id)
        if run_payload is not None:
            existing_request = orchestrator.get_run_request(args.run_id)
            if (
                existing_request is not None
                and existing_request.model_dump() != request.model_dump()
            ):
                print("run_id already exists with a different request payload.")
                return 1
            if run_payload.result is not None:
                result = run_payload.result
        else:
            run_id = orchestrator.create_run(
                request=request,
                status="running",
                run_id=args.run_id,
            )
            result = await orchestrator.execute_run(run_id, request)
            run_payload = orchestrator.get_run(run_id)
    else:
        result = await orchestrator.analyze(request)
        run_payload = orchestrator.get_run(result.run_id)

    if run_payload is not None and run_payload.result is None:
        payload = run_payload.model_dump()
    elif result is not None:
        payload = result.model_dump()
    else:
        payload = {"status": "unknown", "error": "No run output available."}

    if args.save:
        Path(args.save).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if result is not None:
            status = run_payload.status if run_payload is not None else "completed"
            print(f"run_id: {result.run_id}")
            print(f"status: {status}")
            print(f"mode: {result.mode}")
            print(f"confidence_score: {result.confidence_score:.2f}")
            print(f"synthesized_idea: {result.synthesized_idea}")
            if run_payload is not None and run_payload.error_summary:
                print(f"error_summary: {run_payload.error_summary}")
        elif run_payload is not None:
            print(f"run_id: {run_payload.run_id}")
            print(f"status: {run_payload.status}")
            if run_payload.error_summary:
                print(f"error_summary: {run_payload.error_summary}")
    if run_payload is not None and run_payload.status == "failed":
        return 2
    return 0


def _cmd_run_show(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    run = orchestrator.get_run(args.run_id)
    if run is None:
        print("Run not found")
        return 1
    print(json.dumps(run.model_dump(), indent=2))
    return 0


def _cmd_run_list(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    payload = orchestrator.list_runs(
        status=args.status,
        mode=args.mode,
        limit=args.limit,
        offset=args.offset,
    )
    print(json.dumps(payload.model_dump(), indent=2))
    return 0


def _cmd_run_logs(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    run = orchestrator.get_run(args.run_id)
    if run is None:
        print("Run not found")
        return 1
    logs = orchestrator.get_run_logs(args.run_id)
    print(json.dumps(logs.model_dump(), indent=2))
    return 0


def _cmd_run_feedback(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    run = orchestrator.get_run(args.run_id)
    if run is None:
        print("Run not found")
        return 1
    payload = orchestrator.submit_feedback(
        run_id=args.run_id,
        verdict=args.verdict,
        notes=args.notes,
    )
    print(json.dumps(payload, indent=2))
    return 0


async def _cmd_run_cancel(args: argparse.Namespace) -> int:
    repository = RunRepository()
    orchestrator = _build_orchestrator(repository=repository, router=ProviderRouter())
    queue = RunJobQueue(orchestrator=orchestrator, repository=repository)
    run = await queue.cancel(args.run_id)
    if run is None:
        print("Run not found")
        return 1
    print(json.dumps(run.model_dump(), indent=2))
    return 0


async def _cmd_run_retry(args: argparse.Namespace) -> int:
    repository = RunRepository()
    orchestrator = _build_orchestrator(repository=repository, router=ProviderRouter())
    queue = RunJobQueue(orchestrator=orchestrator, repository=repository)
    try:
        run = await queue.retry(args.run_id)
    except ValueError as exc:
        print(str(exc))
        return 1
    if run is None:
        print("Run not found")
        return 1
    print(json.dumps(run.model_dump(), indent=2))
    return 0


async def _cmd_health() -> int:
    repository = RunRepository()
    router = ProviderRouter()
    status = {
        "db": repository.healthcheck(),
        "providers": await router.healthcheck(),
    }
    status["ok"] = status["db"] and all(status["providers"].values())
    print(json.dumps(status, indent=2))
    return 0


async def _cmd_eval_run(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    dataset = Path(args.dataset) if args.dataset else DEFAULT_DATASET_PATH
    report = await run_benchmark(
        orchestrator=orchestrator,
        dataset_path=dataset,
        mode=args.mode,
        limit=args.limit,
    )
    save_path = save_report(report, Path(args.save) if args.save else None)
    report["report_path"] = str(save_path)
    print(json.dumps(report, indent=2))
    return 0


async def _cmd_eval_duel(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    dataset = Path(args.dataset) if args.dataset else DEFAULT_DATASET_PATH
    result = await run_duel_benchmark(
        orchestrator=orchestrator,
        dataset_path=dataset,
        mode=args.mode,
        limit=args.limit,
        min_quality_delta=args.min_quality_delta,
        max_runtime_delta_sec=args.max_runtime_delta_sec,
        min_shared_runs=args.min_shared_runs,
        baseline_provider=args.baseline_provider,
    )
    baseline_path = save_report(
        result["baseline"],
        Path(args.save_baseline) if args.save_baseline else None,
    )
    target_path = save_report(
        result["target"],
        Path(args.save_target) if args.save_target else None,
    )
    payload = {
        "baseline_report": str(baseline_path),
        "target_report": str(target_path),
        "baseline_provider": args.baseline_provider,
        "comparison": result["comparison"],
        "gate": result["gate"],
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if result["gate"]["passed"] else 2


def _cmd_eval_list(_args: argparse.Namespace) -> int:
    reports = [str(path) for path in list_reports(DEFAULT_REPORTS_DIR)]
    print(json.dumps({"reports": reports}, indent=2))
    return 0


def _cmd_eval_baseline(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset) if args.dataset else DEFAULT_DATASET_PATH
    report = run_single_pass_baseline(
        dataset_path=dataset,
        limit=args.limit,
    )
    save_path = save_report(report, Path(args.save) if args.save else None)
    report["report_path"] = str(save_path)
    print(json.dumps(report, indent=2))
    return 0


async def _cmd_eval_baseline_llm(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset) if args.dataset else DEFAULT_DATASET_PATH
    report = await run_single_call_llm_baseline(
        dataset_path=dataset,
        provider=args.provider,
        mode=args.mode,
        limit=args.limit,
    )
    save_path = save_report(report, Path(args.save) if args.save else None)
    report["report_path"] = str(save_path)
    print(json.dumps(report, indent=2))
    return 0


def _resolve_compare_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.base and args.target:
        return Path(args.base), Path(args.target)
    reports = list_reports(DEFAULT_REPORTS_DIR)
    if len(reports) < 2:
        raise ValueError("Need at least two saved reports for automatic comparison.")
    return reports[-2], reports[-1]


def _cmd_eval_compare(args: argparse.Namespace) -> int:
    base_path, target_path = _resolve_compare_paths(args)
    base = load_report(base_path)
    target = load_report(target_path)
    comparison = compare_reports(base, target)
    comparison["base_report"] = str(base_path)
    comparison["target_report"] = str(target_path)
    print(json.dumps(comparison, indent=2))
    return 0


def _cmd_eval_gate(args: argparse.Namespace) -> int:
    base_path, target_path = _resolve_compare_paths(args)
    base = load_report(base_path)
    target = load_report(target_path)
    comparison = compare_reports(base, target)
    gate = evaluate_regression_gate(
        comparison=comparison,
        min_quality_delta=args.min_quality_delta,
        max_runtime_delta_sec=args.max_runtime_delta_sec,
        min_shared_runs=args.min_shared_runs,
    )
    payload = {
        "base_report": str(base_path),
        "target_report": str(target_path),
        "comparison": comparison,
        "gate": gate,
    }
    print(json.dumps(payload, indent=2))
    return 0 if gate["passed"] else 2


def _resolve_blindpack_report_path(args: argparse.Namespace) -> Path:
    if args.report:
        return Path(args.report)
    reports = list_reports(DEFAULT_REPORTS_DIR)
    if not reports:
        raise ValueError("No saved reports available for blind review packet generation.")
    return reports[-1]


def _cmd_eval_blindpack(args: argparse.Namespace) -> int:
    report_path = _resolve_blindpack_report_path(args)
    report = load_report(report_path)
    packet = build_blind_review_packet(
        report=report,
        max_items=args.limit,
    )
    save_path = save_blind_review_packet(
        packet,
        Path(args.save) if args.save else None,
    )
    payload = {
        "source_report": str(report_path),
        "blind_review_path": str(save_path),
        "item_count": packet["item_count"],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _resolve_calibrate_report_path(args: argparse.Namespace) -> Path:
    if args.report:
        return Path(args.report)
    reports = list_reports(DEFAULT_REPORTS_DIR)
    if not reports:
        raise ValueError("No saved reports available for calibration.")
    return reports[-1]


def _cmd_eval_calibrate(args: argparse.Namespace) -> int:
    report_path = _resolve_calibrate_report_path(args)
    report = load_report(report_path)
    calibration = calibrate_depth_allocator(
        report,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
    )
    payload = {
        "source_report": str(report_path),
        "calibration": calibration,
        "recommended_exports": calibration_env_exports(calibration),
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _resolve_train_policy_reports(args: argparse.Namespace) -> list[Path]:
    if args.reports:
        values = [item.strip() for item in str(args.reports).split(",") if item.strip()]
        return [Path(item) for item in values]
    reports_dir = Path(args.reports_dir) if args.reports_dir else DEFAULT_REPORTS_DIR
    reports = list_reports(reports_dir)
    if not reports:
        raise ValueError("No saved reports available for policy training.")
    return reports


def _cmd_eval_train_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_depth_policy(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_eval_train_specialist_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_specialist_arbitration_policy(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_eval_train_arbitration_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_arbitration_policy(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_eval_train_spawn_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_spawn_policy(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_eval_train_memory_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_memory_policy(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_eval_train_rl_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_rl_depth_and_arbitration_policies(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        reward_runtime_weight=args.reward_runtime_weight,
        reward_failure_penalty=args.reward_failure_penalty,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_eval_train_rl_spawn_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_rl_spawn_policy(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        reward_runtime_weight=args.reward_runtime_weight,
        reward_failure_penalty=args.reward_failure_penalty,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_eval_train_rl_memory_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_rl_memory_policy(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        reward_runtime_weight=args.reward_runtime_weight,
        reward_failure_penalty=args.reward_failure_penalty,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_eval_train_rl_orchestration_policy(args: argparse.Namespace) -> int:
    report_paths = _resolve_train_policy_reports(args)
    reports = [load_report(path) for path in report_paths]
    policy = train_rl_orchestration_policies(
        reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        reward_runtime_weight=args.reward_runtime_weight,
        reward_failure_penalty=args.reward_failure_penalty,
    )
    payload = {
        "report_count": len(report_paths),
        "report_paths": [str(path) for path in report_paths],
        "policy": policy,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


async def _cmd_eval_calibrate_loop(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    dataset = Path(args.dataset) if args.dataset else DEFAULT_DATASET_PATH
    report = await run_benchmark(
        orchestrator=orchestrator,
        dataset_path=dataset,
        mode=args.mode,
        limit=args.limit,
    )
    report_path = save_report(report, Path(args.save_report) if args.save_report else None)
    calibration = calibrate_depth_allocator(
        report,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
    )
    payload = {
        "report_path": str(report_path),
        "calibration": calibration,
        "recommended_exports": calibration_env_exports(calibration),
    }
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


async def _cmd_eval_autolearn(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    dataset = Path(args.dataset) if args.dataset else DEFAULT_DATASET_PATH
    reports_dir = Path(args.reports_dir) if args.reports_dir else DEFAULT_REPORTS_DIR
    payload = await run_online_depth_arbitration_learning_loop(
        orchestrator=orchestrator,
        dataset_path=dataset,
        mode=args.mode,
        limit=args.limit,
        iterations=args.iterations,
        lookback_reports=args.lookback_reports,
        target_quality=args.target_quality,
        target_runtime_sec=args.target_runtime_sec,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        rl_epochs=args.rl_epochs,
        rl_reward_runtime_weight=args.rl_reward_runtime_weight,
        rl_reward_failure_penalty=args.rl_reward_failure_penalty,
        reports_dir=reports_dir,
        depth_policy_path=Path(args.depth_policy_path),
        specialist_policy_path=Path(args.specialist_policy_path),
        arbitration_policy_path=Path(args.arbitration_policy_path),
        spawn_policy_path=Path(args.spawn_policy_path),
        memory_policy_path=Path(args.memory_policy_path),
    )
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rim")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run idea analysis")
    analyze.add_argument("--idea", required=True)
    analyze.add_argument("--mode", choices=["deep", "fast"], default="deep")
    analyze.add_argument("--domain")
    analyze.add_argument("--constraint", action="append")
    analyze.add_argument("--desired-outcome")
    analyze.add_argument("--run-id")
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--save")

    run = sub.add_parser("run", help="Inspect runs")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    run_list = run_sub.add_parser("list", help="List runs")
    run_list.add_argument(
        "--status",
        choices=["queued", "running", "completed", "failed", "partial", "canceled"],
    )
    run_list.add_argument("--mode", choices=["deep", "fast"])
    run_list.add_argument("--limit", type=int, default=20)
    run_list.add_argument("--offset", type=int, default=0)
    run_show = run_sub.add_parser("show", help="Show run details")
    run_show.add_argument("run_id")
    run_logs = run_sub.add_parser("logs", help="Show stage telemetry logs")
    run_logs.add_argument("run_id")
    run_feedback = run_sub.add_parser("feedback", help="Submit run feedback")
    run_feedback.add_argument("run_id")
    run_feedback.add_argument("--verdict", choices=["accept", "reject"], required=True)
    run_feedback.add_argument("--notes")
    run_cancel = run_sub.add_parser("cancel", help="Cancel a queued/running run")
    run_cancel.add_argument("run_id")
    run_retry = run_sub.add_parser("retry", help="Retry a failed/partial/canceled run")
    run_retry.add_argument("run_id")

    eval_parser = sub.add_parser("eval", help="Benchmark and scoring")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run", help="Run benchmark dataset")
    eval_run.add_argument("--dataset")
    eval_run.add_argument("--mode", choices=["deep", "fast"], default="deep")
    eval_run.add_argument("--limit", type=int)
    eval_run.add_argument("--save")
    eval_duel = eval_sub.add_parser(
        "duel",
        help="Run baseline + benchmark comparison + regression gate",
    )
    eval_duel.add_argument("--dataset")
    eval_duel.add_argument("--mode", choices=["deep", "fast"], default="deep")
    eval_duel.add_argument("--limit", type=int)
    eval_duel.add_argument("--save")
    eval_duel.add_argument("--save-baseline")
    eval_duel.add_argument("--save-target")
    eval_duel.add_argument(
        "--baseline-provider",
        choices=["proxy", "claude", "codex"],
        default="proxy",
        help="baseline type: proxy=deterministic placeholder, claude/codex=single-call LLM baseline",
    )
    eval_duel.add_argument("--min-quality-delta", type=float, default=0.0)
    eval_duel.add_argument("--max-runtime-delta-sec", type=float)
    eval_duel.add_argument("--min-shared-runs", type=int, default=1)
    eval_sub.add_parser("list", help="List saved benchmark reports")
    eval_baseline = eval_sub.add_parser(
        "baseline",
        help="Run deterministic single-pass baseline benchmark",
    )
    eval_baseline.add_argument("--dataset")
    eval_baseline.add_argument("--limit", type=int)
    eval_baseline.add_argument("--save")
    eval_baseline_llm = eval_sub.add_parser(
        "baseline-llm",
        help="Run single-call Claude/Codex baseline benchmark",
    )
    eval_baseline_llm.add_argument("--provider", choices=["claude", "codex"], required=True)
    eval_baseline_llm.add_argument("--dataset")
    eval_baseline_llm.add_argument("--mode", choices=["deep", "fast"], default="deep")
    eval_baseline_llm.add_argument("--limit", type=int)
    eval_baseline_llm.add_argument("--save")
    eval_compare = eval_sub.add_parser(
        "compare",
        help="Compare two benchmark reports (defaults to latest two)",
    )
    eval_compare.add_argument("--base")
    eval_compare.add_argument("--target")
    eval_gate = eval_sub.add_parser(
        "gate",
        help="Apply regression thresholds to report comparison",
    )
    eval_gate.add_argument("--base")
    eval_gate.add_argument("--target")
    eval_gate.add_argument("--min-quality-delta", type=float, default=0.0)
    eval_gate.add_argument("--max-runtime-delta-sec", type=float)
    eval_gate.add_argument("--min-shared-runs", type=int, default=1)
    eval_blindpack = eval_sub.add_parser(
        "blindpack",
        help="Generate anonymized blind-review packet from a benchmark report",
    )
    eval_blindpack.add_argument("--report")
    eval_blindpack.add_argument("--limit", type=int)
    eval_blindpack.add_argument("--save")
    eval_calibrate = eval_sub.add_parser(
        "calibrate",
        help="Recommend depth-allocator settings from benchmark report",
    )
    eval_calibrate.add_argument("--report")
    eval_calibrate.add_argument("--target-quality", type=float, default=0.65)
    eval_calibrate.add_argument("--target-runtime-sec", type=float)
    eval_calibrate.add_argument("--save")
    eval_calibrate_loop = eval_sub.add_parser(
        "calibrate-loop",
        help="Run benchmark then output depth-allocator calibration recommendations",
    )
    eval_calibrate_loop.add_argument("--dataset")
    eval_calibrate_loop.add_argument("--mode", choices=["deep", "fast"], default="deep")
    eval_calibrate_loop.add_argument("--limit", type=int)
    eval_calibrate_loop.add_argument("--target-quality", type=float, default=0.65)
    eval_calibrate_loop.add_argument("--target-runtime-sec", type=float)
    eval_calibrate_loop.add_argument("--save")
    eval_calibrate_loop.add_argument("--save-report")
    eval_train_policy = eval_sub.add_parser(
        "train-policy",
        help="Train an aggregated depth policy from multiple benchmark reports",
    )
    eval_train_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_policy.add_argument("--reports-dir")
    eval_train_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_policy.add_argument("--save")
    eval_train_specialist_policy = eval_sub.add_parser(
        "train-specialist-policy",
        help="Train an aggregated specialist arbitration policy from benchmark reports",
    )
    eval_train_specialist_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_specialist_policy.add_argument("--reports-dir")
    eval_train_specialist_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_specialist_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_specialist_policy.add_argument("--save")
    eval_train_arbitration_policy = eval_sub.add_parser(
        "train-arbitration-policy",
        help="Train an aggregated arbitration policy from benchmark reports",
    )
    eval_train_arbitration_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_arbitration_policy.add_argument("--reports-dir")
    eval_train_arbitration_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_arbitration_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_arbitration_policy.add_argument("--save")
    eval_train_spawn_policy = eval_sub.add_parser(
        "train-spawn-policy",
        help="Train an aggregated spawn policy from benchmark reports",
    )
    eval_train_spawn_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_spawn_policy.add_argument("--reports-dir")
    eval_train_spawn_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_spawn_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_spawn_policy.add_argument("--save")
    eval_train_memory_policy = eval_sub.add_parser(
        "train-memory-policy",
        help="Train an aggregated memory-fold policy from benchmark reports",
    )
    eval_train_memory_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_memory_policy.add_argument("--reports-dir")
    eval_train_memory_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_memory_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_memory_policy.add_argument("--save")
    eval_train_rl_policy = eval_sub.add_parser(
        "train-rl-policy",
        help="Train RL-style depth + arbitration + specialist policy updates with reward credit assignment",
    )
    eval_train_rl_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_rl_policy.add_argument("--reports-dir")
    eval_train_rl_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_rl_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_rl_policy.add_argument("--learning-rate", type=float, default=0.18)
    eval_train_rl_policy.add_argument("--epochs", type=int, default=3)
    eval_train_rl_policy.add_argument("--reward-runtime-weight", type=float, default=0.35)
    eval_train_rl_policy.add_argument("--reward-failure-penalty", type=float, default=1.0)
    eval_train_rl_policy.add_argument("--save")
    eval_train_rl_spawn_policy = eval_sub.add_parser(
        "train-rl-spawn-policy",
        help="Train RL-style spawn policy updates with role/tool reward credit assignment",
    )
    eval_train_rl_spawn_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_rl_spawn_policy.add_argument("--reports-dir")
    eval_train_rl_spawn_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_rl_spawn_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_rl_spawn_policy.add_argument("--learning-rate", type=float, default=0.18)
    eval_train_rl_spawn_policy.add_argument("--epochs", type=int, default=3)
    eval_train_rl_spawn_policy.add_argument("--reward-runtime-weight", type=float, default=0.35)
    eval_train_rl_spawn_policy.add_argument("--reward-failure-penalty", type=float, default=1.0)
    eval_train_rl_spawn_policy.add_argument("--save")
    eval_train_rl_memory_policy = eval_sub.add_parser(
        "train-rl-memory-policy",
        help="Train RL-style memory fold policy updates with degradation/runtime reward credit assignment",
    )
    eval_train_rl_memory_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_rl_memory_policy.add_argument("--reports-dir")
    eval_train_rl_memory_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_rl_memory_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_rl_memory_policy.add_argument("--learning-rate", type=float, default=0.18)
    eval_train_rl_memory_policy.add_argument("--epochs", type=int, default=3)
    eval_train_rl_memory_policy.add_argument("--reward-runtime-weight", type=float, default=0.35)
    eval_train_rl_memory_policy.add_argument("--reward-failure-penalty", type=float, default=1.0)
    eval_train_rl_memory_policy.add_argument("--save")
    eval_train_rl_orch_policy = eval_sub.add_parser(
        "train-rl-orchestration-policy",
        help="Train bundled RL-style depth/arbitration/specialist/spawn/memory policy updates",
    )
    eval_train_rl_orch_policy.add_argument("--reports", help="comma-separated report paths")
    eval_train_rl_orch_policy.add_argument("--reports-dir")
    eval_train_rl_orch_policy.add_argument("--target-quality", type=float, default=0.65)
    eval_train_rl_orch_policy.add_argument("--target-runtime-sec", type=float)
    eval_train_rl_orch_policy.add_argument("--learning-rate", type=float, default=0.18)
    eval_train_rl_orch_policy.add_argument("--epochs", type=int, default=3)
    eval_train_rl_orch_policy.add_argument("--reward-runtime-weight", type=float, default=0.35)
    eval_train_rl_orch_policy.add_argument("--reward-failure-penalty", type=float, default=1.0)
    eval_train_rl_orch_policy.add_argument("--save")
    eval_autolearn = eval_sub.add_parser(
        "autolearn",
        help="Run benchmark iterations and auto-update depth + arbitration + specialist + spawn + memory policies from fresh telemetry",
    )
    eval_autolearn.add_argument("--dataset")
    eval_autolearn.add_argument("--mode", choices=["deep", "fast"], default="deep")
    eval_autolearn.add_argument("--limit", type=int)
    eval_autolearn.add_argument("--iterations", type=int, default=2)
    eval_autolearn.add_argument("--lookback-reports", type=int, default=6)
    eval_autolearn.add_argument("--target-quality", type=float, default=0.65)
    eval_autolearn.add_argument("--target-runtime-sec", type=float)
    eval_autolearn.add_argument("--learning-rate", type=float, default=0.35)
    eval_autolearn.add_argument("--optimizer", choices=["blend", "rl"], default="rl")
    eval_autolearn.add_argument("--rl-epochs", type=int, default=3)
    eval_autolearn.add_argument("--rl-reward-runtime-weight", type=float, default=0.35)
    eval_autolearn.add_argument("--rl-reward-failure-penalty", type=float, default=1.0)
    eval_autolearn.add_argument("--reports-dir")
    eval_autolearn.add_argument(
        "--depth-policy-path",
        default="rim/eval/policies/depth_policy.json",
    )
    eval_autolearn.add_argument(
        "--specialist-policy-path",
        default="rim/eval/policies/specialist_policy.json",
    )
    eval_autolearn.add_argument(
        "--arbitration-policy-path",
        default="rim/eval/policies/arbitration_policy.json",
    )
    eval_autolearn.add_argument(
        "--spawn-policy-path",
        default="rim/eval/policies/spawn_policy.json",
    )
    eval_autolearn.add_argument(
        "--memory-policy-path",
        default="rim/eval/policies/memory_policy.json",
    )
    eval_autolearn.add_argument("--save")

    sub.add_parser("health", help="Healthcheck DB and providers")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run" and args.run_command == "list":
        raise SystemExit(_cmd_run_list(args))
    if args.command == "analyze":
        raise SystemExit(asyncio.run(_cmd_analyze(args)))
    if args.command == "run" and args.run_command == "show":
        raise SystemExit(_cmd_run_show(args))
    if args.command == "run" and args.run_command == "logs":
        raise SystemExit(_cmd_run_logs(args))
    if args.command == "run" and args.run_command == "feedback":
        raise SystemExit(_cmd_run_feedback(args))
    if args.command == "run" and args.run_command == "cancel":
        raise SystemExit(asyncio.run(_cmd_run_cancel(args)))
    if args.command == "run" and args.run_command == "retry":
        raise SystemExit(asyncio.run(_cmd_run_retry(args)))
    if args.command == "eval" and args.eval_command == "run":
        raise SystemExit(asyncio.run(_cmd_eval_run(args)))
    if args.command == "eval" and args.eval_command == "duel":
        raise SystemExit(asyncio.run(_cmd_eval_duel(args)))
    if args.command == "eval" and args.eval_command == "list":
        raise SystemExit(_cmd_eval_list(args))
    if args.command == "eval" and args.eval_command == "baseline":
        raise SystemExit(_cmd_eval_baseline(args))
    if args.command == "eval" and args.eval_command == "baseline-llm":
        raise SystemExit(asyncio.run(_cmd_eval_baseline_llm(args)))
    if args.command == "eval" and args.eval_command == "compare":
        raise SystemExit(_cmd_eval_compare(args))
    if args.command == "eval" and args.eval_command == "gate":
        raise SystemExit(_cmd_eval_gate(args))
    if args.command == "eval" and args.eval_command == "blindpack":
        raise SystemExit(_cmd_eval_blindpack(args))
    if args.command == "eval" and args.eval_command == "calibrate":
        raise SystemExit(_cmd_eval_calibrate(args))
    if args.command == "eval" and args.eval_command == "calibrate-loop":
        raise SystemExit(asyncio.run(_cmd_eval_calibrate_loop(args)))
    if args.command == "eval" and args.eval_command == "train-policy":
        raise SystemExit(_cmd_eval_train_policy(args))
    if args.command == "eval" and args.eval_command == "train-specialist-policy":
        raise SystemExit(_cmd_eval_train_specialist_policy(args))
    if args.command == "eval" and args.eval_command == "train-arbitration-policy":
        raise SystemExit(_cmd_eval_train_arbitration_policy(args))
    if args.command == "eval" and args.eval_command == "train-spawn-policy":
        raise SystemExit(_cmd_eval_train_spawn_policy(args))
    if args.command == "eval" and args.eval_command == "train-memory-policy":
        raise SystemExit(_cmd_eval_train_memory_policy(args))
    if args.command == "eval" and args.eval_command == "train-rl-policy":
        raise SystemExit(_cmd_eval_train_rl_policy(args))
    if args.command == "eval" and args.eval_command == "train-rl-spawn-policy":
        raise SystemExit(_cmd_eval_train_rl_spawn_policy(args))
    if args.command == "eval" and args.eval_command == "train-rl-memory-policy":
        raise SystemExit(_cmd_eval_train_rl_memory_policy(args))
    if args.command == "eval" and args.eval_command == "train-rl-orchestration-policy":
        raise SystemExit(_cmd_eval_train_rl_orchestration_policy(args))
    if args.command == "eval" and args.eval_command == "autolearn":
        raise SystemExit(asyncio.run(_cmd_eval_autolearn(args)))
    if args.command == "health":
        raise SystemExit(asyncio.run(_cmd_health()))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
