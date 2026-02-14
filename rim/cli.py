from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest
from rim.eval.runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REPORTS_DIR,
    compare_reports,
    list_reports,
    load_report,
    run_benchmark,
    save_report,
)
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def _build_orchestrator() -> RimOrchestrator:
    return RimOrchestrator(repository=RunRepository(), router=ProviderRouter())


async def _cmd_analyze(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    request = AnalyzeRequest(
        idea=args.idea,
        mode=args.mode,
        domain=args.domain,
        constraints=args.constraint or [],
        desired_outcome=args.desired_outcome,
    )
    result = await orchestrator.analyze(request)
    payload = result.model_dump()
    if args.save:
        Path(args.save).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"run_id: {result.run_id}")
        print(f"mode: {result.mode}")
        print(f"confidence_score: {result.confidence_score:.2f}")
        print(f"synthesized_idea: {result.synthesized_idea}")
    return 0


def _cmd_run_show(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    run = orchestrator.get_run(args.run_id)
    if run is None:
        print("Run not found")
        return 1
    print(json.dumps(run.model_dump(), indent=2))
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


def _cmd_eval_list(_args: argparse.Namespace) -> int:
    reports = [str(path) for path in list_reports(DEFAULT_REPORTS_DIR)]
    print(json.dumps({"reports": reports}, indent=2))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rim")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run idea analysis")
    analyze.add_argument("--idea", required=True)
    analyze.add_argument("--mode", choices=["deep", "fast"], default="deep")
    analyze.add_argument("--domain")
    analyze.add_argument("--constraint", action="append")
    analyze.add_argument("--desired-outcome")
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--save")

    run = sub.add_parser("run", help="Inspect runs")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    run_show = run_sub.add_parser("show", help="Show run details")
    run_show.add_argument("run_id")
    run_logs = run_sub.add_parser("logs", help="Show stage telemetry logs")
    run_logs.add_argument("run_id")

    eval_parser = sub.add_parser("eval", help="Benchmark and scoring")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run", help="Run benchmark dataset")
    eval_run.add_argument("--dataset")
    eval_run.add_argument("--mode", choices=["deep", "fast"], default="deep")
    eval_run.add_argument("--limit", type=int)
    eval_run.add_argument("--save")
    eval_sub.add_parser("list", help="List saved benchmark reports")
    eval_compare = eval_sub.add_parser(
        "compare",
        help="Compare two benchmark reports (defaults to latest two)",
    )
    eval_compare.add_argument("--base")
    eval_compare.add_argument("--target")

    sub.add_parser("health", help="Healthcheck DB and providers")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        raise SystemExit(asyncio.run(_cmd_analyze(args)))
    if args.command == "run" and args.run_command == "show":
        raise SystemExit(_cmd_run_show(args))
    if args.command == "run" and args.run_command == "logs":
        raise SystemExit(_cmd_run_logs(args))
    if args.command == "eval" and args.eval_command == "run":
        raise SystemExit(asyncio.run(_cmd_eval_run(args)))
    if args.command == "eval" and args.eval_command == "list":
        raise SystemExit(_cmd_eval_list(args))
    if args.command == "eval" and args.eval_command == "compare":
        raise SystemExit(_cmd_eval_compare(args))
    if args.command == "health":
        raise SystemExit(asyncio.run(_cmd_health()))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
