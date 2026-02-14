import argparse
import asyncio
import json
from pathlib import Path

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest
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

    sub.add_parser("health", help="Healthcheck DB and providers")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        raise SystemExit(asyncio.run(_cmd_analyze(args)))
    if args.command == "run" and args.run_command == "show":
        raise SystemExit(_cmd_run_show(args))
    if args.command == "health":
        raise SystemExit(asyncio.run(_cmd_health()))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
