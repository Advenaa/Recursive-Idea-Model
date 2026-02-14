from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, AnalyzeResult
from rim.eval.benchmark import evaluate_run

DEFAULT_DATASET_PATH = Path("rim/eval/data/benchmark_ideas.jsonl")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        payload = json.loads(row)
        if isinstance(payload, dict) and payload.get("idea"):
            items.append(payload)
    return items


def heuristic_reviewer(result: AnalyzeResult) -> tuple[float, float, float]:
    node_count = len(result.decomposition)
    finding_count = len(result.critic_findings)
    change_count = len(result.changes_summary)
    experiment_count = len(result.next_experiments)

    rigor = min(1.0, 0.2 + (0.05 * min(node_count, 10)) + (0.03 * min(finding_count, 20)))
    novelty = min(1.0, 0.2 + (0.08 * min(change_count, 8)))
    practicality = min(1.0, 0.2 + (0.12 * min(experiment_count, 6)))
    return rigor, novelty, practicality


async def run_benchmark(
    orchestrator: RimOrchestrator,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mode: str = "deep",
    limit: int | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    if limit is not None and limit > 0:
        dataset = dataset[:limit]

    runs: list[dict[str, Any]] = []
    started = time.perf_counter()

    for item in dataset:
        request = AnalyzeRequest(
            idea=str(item["idea"]),
            mode=mode,
            domain=item.get("domain"),
            constraints=item.get("constraints") or [],
            desired_outcome=item.get("desired_outcome"),
        )
        run_started = time.perf_counter()
        result = await orchestrator.analyze(request)
        runtime_sec = round(time.perf_counter() - run_started, 3)
        score = evaluate_run(result, heuristic_reviewer)

        runs.append(
            {
                "id": item.get("id"),
                "idea": request.idea,
                "mode": mode,
                "runtime_sec": runtime_sec,
                "run_id": result.run_id,
                "quality": score,
            }
        )

    total_runtime_sec = round(time.perf_counter() - started, 3)
    if not runs:
        return {
            "dataset_size": 0,
            "mode": mode,
            "total_runtime_sec": total_runtime_sec,
            "average_runtime_sec": 0.0,
            "average_quality_score": 0.0,
            "runs": [],
        }

    avg_runtime = sum(item["runtime_sec"] for item in runs) / len(runs)
    avg_quality = sum(item["quality"]["quality_score"] for item in runs) / len(runs)
    return {
        "dataset_size": len(runs),
        "mode": mode,
        "total_runtime_sec": total_runtime_sec,
        "average_runtime_sec": round(avg_runtime, 3),
        "average_quality_score": round(avg_quality, 3),
        "runs": runs,
    }
