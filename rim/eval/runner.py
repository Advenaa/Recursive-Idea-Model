from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, AnalyzeResult, DecompositionNode
from rim.eval.benchmark import evaluate_run

DEFAULT_DATASET_PATH = Path("rim/eval/data/benchmark_ideas.jsonl")
DEFAULT_REPORTS_DIR = Path("rim/eval/reports")


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


def _normalize_domain(value: Any) -> str:
    parsed = str(value or "").strip().lower()
    return parsed if parsed else "unspecified"


def _summarize_domain_metrics(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for item in runs:
        domain = _normalize_domain(item.get("domain"))
        status = str(item.get("status") or "unknown").strip().lower()
        bucket = metrics.setdefault(
            domain,
            {
                "dataset_size": 0,
                "success_count": 0,
                "failure_count": 0,
                "average_runtime_sec": 0.0,
                "average_quality_score": 0.0,
                "failure_modes": {},
                "_runtime_total": 0.0,
                "_quality_total": 0.0,
            },
        )
        bucket["dataset_size"] += 1

        if status == "completed":
            quality = item.get("quality")
            if not isinstance(quality, dict):
                continue
            quality_score = quality.get("quality_score")
            if not isinstance(quality_score, (int, float)):
                continue
            runtime = float(item.get("runtime_sec", 0.0))
            bucket["success_count"] += 1
            bucket["_runtime_total"] += runtime
            bucket["_quality_total"] += float(quality_score)
            continue

        if status in {"failed", "canceled"}:
            bucket["failure_count"] += 1
            mode_name = str(item.get("error_type") or status or "unknown")
            failure_modes = bucket["failure_modes"]
            failure_modes[mode_name] = failure_modes.get(mode_name, 0) + 1

    for domain, bucket in metrics.items():
        success_count = int(bucket["success_count"])
        if success_count > 0:
            bucket["average_runtime_sec"] = round(bucket["_runtime_total"] / success_count, 3)
            bucket["average_quality_score"] = round(bucket["_quality_total"] / success_count, 3)
        bucket.pop("_runtime_total", None)
        bucket.pop("_quality_total", None)
        metrics[domain] = bucket
    return metrics


def _summarize_report(
    started_at: str,
    dataset_path: Path,
    mode: str,
    total_runtime_sec: float,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_runs = [
        item
        for item in runs
        if item.get("status", "completed") == "completed"
        and isinstance(item.get("quality"), dict)
        and isinstance(item["quality"].get("quality_score"), (int, float))
    ]
    failed_runs = [item for item in runs if item.get("status") == "failed"]
    failure_modes: dict[str, int] = {}
    for item in failed_runs:
        mode_name = str(item.get("error_type") or "UnknownError")
        failure_modes[mode_name] = failure_modes.get(mode_name, 0) + 1

    if not runs:
        return {
            "created_at": started_at,
            "dataset_size": 0,
            "mode": mode,
            "dataset_path": str(dataset_path),
            "total_runtime_sec": total_runtime_sec,
            "average_runtime_sec": 0.0,
            "average_quality_score": 0.0,
            "success_count": 0,
            "failure_count": 0,
            "failure_modes": {},
            "domain_metrics": {},
            "runs": [],
        }

    avg_runtime = 0.0
    avg_quality = 0.0
    if completed_runs:
        avg_runtime = sum(item["runtime_sec"] for item in completed_runs) / len(completed_runs)
        avg_quality = sum(item["quality"]["quality_score"] for item in completed_runs) / len(
            completed_runs
        )
    return {
        "created_at": started_at,
        "dataset_size": len(runs),
        "mode": mode,
        "dataset_path": str(dataset_path),
        "total_runtime_sec": total_runtime_sec,
        "average_runtime_sec": round(avg_runtime, 3),
        "average_quality_score": round(avg_quality, 3),
        "success_count": len(completed_runs),
        "failure_count": len(failed_runs),
        "failure_modes": failure_modes,
        "domain_metrics": _summarize_domain_metrics(runs),
        "runs": runs,
    }


def _single_pass_baseline_result(idea: str, run_id: str) -> AnalyzeResult:
    return AnalyzeResult(
        run_id=run_id,
        mode="fast",
        input_idea=idea,
        decomposition=[
            DecompositionNode(
                depth=0,
                component_text=idea.strip(),
                node_type="claim",
                confidence=0.95,
            )
        ],
        critic_findings=[],
        synthesized_idea=idea.strip(),
        changes_summary=[],
        residual_risks=[],
        next_experiments=[],
        confidence_score=0.45,
    )


async def run_benchmark(
    orchestrator: RimOrchestrator,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mode: str = "deep",
    limit: int | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
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
        try:
            result = await orchestrator.analyze(request)
            runtime_sec = round(time.perf_counter() - run_started, 3)
            score = evaluate_run(result, heuristic_reviewer, domain=request.domain)
            runs.append(
                {
                    "id": item.get("id"),
                    "idea": request.idea,
                    "domain": request.domain,
                    "mode": mode,
                    "runtime_sec": runtime_sec,
                    "run_id": result.run_id,
                    "status": "completed",
                    "quality": score,
                    "synthesized_idea": result.synthesized_idea,
                    "changes_summary": list(result.changes_summary),
                    "residual_risks": list(result.residual_risks),
                    "next_experiments": list(result.next_experiments),
                }
            )
        except Exception as exc:  # noqa: BLE001
            runtime_sec = round(time.perf_counter() - run_started, 3)
            runs.append(
                {
                    "id": item.get("id"),
                    "idea": request.idea,
                    "domain": request.domain,
                    "mode": mode,
                    "runtime_sec": runtime_sec,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    total_runtime_sec = round(time.perf_counter() - started, 3)
    return _summarize_report(
        started_at=started_at,
        dataset_path=dataset_path,
        mode=mode,
        total_runtime_sec=total_runtime_sec,
        runs=runs,
    )


def run_single_pass_baseline(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    limit: int | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    dataset = load_dataset(dataset_path)
    if limit is not None and limit > 0:
        dataset = dataset[:limit]

    runs: list[dict[str, Any]] = []
    started = time.perf_counter()

    for item in dataset:
        item_id = str(item.get("id") or f"row-{len(runs) + 1}")
        run_started = time.perf_counter()
        result = _single_pass_baseline_result(
            idea=str(item["idea"]),
            run_id=f"baseline-{item_id}",
        )
        runtime_sec = round(time.perf_counter() - run_started, 3)
        domain = item.get("domain")
        score = evaluate_run(result, heuristic_reviewer, domain=domain)
        runs.append(
            {
                "id": item.get("id"),
                "idea": result.input_idea,
                "domain": domain,
                "mode": "single_pass_baseline",
                "runtime_sec": runtime_sec,
                "run_id": result.run_id,
                "status": "completed",
                "quality": score,
                "synthesized_idea": result.synthesized_idea,
                "changes_summary": list(result.changes_summary),
                "residual_risks": list(result.residual_risks),
                "next_experiments": list(result.next_experiments),
            }
        )

    total_runtime_sec = round(time.perf_counter() - started, 3)
    return _summarize_report(
        started_at=started_at,
        dataset_path=dataset_path,
        mode="single_pass_baseline",
        total_runtime_sec=total_runtime_sec,
        runs=runs,
    )


async def run_duel_benchmark(
    orchestrator: RimOrchestrator,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mode: str = "deep",
    limit: int | None = None,
    min_quality_delta: float = 0.0,
    max_runtime_delta_sec: float | None = None,
    min_shared_runs: int = 1,
) -> dict[str, Any]:
    baseline = run_single_pass_baseline(
        dataset_path=dataset_path,
        limit=limit,
    )
    target = await run_benchmark(
        orchestrator=orchestrator,
        dataset_path=dataset_path,
        mode=mode,
        limit=limit,
    )
    comparison = compare_reports(base=baseline, target=target)
    gate = evaluate_regression_gate(
        comparison=comparison,
        min_quality_delta=min_quality_delta,
        max_runtime_delta_sec=max_runtime_delta_sec,
        min_shared_runs=min_shared_runs,
    )
    return {
        "baseline": baseline,
        "target": target,
        "comparison": comparison,
        "gate": gate,
    }


def save_report(report: dict[str, Any], output_path: Path | None = None) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return output_path

    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    auto_path = DEFAULT_REPORTS_DIR / f"benchmark_{stamp}.json"
    auto_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return auto_path


def build_blind_review_packet(
    report: dict[str, Any],
    *,
    max_items: int | None = None,
) -> dict[str, Any]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        runs = []
    completed = [
        item
        for item in runs
        if isinstance(item, dict) and str(item.get("status") or "").strip().lower() == "completed"
    ]
    if max_items is not None and max_items > 0:
        completed = completed[:max_items]

    items: list[dict[str, Any]] = []
    for index, item in enumerate(completed, start=1):
        items.append(
            {
                "blind_id": f"candidate-{index:03d}",
                "idea": str(item.get("idea") or ""),
                "domain": item.get("domain"),
                "synthesized_idea": str(item.get("synthesized_idea") or ""),
                "changes_summary": list(item.get("changes_summary") or []),
                "residual_risks": list(item.get("residual_risks") or []),
                "next_experiments": list(item.get("next_experiments") or []),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report_created_at": report.get("created_at"),
        "source_dataset_path": report.get("dataset_path"),
        "item_count": len(items),
        "rubric": {
            "dimensions": ["rigor", "novelty", "practicality", "overall"],
            "scale": "1-5",
            "instructions": (
                "Score each candidate independently. Do not infer provider/mode metadata."
            ),
        },
        "items": items,
    }


def save_blind_review_packet(packet: dict[str, Any], output_path: Path | None = None) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
        return output_path

    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    auto_path = DEFAULT_REPORTS_DIR / f"blind_review_{stamp}.json"
    auto_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return auto_path


def list_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> list[Path]:
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("*.json"))


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid report file: {path}")
    return payload


def compare_reports(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    base_runtime = float(base.get("average_runtime_sec", 0.0))
    target_runtime = float(target.get("average_runtime_sec", 0.0))
    base_quality = float(base.get("average_quality_score", 0.0))
    target_quality = float(target.get("average_quality_score", 0.0))

    def _eligible(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if item.get("id") is None:
            return False
        if item.get("status", "completed") != "completed":
            return False
        quality = item.get("quality")
        if not isinstance(quality, dict):
            return False
        return isinstance(quality.get("quality_score"), (int, float))

    base_runs = {
        str(item.get("id")): item
        for item in base.get("runs", [])
        if _eligible(item)
    }
    target_runs = {
        str(item.get("id")): item
        for item in target.get("runs", [])
        if _eligible(item)
    }

    shared_ids = sorted(set(base_runs.keys()) & set(target_runs.keys()))
    run_deltas: list[dict[str, Any]] = []
    for run_id in shared_ids:
        base_run = base_runs[run_id]
        target_run = target_runs[run_id]
        base_q = float(base_run.get("quality", {}).get("quality_score", 0.0))
        target_q = float(target_run.get("quality", {}).get("quality_score", 0.0))
        base_r = float(base_run.get("runtime_sec", 0.0))
        target_r = float(target_run.get("runtime_sec", 0.0))
        run_deltas.append(
            {
                "id": run_id,
                "quality_delta": round(target_q - base_q, 4),
                "runtime_delta_sec": round(target_r - base_r, 4),
            }
        )

    base_domain_metrics = base.get("domain_metrics")
    target_domain_metrics = target.get("domain_metrics")
    domain_deltas: list[dict[str, Any]] = []
    if isinstance(base_domain_metrics, dict) and isinstance(target_domain_metrics, dict):
        shared_domains = sorted(set(base_domain_metrics.keys()) & set(target_domain_metrics.keys()))
        for domain in shared_domains:
            base_bucket = base_domain_metrics.get(domain)
            target_bucket = target_domain_metrics.get(domain)
            if not isinstance(base_bucket, dict) or not isinstance(target_bucket, dict):
                continue
            base_quality_domain = float(base_bucket.get("average_quality_score", 0.0))
            target_quality_domain = float(target_bucket.get("average_quality_score", 0.0))
            base_runtime_domain = float(base_bucket.get("average_runtime_sec", 0.0))
            target_runtime_domain = float(target_bucket.get("average_runtime_sec", 0.0))
            domain_deltas.append(
                {
                    "domain": domain,
                    "quality_delta": round(target_quality_domain - base_quality_domain, 4),
                    "runtime_delta_sec": round(target_runtime_domain - base_runtime_domain, 4),
                    "base_success_count": int(base_bucket.get("success_count", 0)),
                    "target_success_count": int(target_bucket.get("success_count", 0)),
                }
            )

    return {
        "base_created_at": base.get("created_at"),
        "target_created_at": target.get("created_at"),
        "base_mode": base.get("mode"),
        "target_mode": target.get("mode"),
        "base_dataset_size": int(base.get("dataset_size", 0)),
        "target_dataset_size": int(target.get("dataset_size", 0)),
        "average_quality_delta": round(target_quality - base_quality, 4),
        "average_runtime_delta_sec": round(target_runtime - base_runtime, 4),
        "shared_run_count": len(shared_ids),
        "domain_deltas": domain_deltas,
        "run_deltas": run_deltas,
    }


def evaluate_regression_gate(
    comparison: dict[str, Any],
    min_quality_delta: float = 0.0,
    max_runtime_delta_sec: float | None = None,
    min_shared_runs: int = 1,
) -> dict[str, Any]:
    quality_delta = float(comparison.get("average_quality_delta", 0.0))
    runtime_delta = float(comparison.get("average_runtime_delta_sec", 0.0))
    shared_runs = int(comparison.get("shared_run_count", 0))

    checks: list[dict[str, Any]] = [
        {
            "name": "shared_runs",
            "passed": shared_runs >= min_shared_runs,
            "observed": shared_runs,
            "threshold": min_shared_runs,
            "direction": ">=",
        },
        {
            "name": "quality_delta",
            "passed": quality_delta >= min_quality_delta,
            "observed": quality_delta,
            "threshold": min_quality_delta,
            "direction": ">=",
        },
    ]

    if max_runtime_delta_sec is not None:
        checks.append(
            {
                "name": "runtime_delta_sec",
                "passed": runtime_delta <= max_runtime_delta_sec,
                "observed": runtime_delta,
                "threshold": max_runtime_delta_sec,
                "direction": "<=",
            }
        )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "checks": checks,
        "observed": {
            "average_quality_delta": quality_delta,
            "average_runtime_delta_sec": runtime_delta,
            "shared_run_count": shared_runs,
        },
        "thresholds": {
            "min_quality_delta": min_quality_delta,
            "max_runtime_delta_sec": max_runtime_delta_sec,
            "min_shared_runs": min_shared_runs,
        },
    }


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def calibrate_depth_allocator(
    report: dict[str, Any],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
) -> dict[str, Any]:
    avg_quality = float(report.get("average_quality_score", 0.0))
    avg_runtime = float(report.get("average_runtime_sec", 0.0))
    dataset_size = max(1, int(report.get("dataset_size", 1)))
    failures = int(report.get("failure_count", 0))
    failure_rate = failures / float(dataset_size)

    quality_gap = float(target_quality) - avg_quality
    quality_pressure = quality_gap / max(float(target_quality), 0.05)
    runtime_pressure = 0.0
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        runtime_pressure = (avg_runtime - float(target_runtime_sec)) / float(target_runtime_sec)

    depth_pressure = quality_pressure - (0.6 * runtime_pressure) - (0.4 * failure_rate)
    depth_pressure = _clamp_float(depth_pressure, -1.0, 1.0)

    base = {
        "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": 0.78,
        "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": 2,
        "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": 1,
        "RIM_MAX_ANALYSIS_CYCLES": 1,
    }
    suggested_min_conf = round(_clamp_float(base["RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE"] + (0.10 * depth_pressure), 0.65, 0.93), 3)
    suggested_max_risks = _clamp_int(
        int(round(base["RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS"] - (1.5 * depth_pressure))),
        0,
        4,
    )
    suggested_max_high = _clamp_int(
        int(round(base["RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS"] - (1.0 * depth_pressure))),
        0,
        3,
    )
    suggested_max_cycles = _clamp_int(
        int(round(base["RIM_MAX_ANALYSIS_CYCLES"] + (2.0 * max(0.0, depth_pressure)))),
        1,
        4,
    )

    rationale: list[str] = []
    if depth_pressure > 0.2:
        rationale.append("Quality is below target; increase analytical depth and stricter stop thresholds.")
    elif depth_pressure < -0.2:
        rationale.append("Runtime/failure pressure is high; relax depth to stabilize throughput.")
    else:
        rationale.append("Current depth profile is near target; keep moderate settings.")
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        if avg_runtime > float(target_runtime_sec):
            rationale.append("Average runtime exceeds target runtime budget.")
        else:
            rationale.append("Average runtime is within target runtime budget.")
    if failure_rate > 0.2:
        rationale.append("Failure rate is elevated; avoid aggressive depth expansion until reliability improves.")

    env = {
        "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": suggested_min_conf,
        "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": suggested_max_risks,
        "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": suggested_max_high,
        "RIM_MAX_ANALYSIS_CYCLES": suggested_max_cycles,
    }
    return {
        "inputs": {
            "average_quality_score": avg_quality,
            "average_runtime_sec": avg_runtime,
            "failure_rate": round(failure_rate, 4),
            "dataset_size": dataset_size,
            "target_quality": float(target_quality),
            "target_runtime_sec": target_runtime_sec,
        },
        "signals": {
            "quality_pressure": round(quality_pressure, 4),
            "runtime_pressure": round(runtime_pressure, 4),
            "depth_pressure": round(depth_pressure, 4),
        },
        "base": base,
        "recommended_env": env,
        "rationale": rationale,
    }


def calibration_env_exports(calibration: dict[str, Any]) -> list[str]:
    env = calibration.get("recommended_env")
    if not isinstance(env, dict):
        return []
    lines: list[str] = []
    for key in sorted(env.keys()):
        value = env[key]
        if isinstance(value, float):
            lines.append(f"export {key}={value:.3f}".rstrip("0").rstrip("."))
        else:
            lines.append(f"export {key}={value}")
    return lines
