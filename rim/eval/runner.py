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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _extract_run_telemetry(orchestrator: Any, run_id: str) -> dict[str, Any]:
    getter = getattr(orchestrator, "get_run_logs", None)
    if getter is None or not callable(getter):
        return {}
    try:
        payload = getter(run_id)
    except Exception:  # noqa: BLE001
        return {}
    logs = getattr(payload, "logs", None)
    if not isinstance(logs, list):
        return {}

    telemetry = {
        "disagreement_count": 0,
        "diversity_flagged_count": 0,
        "arbitration_resolved_count": 0,
        "devils_advocate_count": 0,
        "specialist_count": 0,
        "spawn_selected_count": 0,
        "spawn_dynamic_count": 0,
        "memory_fold_count": 0,
        "memory_fold_degradation_count": 0,
        "memory_fold_avg_novelty_ratio": 0.0,
        "memory_fold_avg_duplicate_ratio": 0.0,
    }
    memory_fold_novelty_total = 0.0
    memory_fold_duplicate_total = 0.0
    for item in logs:
        stage = str(getattr(item, "stage", "")).strip().lower()
        meta = getattr(item, "meta", None)
        if not isinstance(meta, dict):
            continue
        if stage == "challenge_reconciliation":
            telemetry["disagreement_count"] = max(
                telemetry["disagreement_count"],
                _to_int(meta.get("disagreement_count"), 0),
            )
            telemetry["diversity_flagged_count"] = max(
                telemetry["diversity_flagged_count"],
                _to_int(meta.get("diversity_flagged_count"), 0),
            )
            continue
        if stage == "challenge_arbitration":
            telemetry["arbitration_resolved_count"] = max(
                telemetry["arbitration_resolved_count"],
                _to_int(meta.get("resolved_count"), 0),
            )
            telemetry["devils_advocate_count"] = max(
                telemetry["devils_advocate_count"],
                _to_int(meta.get("devils_advocate_count"), 0),
            )
            telemetry["specialist_count"] = max(
                telemetry["specialist_count"],
                _to_int(meta.get("specialist_count"), 0),
            )
            continue
        if stage == "specialization_spawn":
            telemetry["spawn_selected_count"] = max(
                telemetry["spawn_selected_count"],
                _to_int(meta.get("selected_count"), 0),
            )
            extra = list(meta.get("extra_critics") or [])
            dynamic = [
                entry
                for entry in extra
                if isinstance(entry, dict) and bool(entry.get("dynamic"))
            ]
            telemetry["spawn_dynamic_count"] = max(
                telemetry["spawn_dynamic_count"],
                len(dynamic),
            )
            continue
        if stage == "memory_fold":
            telemetry["memory_fold_count"] += 1
            if bool(meta.get("degradation_detected")):
                telemetry["memory_fold_degradation_count"] += 1
            memory_fold_novelty_total += _to_float(meta.get("novelty_ratio"), 0.0)
            memory_fold_duplicate_total += _to_float(meta.get("duplicate_ratio"), 0.0)

    if telemetry["memory_fold_count"] > 0:
        folds = float(telemetry["memory_fold_count"])
        telemetry["memory_fold_avg_novelty_ratio"] = round(memory_fold_novelty_total / folds, 4)
        telemetry["memory_fold_avg_duplicate_ratio"] = round(memory_fold_duplicate_total / folds, 4)
    if not any(value for value in telemetry.values()):
        return {}
    return telemetry


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
            telemetry = _extract_run_telemetry(orchestrator, result.run_id)
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
                    "telemetry": telemetry,
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


def _specialist_report_signals(report: dict[str, Any]) -> dict[str, float]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        runs = []
    completed = [
        item
        for item in runs
        if isinstance(item, dict) and str(item.get("status") or "").strip().lower() == "completed"
    ]
    if not completed:
        return {
            "completed_runs": 0.0,
            "avg_disagreement_count": 0.0,
            "avg_diversity_flagged_count": 0.0,
            "avg_specialist_count": 0.0,
            "avg_spawn_dynamic_count": 0.0,
        }

    disagreement_total = 0.0
    diversity_total = 0.0
    specialist_total = 0.0
    dynamic_total = 0.0
    for item in completed:
        telemetry = item.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        disagreement_total += _to_float(telemetry.get("disagreement_count"), 0.0)
        diversity_total += _to_float(telemetry.get("diversity_flagged_count"), 0.0)
        specialist_total += _to_float(telemetry.get("specialist_count"), 0.0)
        dynamic_total += _to_float(telemetry.get("spawn_dynamic_count"), 0.0)

    count = float(len(completed))
    return {
        "completed_runs": count,
        "avg_disagreement_count": round(disagreement_total / count, 4),
        "avg_diversity_flagged_count": round(diversity_total / count, 4),
        "avg_specialist_count": round(specialist_total / count, 4),
        "avg_spawn_dynamic_count": round(dynamic_total / count, 4),
    }


def calibrate_specialist_arbitration_policy(
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
    telemetry = _specialist_report_signals(report)

    quality_gap = float(target_quality) - avg_quality
    quality_pressure = _clamp_float(quality_gap / max(float(target_quality), 0.05), -1.0, 1.0)
    runtime_pressure = 0.0
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        runtime_pressure = (avg_runtime - float(target_runtime_sec)) / float(target_runtime_sec)

    disagreement_pressure = _clamp_float(
        float(telemetry["avg_disagreement_count"]) / 2.0,
        0.0,
        1.0,
    )
    diversity_pressure = _clamp_float(
        float(telemetry["avg_diversity_flagged_count"]) / 2.0,
        0.0,
        1.0,
    )
    specialist_pressure = _clamp_float(
        float(telemetry["avg_specialist_count"]) / 2.0,
        0.0,
        1.0,
    )
    review_pressure = (
        (0.7 * quality_pressure)
        + (0.6 * disagreement_pressure)
        + (0.35 * diversity_pressure)
        + (0.2 * specialist_pressure)
        - (0.5 * max(runtime_pressure, 0.0))
        - (0.45 * failure_rate)
    )
    review_pressure = _clamp_float(review_pressure, -1.0, 1.0)

    base = {
        "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP": 1,
        "RIM_SPECIALIST_ARBITRATION_MAX_JOBS": 2,
        "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE": 0.78,
    }
    disable_for_latency = (
        quality_pressure <= 0.0
        and max(runtime_pressure, 0.0) > 0.3
        and disagreement_pressure < 0.25
        and diversity_pressure < 0.2
    )
    enable_loop = 0 if disable_for_latency else 1
    if review_pressure < -0.45:
        enable_loop = 0
    if review_pressure > 0.1:
        enable_loop = 1

    max_jobs = _clamp_int(
        int(
            round(
                base["RIM_SPECIALIST_ARBITRATION_MAX_JOBS"]
                + (2.0 * max(review_pressure, 0.0))
                + (1.0 * diversity_pressure)
            )
        ),
        0,
        6,
    )
    min_confidence = round(
        _clamp_float(
            base["RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE"] + (0.1 * review_pressure),
            0.6,
            0.95,
        ),
        3,
    )
    if enable_loop == 0:
        max_jobs = 0

    rationale: list[str] = []
    if quality_pressure > 0.15:
        rationale.append("Average quality is below target; specialist arbitration is expanded.")
    elif quality_pressure < -0.15:
        rationale.append("Average quality is above target; specialist arbitration can be relaxed.")
    if disagreement_pressure > 0.25 or diversity_pressure > 0.2:
        rationale.append("Frequent disagreement/diversity flags suggest stronger specialist review.")
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        if avg_runtime > float(target_runtime_sec):
            rationale.append("Average runtime exceeds target; specialist load is moderated.")
        else:
            rationale.append("Average runtime is within target runtime budget.")
    if failure_rate > 0.2:
        rationale.append("Failure rate is elevated; policy avoids aggressive arbitration expansion.")

    recommended_env = {
        "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP": enable_loop,
        "RIM_SPECIALIST_ARBITRATION_MAX_JOBS": max_jobs,
        "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE": min_confidence,
    }
    return {
        "inputs": {
            "average_quality_score": avg_quality,
            "average_runtime_sec": avg_runtime,
            "failure_rate": round(failure_rate, 4),
            "dataset_size": dataset_size,
            "target_quality": float(target_quality),
            "target_runtime_sec": target_runtime_sec,
            "telemetry": telemetry,
        },
        "signals": {
            "quality_pressure": round(quality_pressure, 4),
            "runtime_pressure": round(runtime_pressure, 4),
            "disagreement_pressure": round(disagreement_pressure, 4),
            "diversity_pressure": round(diversity_pressure, 4),
            "specialist_pressure": round(specialist_pressure, 4),
            "review_pressure": round(review_pressure, 4),
        },
        "base": base,
        "recommended_env": recommended_env,
        "rationale": rationale,
    }


def train_specialist_arbitration_policy(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
) -> dict[str, Any]:
    valid_reports: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        if int(report.get("dataset_size", 0)) <= 0:
            continue
        valid_reports.append(report)

    if not valid_reports:
        empty_policy = {
            "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP": 1,
            "RIM_SPECIALIST_ARBITRATION_MAX_JOBS": 2,
            "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE": 0.78,
        }
        return {
            "report_count": 0,
            "policy_env": empty_policy,
            "rationale": ["No valid reports were available; returning default specialist policy."],
        }

    weighted_enable = 0.0
    weighted_jobs = 0.0
    weighted_min_conf = 0.0
    total_weight = 0.0
    quality_sum = 0.0
    runtime_sum = 0.0
    failure_sum = 0.0
    disagreement_sum = 0.0
    diversity_sum = 0.0
    samples: list[dict[str, Any]] = []

    for report in valid_reports:
        calibration = calibrate_specialist_arbitration_policy(
            report,
            target_quality=target_quality,
            target_runtime_sec=target_runtime_sec,
        )
        env = calibration["recommended_env"]
        report_quality = float(report.get("average_quality_score", 0.0))
        report_failure = (
            float(report.get("failure_count", 0)) / max(1, int(report.get("dataset_size", 1)))
        )
        telemetry = calibration.get("inputs", {}).get("telemetry", {})
        avg_disagreement = _to_float(
            telemetry.get("avg_disagreement_count") if isinstance(telemetry, dict) else 0.0,
            0.0,
        )
        avg_diversity = _to_float(
            telemetry.get("avg_diversity_flagged_count") if isinstance(telemetry, dict) else 0.0,
            0.0,
        )
        weight = max(0.1, report_quality + 0.2 - (0.25 * report_failure) + (0.05 * avg_disagreement))
        total_weight += weight
        quality_sum += report_quality
        runtime_sum += float(report.get("average_runtime_sec", 0.0))
        failure_sum += report_failure
        disagreement_sum += avg_disagreement
        diversity_sum += avg_diversity

        weighted_enable += float(env["RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP"]) * weight
        weighted_jobs += float(env["RIM_SPECIALIST_ARBITRATION_MAX_JOBS"]) * weight
        weighted_min_conf += float(env["RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE"]) * weight
        samples.append(
            {
                "created_at": report.get("created_at"),
                "mode": report.get("mode"),
                "weight": round(weight, 4),
                "recommended_env": env,
            }
        )

    if total_weight <= 0:
        total_weight = float(len(valid_reports))
    enable_loop = 1 if (weighted_enable / total_weight) >= 0.5 else 0
    max_jobs = _clamp_int(
        int(round(weighted_jobs / total_weight)),
        0,
        6,
    )
    if enable_loop == 0:
        max_jobs = 0
    policy_env = {
        "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP": enable_loop,
        "RIM_SPECIALIST_ARBITRATION_MAX_JOBS": max_jobs,
        "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE": round(
            _clamp_float(weighted_min_conf / total_weight, 0.6, 0.95),
            3,
        ),
    }
    avg_quality = quality_sum / len(valid_reports)
    avg_runtime = runtime_sum / len(valid_reports)
    avg_failure = failure_sum / len(valid_reports)
    avg_disagreement = disagreement_sum / len(valid_reports)
    avg_diversity = diversity_sum / len(valid_reports)
    rationale = [
        "Policy aggregates specialist-calibration recommendations using weighted averaging.",
    ]
    if avg_quality < target_quality:
        rationale.append("Average quality is below target, so specialist arbitration remains active.")
    else:
        rationale.append("Average quality meets target, so specialist arbitration can be balanced.")
    if target_runtime_sec is not None and target_runtime_sec > 0 and avg_runtime > target_runtime_sec:
        rationale.append("Average runtime exceeds target, so specialist job volume is moderated.")
    if avg_failure > 0.2:
        rationale.append("Failure rate is elevated; policy avoids aggressive escalation.")
    if avg_disagreement > 0.5 or avg_diversity > 0.4:
        rationale.append("Disagreement/diversity pressure supports stronger specialist coverage.")

    return {
        "report_count": len(valid_reports),
        "policy_env": policy_env,
        "recommended_exports": calibration_env_exports({"recommended_env": policy_env}),
        "summary": {
            "average_quality_score": round(avg_quality, 4),
            "average_runtime_sec": round(avg_runtime, 4),
            "average_failure_rate": round(avg_failure, 4),
            "average_disagreement_count": round(avg_disagreement, 4),
            "average_diversity_flagged_count": round(avg_diversity, 4),
            "target_quality": target_quality,
            "target_runtime_sec": target_runtime_sec,
        },
        "rationale": rationale,
        "samples": samples[:20],
    }


def _spawn_report_signals(report: dict[str, Any]) -> dict[str, float]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        runs = []
    completed = [
        item
        for item in runs
        if isinstance(item, dict) and str(item.get("status") or "").strip().lower() == "completed"
    ]
    if not completed:
        return {
            "completed_runs": 0.0,
            "avg_disagreement_count": 0.0,
            "avg_spawn_selected_count": 0.0,
            "avg_spawn_dynamic_count": 0.0,
        }

    disagreement_total = 0.0
    selected_total = 0.0
    dynamic_total = 0.0
    for item in completed:
        telemetry = item.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        disagreement_total += _to_float(telemetry.get("disagreement_count"), 0.0)
        selected_total += _to_float(telemetry.get("spawn_selected_count"), 0.0)
        dynamic_total += _to_float(telemetry.get("spawn_dynamic_count"), 0.0)

    count = float(len(completed))
    return {
        "completed_runs": count,
        "avg_disagreement_count": round(disagreement_total / count, 4),
        "avg_spawn_selected_count": round(selected_total / count, 4),
        "avg_spawn_dynamic_count": round(dynamic_total / count, 4),
    }


def calibrate_spawn_policy(
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
    telemetry = _spawn_report_signals(report)

    quality_gap = float(target_quality) - avg_quality
    quality_pressure = _clamp_float(quality_gap / max(float(target_quality), 0.05), -1.0, 1.0)
    runtime_pressure = 0.0
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        runtime_pressure = (avg_runtime - float(target_runtime_sec)) / float(target_runtime_sec)

    disagreement_pressure = _clamp_float(
        float(telemetry["avg_disagreement_count"]) / 2.0,
        0.0,
        1.0,
    )
    dynamic_pressure = _clamp_float(
        float(telemetry["avg_spawn_dynamic_count"]) / 2.0,
        0.0,
        1.0,
    )
    spawn_pressure = (
        (0.7 * quality_pressure)
        + (0.5 * disagreement_pressure)
        + (0.25 * dynamic_pressure)
        - (0.6 * max(runtime_pressure, 0.0))
        - (0.45 * failure_rate)
    )
    spawn_pressure = _clamp_float(spawn_pressure, -1.0, 1.0)

    base = {
        "RIM_SPAWN_MIN_ROLE_SCORE": 1.0,
        "RIM_SPAWN_MAX_SPECIALISTS_DEEP": 3,
        "RIM_SPAWN_MAX_SPECIALISTS_FAST": 1,
        "RIM_ENABLE_DYNAMIC_SPECIALISTS": 1,
        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": 2,
    }
    min_role_score = round(
        _clamp_float(
            base["RIM_SPAWN_MIN_ROLE_SCORE"] - (0.45 * max(spawn_pressure, 0.0)) + (0.35 * max(-spawn_pressure, 0.0)),
            0.4,
            2.5,
        ),
        3,
    )
    max_specialists_deep = _clamp_int(
        int(round(base["RIM_SPAWN_MAX_SPECIALISTS_DEEP"] + (2.0 * max(spawn_pressure, 0.0)) - (1.0 * max(-spawn_pressure, 0.0)))),
        1,
        8,
    )
    max_specialists_fast = _clamp_int(
        int(round(base["RIM_SPAWN_MAX_SPECIALISTS_FAST"] + (1.0 * max(spawn_pressure, 0.0)))),
        1,
        4,
    )
    enable_dynamic = 1
    if (
        quality_pressure <= 0.0
        and max(runtime_pressure, 0.0) > 0.3
        and disagreement_pressure < 0.2
        and dynamic_pressure < 0.2
    ):
        enable_dynamic = 0
    max_dynamic = _clamp_int(
        int(round(base["RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"] + (2.0 * max(spawn_pressure, 0.0)) - (1.0 * max(runtime_pressure, 0.0)))),
        0,
        6,
    )
    if enable_dynamic == 0:
        max_dynamic = 0

    rationale: list[str] = []
    if quality_pressure > 0.15:
        rationale.append("Average quality is below target; spawn thresholds are relaxed for broader specialist coverage.")
    elif quality_pressure < -0.15:
        rationale.append("Average quality is above target; spawn thresholds can tighten.")
    if disagreement_pressure > 0.25:
        rationale.append("Disagreement pressure indicates value from broader specialist exploration.")
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        if avg_runtime > float(target_runtime_sec):
            rationale.append("Average runtime exceeds target; spawn breadth is moderated.")
        else:
            rationale.append("Average runtime is within target runtime budget.")
    if failure_rate > 0.2:
        rationale.append("Failure rate is elevated; policy avoids aggressive specialist expansion.")

    recommended_env = {
        "RIM_SPAWN_MIN_ROLE_SCORE": min_role_score,
        "RIM_SPAWN_MAX_SPECIALISTS_DEEP": max_specialists_deep,
        "RIM_SPAWN_MAX_SPECIALISTS_FAST": max_specialists_fast,
        "RIM_ENABLE_DYNAMIC_SPECIALISTS": enable_dynamic,
        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": max_dynamic,
    }
    return {
        "inputs": {
            "average_quality_score": avg_quality,
            "average_runtime_sec": avg_runtime,
            "failure_rate": round(failure_rate, 4),
            "dataset_size": dataset_size,
            "target_quality": float(target_quality),
            "target_runtime_sec": target_runtime_sec,
            "telemetry": telemetry,
        },
        "signals": {
            "quality_pressure": round(quality_pressure, 4),
            "runtime_pressure": round(runtime_pressure, 4),
            "disagreement_pressure": round(disagreement_pressure, 4),
            "dynamic_pressure": round(dynamic_pressure, 4),
            "spawn_pressure": round(spawn_pressure, 4),
        },
        "base": base,
        "recommended_env": recommended_env,
        "rationale": rationale,
    }


def train_spawn_policy(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
) -> dict[str, Any]:
    valid_reports: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        if int(report.get("dataset_size", 0)) <= 0:
            continue
        valid_reports.append(report)

    if not valid_reports:
        empty_policy = {
            "RIM_SPAWN_MIN_ROLE_SCORE": 1.0,
            "RIM_SPAWN_MAX_SPECIALISTS_DEEP": 3,
            "RIM_SPAWN_MAX_SPECIALISTS_FAST": 1,
            "RIM_ENABLE_DYNAMIC_SPECIALISTS": 1,
            "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": 2,
        }
        return {
            "report_count": 0,
            "policy_env": empty_policy,
            "rationale": ["No valid reports were available; returning default spawn policy."],
        }

    weighted: dict[str, float] = {
        "RIM_SPAWN_MIN_ROLE_SCORE": 0.0,
        "RIM_SPAWN_MAX_SPECIALISTS_DEEP": 0.0,
        "RIM_SPAWN_MAX_SPECIALISTS_FAST": 0.0,
        "RIM_ENABLE_DYNAMIC_SPECIALISTS": 0.0,
        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": 0.0,
    }
    total_weight = 0.0
    quality_sum = 0.0
    runtime_sum = 0.0
    failure_sum = 0.0
    disagreement_sum = 0.0
    dynamic_sum = 0.0
    samples: list[dict[str, Any]] = []

    for report in valid_reports:
        calibration = calibrate_spawn_policy(
            report,
            target_quality=target_quality,
            target_runtime_sec=target_runtime_sec,
        )
        env = calibration["recommended_env"]
        report_quality = float(report.get("average_quality_score", 0.0))
        report_failure = (
            float(report.get("failure_count", 0)) / max(1, int(report.get("dataset_size", 1)))
        )
        telemetry = calibration.get("inputs", {}).get("telemetry", {})
        avg_disagreement = _to_float(
            telemetry.get("avg_disagreement_count") if isinstance(telemetry, dict) else 0.0,
            0.0,
        )
        avg_dynamic = _to_float(
            telemetry.get("avg_spawn_dynamic_count") if isinstance(telemetry, dict) else 0.0,
            0.0,
        )
        weight = max(0.1, report_quality + 0.15 - (0.25 * report_failure) + (0.05 * avg_disagreement))
        total_weight += weight
        quality_sum += report_quality
        runtime_sum += float(report.get("average_runtime_sec", 0.0))
        failure_sum += report_failure
        disagreement_sum += avg_disagreement
        dynamic_sum += avg_dynamic
        for key in weighted:
            weighted[key] += float(env[key]) * weight
        samples.append(
            {
                "created_at": report.get("created_at"),
                "mode": report.get("mode"),
                "weight": round(weight, 4),
                "recommended_env": env,
            }
        )

    if total_weight <= 0:
        total_weight = float(len(valid_reports))
    policy_env = {
        "RIM_SPAWN_MIN_ROLE_SCORE": round(
            _clamp_float(weighted["RIM_SPAWN_MIN_ROLE_SCORE"] / total_weight, 0.4, 2.5),
            3,
        ),
        "RIM_SPAWN_MAX_SPECIALISTS_DEEP": _clamp_int(
            int(round(weighted["RIM_SPAWN_MAX_SPECIALISTS_DEEP"] / total_weight)),
            1,
            8,
        ),
        "RIM_SPAWN_MAX_SPECIALISTS_FAST": _clamp_int(
            int(round(weighted["RIM_SPAWN_MAX_SPECIALISTS_FAST"] / total_weight)),
            1,
            4,
        ),
        "RIM_ENABLE_DYNAMIC_SPECIALISTS": 1
        if (weighted["RIM_ENABLE_DYNAMIC_SPECIALISTS"] / total_weight) >= 0.5
        else 0,
        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": _clamp_int(
            int(round(weighted["RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"] / total_weight)),
            0,
            6,
        ),
    }
    if policy_env["RIM_ENABLE_DYNAMIC_SPECIALISTS"] == 0:
        policy_env["RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"] = 0
    avg_quality = quality_sum / len(valid_reports)
    avg_runtime = runtime_sum / len(valid_reports)
    avg_failure = failure_sum / len(valid_reports)
    avg_disagreement = disagreement_sum / len(valid_reports)
    avg_dynamic = dynamic_sum / len(valid_reports)
    rationale = [
        "Policy aggregates per-report spawn calibration recommendations using weighted averaging.",
    ]
    if avg_quality < target_quality:
        rationale.append("Average quality is below target, so spawn policy keeps broader specialist coverage.")
    else:
        rationale.append("Average quality meets target, so spawn policy remains balanced.")
    if target_runtime_sec is not None and target_runtime_sec > 0 and avg_runtime > target_runtime_sec:
        rationale.append("Average runtime exceeds target, so spawn breadth is moderated.")
    if avg_failure > 0.2:
        rationale.append("Failure rate is elevated; specialist expansion remains conservative.")
    if avg_disagreement > 0.5 or avg_dynamic > 0.5:
        rationale.append("Disagreement/dynamic-role pressure supports more adaptive specialist spawning.")

    return {
        "report_count": len(valid_reports),
        "policy_env": policy_env,
        "recommended_exports": calibration_env_exports({"recommended_env": policy_env}),
        "summary": {
            "average_quality_score": round(avg_quality, 4),
            "average_runtime_sec": round(avg_runtime, 4),
            "average_failure_rate": round(avg_failure, 4),
            "average_disagreement_count": round(avg_disagreement, 4),
            "average_spawn_dynamic_count": round(avg_dynamic, 4),
            "target_quality": target_quality,
            "target_runtime_sec": target_runtime_sec,
        },
        "rationale": rationale,
        "samples": samples[:20],
    }


def _memory_report_signals(report: dict[str, Any]) -> dict[str, float]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        runs = []
    completed = [
        item
        for item in runs
        if isinstance(item, dict) and str(item.get("status") or "").strip().lower() == "completed"
    ]
    if not completed:
        return {
            "completed_runs": 0.0,
            "total_fold_count": 0.0,
            "total_degradation_count": 0.0,
            "degradation_rate": 0.0,
            "avg_novelty_ratio": 0.0,
            "avg_duplicate_ratio": 0.0,
        }

    fold_count_total = 0.0
    degradation_total = 0.0
    novelty_weighted_total = 0.0
    duplicate_weighted_total = 0.0
    for item in completed:
        telemetry = item.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        run_fold_count = _to_float(telemetry.get("memory_fold_count"), 0.0)
        run_degradation = _to_float(telemetry.get("memory_fold_degradation_count"), 0.0)
        run_novelty = _to_float(telemetry.get("memory_fold_avg_novelty_ratio"), 0.0)
        run_duplicate = _to_float(telemetry.get("memory_fold_avg_duplicate_ratio"), 0.0)
        fold_count_total += run_fold_count
        degradation_total += run_degradation
        novelty_weighted_total += run_novelty * max(run_fold_count, 1.0)
        duplicate_weighted_total += run_duplicate * max(run_fold_count, 1.0)

    effective_folds = max(fold_count_total, 1.0)
    return {
        "completed_runs": float(len(completed)),
        "total_fold_count": fold_count_total,
        "total_degradation_count": degradation_total,
        "degradation_rate": round(degradation_total / effective_folds, 4),
        "avg_novelty_ratio": round(novelty_weighted_total / effective_folds, 4),
        "avg_duplicate_ratio": round(duplicate_weighted_total / effective_folds, 4),
    }


def calibrate_memory_fold_policy(
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
    telemetry = _memory_report_signals(report)

    quality_gap = float(target_quality) - avg_quality
    quality_pressure = _clamp_float(quality_gap / max(float(target_quality), 0.05), -1.0, 1.0)
    runtime_pressure = 0.0
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        runtime_pressure = (avg_runtime - float(target_runtime_sec)) / float(target_runtime_sec)
    degradation_rate = _clamp_float(float(telemetry["degradation_rate"]), 0.0, 1.0)
    novelty_ratio = _clamp_float(float(telemetry["avg_novelty_ratio"]), 0.0, 1.0)
    duplicate_ratio = _clamp_float(float(telemetry["avg_duplicate_ratio"]), 0.0, 1.0)

    memory_pressure = (
        (0.65 * quality_pressure)
        + (0.7 * degradation_rate)
        + (0.3 * max(0.35 - novelty_ratio, 0.0))
        + (0.2 * max(duplicate_ratio - 0.5, 0.0))
        - (0.6 * max(runtime_pressure, 0.0))
        - (0.4 * failure_rate)
    )
    memory_pressure = _clamp_float(memory_pressure, -1.0, 1.0)

    base = {
        "RIM_ENABLE_MEMORY_FOLDING": 1,
        "RIM_MEMORY_FOLD_MAX_ENTRIES": 12,
        "RIM_MEMORY_FOLD_NOVELTY_FLOOR": 0.35,
        "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO": 0.5,
    }
    enable_folding = 1
    if quality_pressure <= 0.0 and max(runtime_pressure, 0.0) > 0.45 and degradation_rate < 0.1:
        enable_folding = 0
    max_entries = _clamp_int(
        int(round(base["RIM_MEMORY_FOLD_MAX_ENTRIES"] + (2.0 * max(memory_pressure, 0.0)) - (3.0 * max(runtime_pressure, 0.0)) - (2.0 * degradation_rate))),
        6,
        40,
    )
    novelty_floor = round(
        _clamp_float(
            base["RIM_MEMORY_FOLD_NOVELTY_FLOOR"] + (0.12 * degradation_rate) + (0.06 * quality_pressure) - (0.05 * max(runtime_pressure, 0.0)),
            0.15,
            0.8,
        ),
        3,
    )
    max_duplicate_ratio = round(
        _clamp_float(
            base["RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO"] - (0.2 * degradation_rate) - (0.08 * quality_pressure) + (0.12 * max(runtime_pressure, 0.0)),
            0.2,
            0.8,
        ),
        3,
    )

    rationale: list[str] = []
    if degradation_rate > 0.2:
        rationale.append("Memory fold degradation rate is elevated; tightening fold quality guardrails.")
    if novelty_ratio < 0.35:
        rationale.append("Novelty ratio is low; policy raises novelty floor expectations.")
    if duplicate_ratio > 0.5:
        rationale.append("Duplicate ratio is high; policy lowers duplicate tolerance.")
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        if avg_runtime > float(target_runtime_sec):
            rationale.append("Average runtime exceeds target; fold budget is moderated.")
        else:
            rationale.append("Average runtime is within target runtime budget.")
    if quality_pressure > 0.15:
        rationale.append("Average quality is below target; memory fold quality controls are strengthened.")

    recommended_env = {
        "RIM_ENABLE_MEMORY_FOLDING": enable_folding,
        "RIM_MEMORY_FOLD_MAX_ENTRIES": max_entries,
        "RIM_MEMORY_FOLD_NOVELTY_FLOOR": novelty_floor,
        "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO": max_duplicate_ratio,
    }
    return {
        "inputs": {
            "average_quality_score": avg_quality,
            "average_runtime_sec": avg_runtime,
            "failure_rate": round(failure_rate, 4),
            "dataset_size": dataset_size,
            "target_quality": float(target_quality),
            "target_runtime_sec": target_runtime_sec,
            "telemetry": telemetry,
        },
        "signals": {
            "quality_pressure": round(quality_pressure, 4),
            "runtime_pressure": round(runtime_pressure, 4),
            "memory_pressure": round(memory_pressure, 4),
            "degradation_rate": round(degradation_rate, 4),
            "novelty_ratio": round(novelty_ratio, 4),
            "duplicate_ratio": round(duplicate_ratio, 4),
        },
        "base": base,
        "recommended_env": recommended_env,
        "rationale": rationale,
    }


def train_memory_policy(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
) -> dict[str, Any]:
    valid_reports: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        if int(report.get("dataset_size", 0)) <= 0:
            continue
        valid_reports.append(report)

    if not valid_reports:
        empty_policy = {
            "RIM_ENABLE_MEMORY_FOLDING": 1,
            "RIM_MEMORY_FOLD_MAX_ENTRIES": 12,
            "RIM_MEMORY_FOLD_NOVELTY_FLOOR": 0.35,
            "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO": 0.5,
        }
        return {
            "report_count": 0,
            "policy_env": empty_policy,
            "rationale": ["No valid reports were available; returning default memory policy."],
        }

    weighted: dict[str, float] = {
        "RIM_ENABLE_MEMORY_FOLDING": 0.0,
        "RIM_MEMORY_FOLD_MAX_ENTRIES": 0.0,
        "RIM_MEMORY_FOLD_NOVELTY_FLOOR": 0.0,
        "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO": 0.0,
    }
    total_weight = 0.0
    quality_sum = 0.0
    runtime_sum = 0.0
    failure_sum = 0.0
    degradation_sum = 0.0
    novelty_sum = 0.0
    duplicate_sum = 0.0
    samples: list[dict[str, Any]] = []

    for report in valid_reports:
        calibration = calibrate_memory_fold_policy(
            report,
            target_quality=target_quality,
            target_runtime_sec=target_runtime_sec,
        )
        env = calibration["recommended_env"]
        report_quality = float(report.get("average_quality_score", 0.0))
        report_failure = (
            float(report.get("failure_count", 0)) / max(1, int(report.get("dataset_size", 1)))
        )
        telemetry = calibration.get("inputs", {}).get("telemetry", {})
        degradation_rate = _to_float(
            telemetry.get("degradation_rate") if isinstance(telemetry, dict) else 0.0,
            0.0,
        )
        novelty_ratio = _to_float(
            telemetry.get("avg_novelty_ratio") if isinstance(telemetry, dict) else 0.0,
            0.0,
        )
        duplicate_ratio = _to_float(
            telemetry.get("avg_duplicate_ratio") if isinstance(telemetry, dict) else 0.0,
            0.0,
        )
        weight = max(0.1, report_quality + 0.15 - (0.25 * report_failure) + (0.1 * degradation_rate))
        total_weight += weight
        quality_sum += report_quality
        runtime_sum += float(report.get("average_runtime_sec", 0.0))
        failure_sum += report_failure
        degradation_sum += degradation_rate
        novelty_sum += novelty_ratio
        duplicate_sum += duplicate_ratio
        for key in weighted:
            weighted[key] += float(env[key]) * weight
        samples.append(
            {
                "created_at": report.get("created_at"),
                "mode": report.get("mode"),
                "weight": round(weight, 4),
                "recommended_env": env,
            }
        )

    if total_weight <= 0:
        total_weight = float(len(valid_reports))
    policy_env = {
        "RIM_ENABLE_MEMORY_FOLDING": 1
        if (weighted["RIM_ENABLE_MEMORY_FOLDING"] / total_weight) >= 0.5
        else 0,
        "RIM_MEMORY_FOLD_MAX_ENTRIES": _clamp_int(
            int(round(weighted["RIM_MEMORY_FOLD_MAX_ENTRIES"] / total_weight)),
            6,
            40,
        ),
        "RIM_MEMORY_FOLD_NOVELTY_FLOOR": round(
            _clamp_float(weighted["RIM_MEMORY_FOLD_NOVELTY_FLOOR"] / total_weight, 0.15, 0.8),
            3,
        ),
        "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO": round(
            _clamp_float(weighted["RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO"] / total_weight, 0.2, 0.8),
            3,
        ),
    }
    avg_quality = quality_sum / len(valid_reports)
    avg_runtime = runtime_sum / len(valid_reports)
    avg_failure = failure_sum / len(valid_reports)
    avg_degradation = degradation_sum / len(valid_reports)
    avg_novelty = novelty_sum / len(valid_reports)
    avg_duplicate = duplicate_sum / len(valid_reports)
    rationale = [
        "Policy aggregates per-report memory calibration recommendations using weighted averaging.",
    ]
    if avg_degradation > 0.2:
        rationale.append("Observed memory degradation is elevated, so quality guardrails are tightened.")
    if avg_novelty < 0.35:
        rationale.append("Observed novelty ratio is low, so novelty floor remains strict.")
    if target_runtime_sec is not None and target_runtime_sec > 0 and avg_runtime > target_runtime_sec:
        rationale.append("Average runtime exceeds target, so memory fold budget is moderated.")
    if avg_quality < target_quality:
        rationale.append("Average quality is below target, so memory fold controls remain active.")

    return {
        "report_count": len(valid_reports),
        "policy_env": policy_env,
        "recommended_exports": calibration_env_exports({"recommended_env": policy_env}),
        "summary": {
            "average_quality_score": round(avg_quality, 4),
            "average_runtime_sec": round(avg_runtime, 4),
            "average_failure_rate": round(avg_failure, 4),
            "average_memory_degradation_rate": round(avg_degradation, 4),
            "average_memory_novelty_ratio": round(avg_novelty, 4),
            "average_memory_duplicate_ratio": round(avg_duplicate, 4),
            "target_quality": target_quality,
            "target_runtime_sec": target_runtime_sec,
        },
        "rationale": rationale,
        "samples": samples[:20],
    }


def train_depth_policy(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
) -> dict[str, Any]:
    valid_reports: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        if int(report.get("dataset_size", 0)) <= 0:
            continue
        valid_reports.append(report)

    if not valid_reports:
        empty_policy = {
            "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": 0.78,
            "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": 2,
            "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": 1,
            "RIM_MAX_ANALYSIS_CYCLES": 1,
        }
        return {
            "report_count": 0,
            "policy_env": empty_policy,
            "rationale": ["No valid reports were available; returning default policy."],
        }

    weighted: dict[str, float] = {
        "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": 0.0,
        "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": 0.0,
        "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": 0.0,
        "RIM_MAX_ANALYSIS_CYCLES": 0.0,
    }
    total_weight = 0.0
    quality_sum = 0.0
    runtime_sum = 0.0
    failure_sum = 0.0
    samples: list[dict[str, Any]] = []
    for report in valid_reports:
        calibration = calibrate_depth_allocator(
            report,
            target_quality=target_quality,
            target_runtime_sec=target_runtime_sec,
        )
        env = calibration["recommended_env"]
        report_quality = float(report.get("average_quality_score", 0.0))
        report_failure = (
            float(report.get("failure_count", 0)) / max(1, int(report.get("dataset_size", 1)))
        )
        weight = max(0.1, report_quality + 0.15 - (0.25 * report_failure))
        total_weight += weight
        quality_sum += report_quality
        runtime_sum += float(report.get("average_runtime_sec", 0.0))
        failure_sum += report_failure
        for key in weighted:
            weighted[key] += float(env[key]) * weight
        samples.append(
            {
                "created_at": report.get("created_at"),
                "mode": report.get("mode"),
                "weight": round(weight, 4),
                "recommended_env": env,
            }
        )

    if total_weight <= 0:
        total_weight = float(len(valid_reports))
    policy_env = {
        "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": round(
            _clamp_float(weighted["RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE"] / total_weight, 0.65, 0.93),
            3,
        ),
        "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": _clamp_int(
            int(round(weighted["RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS"] / total_weight)),
            0,
            4,
        ),
        "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": _clamp_int(
            int(round(weighted["RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS"] / total_weight)),
            0,
            3,
        ),
        "RIM_MAX_ANALYSIS_CYCLES": _clamp_int(
            int(round(weighted["RIM_MAX_ANALYSIS_CYCLES"] / total_weight)),
            1,
            4,
        ),
    }
    avg_quality = quality_sum / len(valid_reports)
    avg_runtime = runtime_sum / len(valid_reports)
    avg_failure = failure_sum / len(valid_reports)
    rationale = [
        "Policy aggregates per-report calibration recommendations using quality-weighted averaging.",
    ]
    if avg_quality < target_quality:
        rationale.append("Average quality is below target, so policy leans deeper.")
    else:
        rationale.append("Average quality meets target, so policy remains balanced.")
    if target_runtime_sec is not None and target_runtime_sec > 0 and avg_runtime > target_runtime_sec:
        rationale.append("Average runtime exceeds target, so depth recommendations are moderated.")
    if avg_failure > 0.2:
        rationale.append("Failure rate is elevated; depth expansion is dampened for stability.")

    return {
        "report_count": len(valid_reports),
        "policy_env": policy_env,
        "recommended_exports": calibration_env_exports({"recommended_env": policy_env}),
        "summary": {
            "average_quality_score": round(avg_quality, 4),
            "average_runtime_sec": round(avg_runtime, 4),
            "average_failure_rate": round(avg_failure, 4),
            "target_quality": target_quality,
            "target_runtime_sec": target_runtime_sec,
        },
        "rationale": rationale,
        "samples": samples[:20],
    }
