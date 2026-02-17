from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, AnalyzeResult, DecompositionNode
from rim.eval.benchmark import evaluate_run
from rim.providers.router import ProviderRouter

DEFAULT_DATASET_PATH = Path("rim/eval/data/benchmark_ideas.jsonl")
DEFAULT_REPORTS_DIR = Path("rim/eval/reports")
DEFAULT_POLICIES_DIR = Path("rim/eval/policies")

SINGLE_CALL_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "synthesized_idea": {"type": "string"},
        "changes_summary": {"type": "array", "items": {"type": "string"}},
        "residual_risks": {"type": "array", "items": {"type": "string"}},
        "next_experiments": {"type": "array", "items": {"type": "string"}},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "synthesized_idea",
        "changes_summary",
        "residual_risks",
        "next_experiments",
        "confidence_score",
    ],
}

SINGLE_CALL_LLM_PROMPT = """You are a deep-thinking assistant.
Solve this in one pass (no recursion or tool orchestration), then return STRICT JSON only:
{{
  "synthesized_idea": "string",
  "changes_summary": ["string"],
  "residual_risks": ["string"],
  "next_experiments": ["string"],
  "confidence_score": 0.0
}}

Domain:
{domain}

Idea:
{idea}

Constraints:
{constraints}

Desired outcome:
{desired_outcome}

Requirements:
- Think deeply and cover tradeoffs.
- Keep recommendations concrete and testable.
- Include major residual risks honestly.
- Output only one valid JSON object.
"""


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


def _clean_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned[:limit]


def _normalize_single_call_payload(payload: dict[str, Any], idea: str) -> dict[str, Any]:
    synthesized_idea = str(payload.get("synthesized_idea", "")).strip() or idea
    confidence = _to_float(payload.get("confidence_score"), 0.5)
    confidence = max(0.0, min(1.0, confidence))
    return {
        "synthesized_idea": synthesized_idea,
        "changes_summary": _clean_list(payload.get("changes_summary"), limit=8),
        "residual_risks": _clean_list(payload.get("residual_risks"), limit=8),
        "next_experiments": _clean_list(payload.get("next_experiments"), limit=8),
        "confidence_score": confidence,
    }


def _single_call_constraints(constraints: list[str] | None) -> str:
    values = [str(item).strip() for item in list(constraints or []) if str(item).strip()]
    if not values:
        return "none"
    return "\n".join(f"- {item}" for item in values[:10])


def _single_call_baseline_prompt(request: AnalyzeRequest) -> str:
    return SINGLE_CALL_LLM_PROMPT.format(
        domain=request.domain or "general",
        idea=request.idea,
        constraints=_single_call_constraints(request.constraints),
        desired_outcome=request.desired_outcome or "none",
    )


def _normalize_llm_provider(provider: str) -> str:
    parsed = str(provider or "").strip().lower()
    if parsed not in {"claude", "codex"}:
        raise ValueError("provider must be one of: claude, codex")
    return parsed


async def _invoke_single_call_llm(
    *,
    router: Any,
    run_id: str,
    provider: str,
    prompt: str,
) -> tuple[dict[str, Any], str]:
    stage = f"baseline_llm_{provider}"
    session_factory = getattr(router, "create_session", None)
    if callable(session_factory):
        session = session_factory(run_id)
        stage_policy = dict(getattr(session, "stage_policy", {}) or {})
        stage_policy[stage] = [provider]
        setattr(session, "stage_policy", stage_policy)
        payload, selected_provider = await session.invoke_json(
            stage,
            prompt,
            json_schema=SINGLE_CALL_LLM_SCHEMA,
        )
    else:
        payload, selected_provider = await router.invoke_json(
            stage,
            prompt,
            json_schema=SINGLE_CALL_LLM_SCHEMA,
        )
    if not isinstance(payload, dict):
        raise ValueError("single-call baseline returned non-object JSON payload")
    return payload, str(selected_provider or provider)


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


def _normalize_role(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_tools(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    tools: list[str] = []
    for item in value:
        parsed = str(item or "").strip()
        if not parsed or parsed in seen:
            continue
        seen.add(parsed)
        tools.append(parsed)
    return tools


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
        "arbitration_jobs_requested_config": 0,
        "arbitration_max_jobs_config": 0,
        "disagreement_arbitration_enabled_config": False,
        "devils_advocate_count": 0,
        "devils_advocate_enabled_config": False,
        "devils_advocate_rounds_config": 0,
        "devils_advocate_min_confidence_config": 0.0,
        "specialist_count": 0,
        "specialist_loop_enabled_config": False,
        "specialist_max_jobs_config": 0,
        "specialist_min_confidence_config": 0.0,
        "spawn_selected_count": 0,
        "spawn_dynamic_count": 0,
        "spawn_selected_roles": [],
        "spawn_dynamic_roles": [],
        "spawn_role_routing": {},
        "spawn_role_tools": {},
        "spawn_min_role_score_config": 0.0,
        "spawn_max_specialists_config": 0,
        "spawn_max_dynamic_specialists_config": 0,
        "spawn_dynamic_enabled_config": False,
        "spawn_policy_applied_config": False,
        "spawn_role_boosts_config": {},
        "spawn_dynamic_token_boosts_config": {},
        "depth_cycles_observed": 0,
        "depth_max_cycles_config": 0,
        "depth_min_confidence_config": 0.0,
        "depth_max_residual_risks_config": 0,
        "depth_max_high_findings_config": 0,
        "memory_fold_count": 0,
        "memory_fold_degradation_count": 0,
        "memory_fold_avg_novelty_ratio": 0.0,
        "memory_fold_avg_duplicate_ratio": 0.0,
    }
    memory_fold_novelty_total = 0.0
    memory_fold_duplicate_total = 0.0
    spawn_selected_roles: set[str] = set()
    spawn_dynamic_roles: set[str] = set()
    spawn_role_routing: dict[str, str] = {}
    spawn_role_tools: dict[str, list[str]] = {}
    spawn_role_boosts: dict[str, float] = {}
    spawn_dynamic_token_boosts: dict[str, float] = {}
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
            telemetry["arbitration_jobs_requested_config"] = max(
                telemetry["arbitration_jobs_requested_config"],
                _to_int(meta.get("jobs_requested"), 0),
            )
            telemetry["arbitration_max_jobs_config"] = max(
                telemetry["arbitration_max_jobs_config"],
                _to_int(meta.get("arbitration_max_jobs"), 0),
            )
            telemetry["disagreement_arbitration_enabled_config"] = (
                telemetry["disagreement_arbitration_enabled_config"]
                or bool(meta.get("disagreement_arbitration_enabled", False))
            )
            telemetry["devils_advocate_count"] = max(
                telemetry["devils_advocate_count"],
                _to_int(meta.get("devils_advocate_count"), 0),
            )
            telemetry["devils_advocate_enabled_config"] = (
                telemetry["devils_advocate_enabled_config"]
                or bool(meta.get("devils_advocate_enabled", False))
            )
            telemetry["devils_advocate_rounds_config"] = max(
                telemetry["devils_advocate_rounds_config"],
                _to_int(meta.get("devils_advocate_rounds"), 0),
            )
            telemetry["devils_advocate_min_confidence_config"] = max(
                telemetry["devils_advocate_min_confidence_config"],
                _to_float(meta.get("devils_advocate_min_confidence"), 0.0),
            )
            telemetry["specialist_count"] = max(
                telemetry["specialist_count"],
                _to_int(meta.get("specialist_count"), 0),
            )
            telemetry["specialist_loop_enabled_config"] = (
                telemetry["specialist_loop_enabled_config"]
                or bool(meta.get("specialist_loop_enabled", False))
            )
            telemetry["specialist_max_jobs_config"] = max(
                telemetry["specialist_max_jobs_config"],
                _to_int(meta.get("specialist_max_jobs"), 0),
            )
            telemetry["specialist_min_confidence_config"] = max(
                telemetry["specialist_min_confidence_config"],
                _to_float(meta.get("specialist_min_confidence"), 0.0),
            )
            continue
        if stage == "depth_allocator":
            telemetry["depth_cycles_observed"] = max(
                telemetry["depth_cycles_observed"],
                _to_int(meta.get("cycle"), 0),
            )
            telemetry["depth_max_cycles_config"] = max(
                telemetry["depth_max_cycles_config"],
                _to_int(meta.get("max_cycles"), 0),
            )
            telemetry["depth_min_confidence_config"] = max(
                telemetry["depth_min_confidence_config"],
                _to_float(meta.get("min_confidence_to_stop"), 0.0),
            )
            telemetry["depth_max_residual_risks_config"] = max(
                telemetry["depth_max_residual_risks_config"],
                _to_int(meta.get("max_residual_risks_to_stop"), 0),
            )
            telemetry["depth_max_high_findings_config"] = max(
                telemetry["depth_max_high_findings_config"],
                _to_int(meta.get("max_high_findings_to_stop"), 0),
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
            telemetry["spawn_min_role_score_config"] = max(
                telemetry["spawn_min_role_score_config"],
                _to_float(meta.get("min_role_score"), 0.0),
            )
            telemetry["spawn_max_specialists_config"] = max(
                telemetry["spawn_max_specialists_config"],
                _to_int(meta.get("max_specialists"), 0),
            )
            telemetry["spawn_max_dynamic_specialists_config"] = max(
                telemetry["spawn_max_dynamic_specialists_config"],
                _to_int(meta.get("max_dynamic_specialists"), 0),
            )
            telemetry["spawn_dynamic_enabled_config"] = (
                telemetry["spawn_dynamic_enabled_config"]
                or bool(meta.get("dynamic_enabled", False))
            )
            telemetry["spawn_policy_applied_config"] = (
                telemetry["spawn_policy_applied_config"]
                or bool(meta.get("policy_applied", False))
            )
            role_boosts_raw = meta.get("role_boosts")
            if isinstance(role_boosts_raw, dict):
                for role_name, boost in role_boosts_raw.items():
                    normalized_role = _normalize_role(role_name)
                    if not normalized_role:
                        continue
                    spawn_role_boosts[normalized_role] = max(
                        spawn_role_boosts.get(normalized_role, 0.0),
                        round(_to_float(boost, 0.0), 4),
                    )
            dynamic_boosts_raw = meta.get("dynamic_token_boosts")
            if isinstance(dynamic_boosts_raw, dict):
                for token_name, boost in dynamic_boosts_raw.items():
                    normalized_token = _normalize_role(token_name)
                    if not normalized_token:
                        continue
                    spawn_dynamic_token_boosts[normalized_token] = max(
                        spawn_dynamic_token_boosts.get(normalized_token, 0.0),
                        round(_to_float(boost, 0.0), 4),
                    )
            for entry in extra:
                if not isinstance(entry, dict):
                    continue
                role_name = _normalize_role(entry.get("role"))
                if not role_name:
                    continue
                spawn_selected_roles.add(role_name)
                if bool(entry.get("dynamic")):
                    spawn_dynamic_roles.add(role_name)
                tool_contract = entry.get("tool_contract")
                if isinstance(tool_contract, dict):
                    routing_policy = str(tool_contract.get("routing_policy") or "").strip()
                    if routing_policy:
                        spawn_role_routing[role_name] = routing_policy
                    tools = _normalize_tools(tool_contract.get("tools"))
                    if tools:
                        spawn_role_tools[role_name] = tools
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
    telemetry["spawn_selected_roles"] = sorted(spawn_selected_roles)
    telemetry["spawn_dynamic_roles"] = sorted(spawn_dynamic_roles)
    telemetry["spawn_role_routing"] = {
        key: spawn_role_routing[key] for key in sorted(spawn_role_routing)
    }
    telemetry["spawn_role_tools"] = {
        key: spawn_role_tools[key] for key in sorted(spawn_role_tools)
    }
    telemetry["spawn_role_boosts_config"] = {
        key: spawn_role_boosts[key] for key in sorted(spawn_role_boosts)
    }
    telemetry["spawn_dynamic_token_boosts_config"] = {
        key: spawn_dynamic_token_boosts[key]
        for key in sorted(spawn_dynamic_token_boosts)
    }
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


async def run_single_call_llm_baseline(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    provider: str = "claude",
    mode: str = "deep",
    limit: int | None = None,
    router: Any | None = None,
) -> dict[str, Any]:
    selected_provider = _normalize_llm_provider(provider)
    selected_mode = "deep" if str(mode).strip().lower() == "deep" else "fast"
    active_router: Any = router if router is not None else ProviderRouter()

    started_at = datetime.now(timezone.utc).isoformat()
    dataset = load_dataset(dataset_path)
    if limit is not None and limit > 0:
        dataset = dataset[:limit]

    runs: list[dict[str, Any]] = []
    started = time.perf_counter()

    for item in dataset:
        request = AnalyzeRequest(
            idea=str(item["idea"]),
            mode=selected_mode,
            domain=item.get("domain"),
            constraints=item.get("constraints") or [],
            desired_outcome=item.get("desired_outcome"),
        )
        run_started = time.perf_counter()
        item_id = str(item.get("id") or f"row-{len(runs) + 1}")
        run_id = f"baseline-llm-{selected_provider}-{item_id}"
        prompt = _single_call_baseline_prompt(request)
        try:
            raw_payload, used_provider = await _invoke_single_call_llm(
                router=active_router,
                run_id=run_id,
                provider=selected_provider,
                prompt=prompt,
            )
            synthesis = _normalize_single_call_payload(raw_payload, request.idea)
            result = AnalyzeResult(
                run_id=run_id,
                mode=request.mode,
                input_idea=request.idea,
                decomposition=[
                    DecompositionNode(
                        depth=0,
                        component_text=request.idea,
                        node_type="claim",
                        confidence=0.6,
                    )
                ],
                critic_findings=[],
                synthesized_idea=str(synthesis["synthesized_idea"]),
                changes_summary=list(synthesis["changes_summary"]),
                residual_risks=list(synthesis["residual_risks"]),
                next_experiments=list(synthesis["next_experiments"]),
                confidence_score=float(synthesis["confidence_score"]),
            )
            runtime_sec = round(time.perf_counter() - run_started, 3)
            score = evaluate_run(result, heuristic_reviewer, domain=request.domain)
            runs.append(
                {
                    "id": item.get("id"),
                    "idea": request.idea,
                    "domain": request.domain,
                    "mode": f"single_call_{selected_provider}",
                    "runtime_sec": runtime_sec,
                    "run_id": result.run_id,
                    "status": "completed",
                    "provider": used_provider,
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
                    "mode": f"single_call_{selected_provider}",
                    "runtime_sec": runtime_sec,
                    "status": "failed",
                    "provider": selected_provider,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    total_runtime_sec = round(time.perf_counter() - started, 3)
    report = _summarize_report(
        started_at=started_at,
        dataset_path=dataset_path,
        mode=f"single_call_{selected_provider}",
        total_runtime_sec=total_runtime_sec,
        runs=runs,
    )
    report["provider"] = selected_provider
    report["baseline_type"] = "single_call_llm"
    return report


async def run_duel_benchmark(
    orchestrator: RimOrchestrator,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mode: str = "deep",
    limit: int | None = None,
    min_quality_delta: float = 0.0,
    max_runtime_delta_sec: float | None = None,
    min_shared_runs: int = 1,
    baseline_provider: str = "proxy",
) -> dict[str, Any]:
    parsed_baseline_provider = str(baseline_provider or "proxy").strip().lower()
    if parsed_baseline_provider not in {"proxy", "claude", "codex"}:
        raise ValueError("baseline_provider must be one of: proxy, claude, codex")
    if parsed_baseline_provider in {"claude", "codex"}:
        baseline = await run_single_call_llm_baseline(
            dataset_path=dataset_path,
            provider=parsed_baseline_provider,
            mode=mode,
            limit=limit,
            router=getattr(orchestrator, "router", None),
        )
    else:
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
        "baseline_provider": parsed_baseline_provider,
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
        elif isinstance(value, (dict, list)):
            serialized = json.dumps(value, separators=(",", ":"), sort_keys=True)
            lines.append(f"export {key}='{serialized}'")
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


def _arbitration_report_signals(report: dict[str, Any]) -> dict[str, float]:
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
            "avg_arbitration_resolved_count": 0.0,
            "avg_devils_advocate_count": 0.0,
            "avg_specialist_count": 0.0,
        }

    disagreement_total = 0.0
    arbitration_resolved_total = 0.0
    devils_advocate_total = 0.0
    specialist_total = 0.0
    for item in completed:
        telemetry = item.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        disagreement_total += _to_float(telemetry.get("disagreement_count"), 0.0)
        arbitration_resolved_total += _to_float(
            telemetry.get("arbitration_resolved_count"),
            0.0,
        )
        devils_advocate_total += _to_float(telemetry.get("devils_advocate_count"), 0.0)
        specialist_total += _to_float(telemetry.get("specialist_count"), 0.0)

    count = float(len(completed))
    return {
        "completed_runs": count,
        "avg_disagreement_count": round(disagreement_total / count, 4),
        "avg_arbitration_resolved_count": round(arbitration_resolved_total / count, 4),
        "avg_devils_advocate_count": round(devils_advocate_total / count, 4),
        "avg_specialist_count": round(specialist_total / count, 4),
    }


def calibrate_arbitration_policy(
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
    telemetry = _arbitration_report_signals(report)

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
    resolved_ratio = float(telemetry["avg_arbitration_resolved_count"]) / max(
        float(telemetry["avg_disagreement_count"]),
        0.5,
    )
    resolution_pressure = _clamp_float(0.55 - resolved_ratio, -1.0, 1.0)
    devils_pressure = _clamp_float(
        float(telemetry["avg_devils_advocate_count"])
        / max(float(telemetry["avg_arbitration_resolved_count"]), 1.0),
        0.0,
        1.0,
    )
    specialist_pressure = _clamp_float(
        float(telemetry["avg_specialist_count"]) / 2.0,
        0.0,
        1.0,
    )

    arbitration_pressure = (
        (0.65 * quality_pressure)
        + (0.7 * disagreement_pressure)
        + (0.45 * resolution_pressure)
        + (0.25 * devils_pressure)
        + (0.2 * specialist_pressure)
        - (0.6 * max(runtime_pressure, 0.0))
        - (0.45 * failure_rate)
    )
    arbitration_pressure = _clamp_float(arbitration_pressure, -1.0, 1.0)

    base = {
        "RIM_ENABLE_DISAGREEMENT_ARBITRATION": 1,
        "RIM_ARBITRATION_MAX_JOBS": 2,
        "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": 1,
        "RIM_DEVILS_ADVOCATE_ROUNDS": 1,
        "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": 0.72,
    }

    enable_arbitration = 1
    if (
        arbitration_pressure < -0.4
        and max(runtime_pressure, 0.0) > 0.2
        and disagreement_pressure < 0.2
    ):
        enable_arbitration = 0

    max_jobs = _clamp_int(
        int(
            round(
                base["RIM_ARBITRATION_MAX_JOBS"]
                + (2.0 * max(arbitration_pressure, 0.0))
                + (1.0 * disagreement_pressure)
                - (1.0 * max(-arbitration_pressure, 0.0))
            )
        ),
        0,
        6,
    )

    enable_devils_advocate = 1
    if enable_arbitration == 0:
        enable_devils_advocate = 0
    elif (
        quality_pressure <= 0.0
        and max(runtime_pressure, 0.0) > 0.25
        and disagreement_pressure < 0.25
    ):
        enable_devils_advocate = 0

    devils_rounds = _clamp_int(
        int(
            round(
                base["RIM_DEVILS_ADVOCATE_ROUNDS"]
                + (1.0 * max(arbitration_pressure, 0.0))
                + (0.5 * devils_pressure)
                - (1.0 * max(runtime_pressure, 0.0))
            )
        ),
        0,
        3,
    )
    devils_min_confidence = round(
        _clamp_float(
            base["RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE"]
            - (0.1 * max(arbitration_pressure, 0.0))
            + (0.08 * max(runtime_pressure, 0.0))
            + (0.05 * max(-quality_pressure, 0.0)),
            0.55,
            0.95,
        ),
        3,
    )

    if enable_arbitration == 0:
        max_jobs = 0
        enable_devils_advocate = 0
        devils_rounds = 0
    if enable_devils_advocate == 0:
        devils_rounds = 0

    rationale: list[str] = []
    if quality_pressure > 0.15:
        rationale.append("Average quality is below target; disagreement arbitration is strengthened.")
    elif quality_pressure < -0.15:
        rationale.append("Average quality is above target; arbitration can be moderated.")
    if disagreement_pressure > 0.3:
        rationale.append("Disagreement pressure suggests keeping arbitration coverage broad.")
    if resolution_pressure > 0.2:
        rationale.append("Observed arbitration resolution appears low; policy expands arbitration capacity.")
    if target_runtime_sec is not None and float(target_runtime_sec) > 0:
        if avg_runtime > float(target_runtime_sec):
            rationale.append("Average runtime exceeds target; arbitration depth is moderated.")
        else:
            rationale.append("Average runtime is within target runtime budget.")
    if failure_rate > 0.2:
        rationale.append("Failure rate is elevated; policy avoids aggressive arbitration expansion.")

    recommended_env = {
        "RIM_ENABLE_DISAGREEMENT_ARBITRATION": enable_arbitration,
        "RIM_ARBITRATION_MAX_JOBS": max_jobs,
        "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": enable_devils_advocate,
        "RIM_DEVILS_ADVOCATE_ROUNDS": devils_rounds,
        "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": devils_min_confidence,
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
            "resolution_pressure": round(resolution_pressure, 4),
            "devils_pressure": round(devils_pressure, 4),
            "specialist_pressure": round(specialist_pressure, 4),
            "arbitration_pressure": round(arbitration_pressure, 4),
        },
        "base": base,
        "recommended_env": recommended_env,
        "rationale": rationale,
    }


def train_arbitration_policy(
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
            "RIM_ENABLE_DISAGREEMENT_ARBITRATION": 1,
            "RIM_ARBITRATION_MAX_JOBS": 2,
            "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": 1,
            "RIM_DEVILS_ADVOCATE_ROUNDS": 1,
            "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": 0.72,
        }
        return {
            "report_count": 0,
            "policy_env": empty_policy,
            "rationale": ["No valid reports were available; returning default arbitration policy."],
        }

    weighted: dict[str, float] = {
        "RIM_ENABLE_DISAGREEMENT_ARBITRATION": 0.0,
        "RIM_ARBITRATION_MAX_JOBS": 0.0,
        "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": 0.0,
        "RIM_DEVILS_ADVOCATE_ROUNDS": 0.0,
        "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": 0.0,
    }
    total_weight = 0.0
    quality_sum = 0.0
    runtime_sum = 0.0
    failure_sum = 0.0
    disagreement_sum = 0.0
    resolved_sum = 0.0
    devils_sum = 0.0
    samples: list[dict[str, Any]] = []

    for report in valid_reports:
        calibration = calibrate_arbitration_policy(
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
        avg_resolved = _to_float(
            telemetry.get("avg_arbitration_resolved_count")
            if isinstance(telemetry, dict)
            else 0.0,
            0.0,
        )
        avg_devils = _to_float(
            telemetry.get("avg_devils_advocate_count")
            if isinstance(telemetry, dict)
            else 0.0,
            0.0,
        )
        weight = max(0.1, report_quality + 0.15 - (0.25 * report_failure) + (0.05 * avg_disagreement))
        total_weight += weight
        quality_sum += report_quality
        runtime_sum += float(report.get("average_runtime_sec", 0.0))
        failure_sum += report_failure
        disagreement_sum += avg_disagreement
        resolved_sum += avg_resolved
        devils_sum += avg_devils
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
    enable_arbitration = 1 if (weighted["RIM_ENABLE_DISAGREEMENT_ARBITRATION"] / total_weight) >= 0.5 else 0
    enable_devils = 1 if (weighted["RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION"] / total_weight) >= 0.5 else 0
    max_jobs = _clamp_int(
        int(round(weighted["RIM_ARBITRATION_MAX_JOBS"] / total_weight)),
        0,
        6,
    )
    devils_rounds = _clamp_int(
        int(round(weighted["RIM_DEVILS_ADVOCATE_ROUNDS"] / total_weight)),
        0,
        3,
    )
    if enable_arbitration == 0:
        max_jobs = 0
        enable_devils = 0
        devils_rounds = 0
    if enable_devils == 0:
        devils_rounds = 0
    policy_env = {
        "RIM_ENABLE_DISAGREEMENT_ARBITRATION": enable_arbitration,
        "RIM_ARBITRATION_MAX_JOBS": max_jobs,
        "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": enable_devils,
        "RIM_DEVILS_ADVOCATE_ROUNDS": devils_rounds,
        "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": round(
            _clamp_float(weighted["RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE"] / total_weight, 0.55, 0.95),
            3,
        ),
    }
    avg_quality = quality_sum / len(valid_reports)
    avg_runtime = runtime_sum / len(valid_reports)
    avg_failure = failure_sum / len(valid_reports)
    avg_disagreement = disagreement_sum / len(valid_reports)
    avg_resolved = resolved_sum / len(valid_reports)
    avg_devils = devils_sum / len(valid_reports)
    rationale = [
        "Policy aggregates arbitration-calibration recommendations using weighted averaging.",
    ]
    if avg_quality < target_quality:
        rationale.append("Average quality is below target, so arbitration coverage remains active.")
    else:
        rationale.append("Average quality meets target, so arbitration policy remains balanced.")
    if target_runtime_sec is not None and target_runtime_sec > 0 and avg_runtime > target_runtime_sec:
        rationale.append("Average runtime exceeds target, so arbitration volume is moderated.")
    if avg_failure > 0.2:
        rationale.append("Failure rate is elevated; policy avoids aggressive arbitration expansion.")
    if avg_disagreement > 0.5 or avg_resolved < 0.4:
        rationale.append("Disagreement and low-resolution signals support stronger arbitration policy defaults.")

    return {
        "report_count": len(valid_reports),
        "policy_env": policy_env,
        "recommended_exports": calibration_env_exports({"recommended_env": policy_env}),
        "summary": {
            "average_quality_score": round(avg_quality, 4),
            "average_runtime_sec": round(avg_runtime, 4),
            "average_failure_rate": round(avg_failure, 4),
            "average_disagreement_count": round(avg_disagreement, 4),
            "average_arbitration_resolved_count": round(avg_resolved, 4),
            "average_devils_advocate_count": round(avg_devils, 4),
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


def _extract_policy_env(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    direct = {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and key.startswith("RIM_")
    }
    if direct:
        return direct
    for key in ("policy_env", "recommended_env", "policy", "calibration"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            extracted = _extract_policy_env(nested)
            if extracted:
                return extracted
    return {}


def load_policy_env(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return _extract_policy_env(payload)


def save_policy_artifact(
    policy: dict[str, Any],
    *,
    policy_kind: str,
    source_reports: list[str],
    output_path: Path | None = None,
    learning_meta: dict[str, Any] | None = None,
) -> Path:
    env = policy.get("policy_env")
    if not isinstance(env, dict):
        raise ValueError("policy payload must include `policy_env`.")
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "policy_kind": policy_kind,
        "policy_env": env,
        "recommended_exports": calibration_env_exports({"recommended_env": env}),
        "source_reports": source_reports,
        "learning_meta": learning_meta or {},
        "policy": policy,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    DEFAULT_POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    auto_path = DEFAULT_POLICIES_DIR / f"{policy_kind}_policy_{stamp}.json"
    auto_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return auto_path


def _blend_depth_policy_env(
    *,
    prior_env: dict[str, Any] | None,
    fresh_env: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    prior = prior_env or {}
    alpha = _clamp_float(float(learning_rate), 0.0, 1.0)
    min_conf = (1.0 - alpha) * _to_float(prior.get("RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE"), 0.78)
    min_conf += alpha * _to_float(fresh_env.get("RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE"), 0.78)
    max_risks = (1.0 - alpha) * _to_float(prior.get("RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS"), 2.0)
    max_risks += alpha * _to_float(fresh_env.get("RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS"), 2.0)
    max_high = (1.0 - alpha) * _to_float(prior.get("RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS"), 1.0)
    max_high += alpha * _to_float(fresh_env.get("RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS"), 1.0)
    max_cycles = (1.0 - alpha) * _to_float(prior.get("RIM_MAX_ANALYSIS_CYCLES"), 1.0)
    max_cycles += alpha * _to_float(fresh_env.get("RIM_MAX_ANALYSIS_CYCLES"), 1.0)
    return {
        "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": round(_clamp_float(min_conf, 0.65, 0.93), 3),
        "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": _clamp_int(int(round(max_risks)), 0, 4),
        "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": _clamp_int(int(round(max_high)), 0, 3),
        "RIM_MAX_ANALYSIS_CYCLES": _clamp_int(int(round(max_cycles)), 1, 4),
    }


def _blend_specialist_policy_env(
    *,
    prior_env: dict[str, Any] | None,
    fresh_env: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    prior = prior_env or {}
    alpha = _clamp_float(float(learning_rate), 0.0, 1.0)
    enable_prior = _to_float(prior.get("RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP"), 1.0)
    enable_fresh = _to_float(fresh_env.get("RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP"), 1.0)
    enable_score = ((1.0 - alpha) * enable_prior) + (alpha * enable_fresh)
    max_jobs = (1.0 - alpha) * _to_float(prior.get("RIM_SPECIALIST_ARBITRATION_MAX_JOBS"), 2.0)
    max_jobs += alpha * _to_float(fresh_env.get("RIM_SPECIALIST_ARBITRATION_MAX_JOBS"), 2.0)
    min_conf = (1.0 - alpha) * _to_float(prior.get("RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE"), 0.78)
    min_conf += alpha * _to_float(fresh_env.get("RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE"), 0.78)
    enable_loop = 1 if enable_score >= 0.5 else 0
    normalized_jobs = _clamp_int(int(round(max_jobs)), 0, 6)
    if enable_loop == 0:
        normalized_jobs = 0
    return {
        "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP": enable_loop,
        "RIM_SPECIALIST_ARBITRATION_MAX_JOBS": normalized_jobs,
        "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE": round(
            _clamp_float(min_conf, 0.6, 0.95),
            3,
        ),
    }


def _blend_arbitration_policy_env(
    *,
    prior_env: dict[str, Any] | None,
    fresh_env: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    prior = prior_env or {}
    alpha = _clamp_float(float(learning_rate), 0.0, 1.0)
    enable_score = (1.0 - alpha) * _to_float(prior.get("RIM_ENABLE_DISAGREEMENT_ARBITRATION"), 1.0)
    enable_score += alpha * _to_float(fresh_env.get("RIM_ENABLE_DISAGREEMENT_ARBITRATION"), 1.0)
    max_jobs = (1.0 - alpha) * _to_float(prior.get("RIM_ARBITRATION_MAX_JOBS"), 2.0)
    max_jobs += alpha * _to_float(fresh_env.get("RIM_ARBITRATION_MAX_JOBS"), 2.0)
    devils_enable_score = (1.0 - alpha) * _to_float(
        prior.get("RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION"),
        1.0,
    )
    devils_enable_score += alpha * _to_float(
        fresh_env.get("RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION"),
        1.0,
    )
    devils_rounds = (1.0 - alpha) * _to_float(prior.get("RIM_DEVILS_ADVOCATE_ROUNDS"), 1.0)
    devils_rounds += alpha * _to_float(fresh_env.get("RIM_DEVILS_ADVOCATE_ROUNDS"), 1.0)
    devils_min_conf = (1.0 - alpha) * _to_float(
        prior.get("RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE"),
        0.72,
    )
    devils_min_conf += alpha * _to_float(
        fresh_env.get("RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE"),
        0.72,
    )

    enable_arbitration = 1 if enable_score >= 0.5 else 0
    enable_devils = 1 if devils_enable_score >= 0.5 else 0
    normalized_jobs = _clamp_int(int(round(max_jobs)), 0, 6)
    normalized_rounds = _clamp_int(int(round(devils_rounds)), 0, 3)
    if enable_arbitration == 0:
        normalized_jobs = 0
        enable_devils = 0
        normalized_rounds = 0
    if enable_devils == 0:
        normalized_rounds = 0
    return {
        "RIM_ENABLE_DISAGREEMENT_ARBITRATION": enable_arbitration,
        "RIM_ARBITRATION_MAX_JOBS": normalized_jobs,
        "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": enable_devils,
        "RIM_DEVILS_ADVOCATE_ROUNDS": normalized_rounds,
        "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": round(
            _clamp_float(devils_min_conf, 0.55, 0.95),
            3,
        ),
    }


def _blend_memory_policy_env(
    *,
    prior_env: dict[str, Any] | None,
    fresh_env: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    prior = prior_env or {}
    alpha = _clamp_float(float(learning_rate), 0.0, 1.0)
    enable_prior = _to_float(prior.get("RIM_ENABLE_MEMORY_FOLDING"), 1.0)
    enable_fresh = _to_float(fresh_env.get("RIM_ENABLE_MEMORY_FOLDING"), 1.0)
    enable_score = ((1.0 - alpha) * enable_prior) + (alpha * enable_fresh)
    max_entries = (1.0 - alpha) * _to_float(prior.get("RIM_MEMORY_FOLD_MAX_ENTRIES"), 12.0)
    max_entries += alpha * _to_float(fresh_env.get("RIM_MEMORY_FOLD_MAX_ENTRIES"), 12.0)
    novelty_floor = (1.0 - alpha) * _to_float(prior.get("RIM_MEMORY_FOLD_NOVELTY_FLOOR"), 0.35)
    novelty_floor += alpha * _to_float(fresh_env.get("RIM_MEMORY_FOLD_NOVELTY_FLOOR"), 0.35)
    max_duplicate_ratio = (1.0 - alpha) * _to_float(
        prior.get("RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO"),
        0.5,
    )
    max_duplicate_ratio += alpha * _to_float(
        fresh_env.get("RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO"),
        0.5,
    )
    return {
        "RIM_ENABLE_MEMORY_FOLDING": 1 if enable_score >= 0.5 else 0,
        "RIM_MEMORY_FOLD_MAX_ENTRIES": _clamp_int(int(round(max_entries)), 6, 40),
        "RIM_MEMORY_FOLD_NOVELTY_FLOOR": round(_clamp_float(novelty_floor, 0.15, 0.8), 3),
        "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO": round(
            _clamp_float(max_duplicate_ratio, 0.2, 0.8),
            3,
        ),
    }


def _normalize_float_map(
    value: Any,
    *,
    lower: float = -4.0,
    upper: float = 4.0,
) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, item in value.items():
        role = _normalize_role(key)
        if not role:
            continue
        normalized[role] = round(_clamp_float(_to_float(item, 0.0), lower, upper), 4)
    return normalized


def _normalize_string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        role = _normalize_role(key)
        policy = str(item or "").strip()
        if role and policy:
            normalized[role] = policy
    return normalized


def _normalize_tool_map(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, item in value.items():
        role = _normalize_role(key)
        tools = _normalize_tools(item)
        if role and tools:
            normalized[role] = tools
    return normalized


def _blend_spawn_policy_env(
    *,
    prior_env: dict[str, Any] | None,
    fresh_env: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    prior = prior_env or {}
    alpha = _clamp_float(float(learning_rate), 0.0, 1.0)
    min_role_score = (1.0 - alpha) * _to_float(prior.get("RIM_SPAWN_MIN_ROLE_SCORE"), 1.0)
    min_role_score += alpha * _to_float(fresh_env.get("RIM_SPAWN_MIN_ROLE_SCORE"), 1.0)
    max_specialists_deep = (1.0 - alpha) * _to_float(
        prior.get("RIM_SPAWN_MAX_SPECIALISTS_DEEP"),
        3.0,
    )
    max_specialists_deep += alpha * _to_float(
        fresh_env.get("RIM_SPAWN_MAX_SPECIALISTS_DEEP"),
        3.0,
    )
    max_specialists_fast = (1.0 - alpha) * _to_float(
        prior.get("RIM_SPAWN_MAX_SPECIALISTS_FAST"),
        1.0,
    )
    max_specialists_fast += alpha * _to_float(
        fresh_env.get("RIM_SPAWN_MAX_SPECIALISTS_FAST"),
        1.0,
    )
    enable_dynamic_score = (1.0 - alpha) * _to_float(
        prior.get("RIM_ENABLE_DYNAMIC_SPECIALISTS"),
        1.0,
    )
    enable_dynamic_score += alpha * _to_float(
        fresh_env.get("RIM_ENABLE_DYNAMIC_SPECIALISTS"),
        1.0,
    )
    max_dynamic_specialists = (1.0 - alpha) * _to_float(
        prior.get("RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"),
        2.0,
    )
    max_dynamic_specialists += alpha * _to_float(
        fresh_env.get("RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"),
        2.0,
    )
    role_boosts_prior = _normalize_float_map(prior.get("RIM_SPAWN_ROLE_BOOSTS"))
    role_boosts_fresh = _normalize_float_map(fresh_env.get("RIM_SPAWN_ROLE_BOOSTS"))
    role_boosts: dict[str, float] = {}
    for key in sorted(set(role_boosts_prior) | set(role_boosts_fresh)):
        blended = ((1.0 - alpha) * role_boosts_prior.get(key, 0.0)) + (
            alpha * role_boosts_fresh.get(key, 0.0)
        )
        if abs(blended) >= 0.05:
            role_boosts[key] = round(_clamp_float(blended, -4.0, 4.0), 4)
    dynamic_boosts_prior = _normalize_float_map(prior.get("RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS"))
    dynamic_boosts_fresh = _normalize_float_map(
        fresh_env.get("RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS")
    )
    dynamic_boosts: dict[str, float] = {}
    for key in sorted(set(dynamic_boosts_prior) | set(dynamic_boosts_fresh)):
        blended = ((1.0 - alpha) * dynamic_boosts_prior.get(key, 0.0)) + (
            alpha * dynamic_boosts_fresh.get(key, 0.0)
        )
        if abs(blended) >= 0.05:
            dynamic_boosts[key] = round(_clamp_float(blended, -4.0, 4.0), 4)
    routing_prior = _normalize_string_map(prior.get("RIM_SPAWN_ROLE_ROUTING_OVERRIDES"))
    routing_fresh = _normalize_string_map(fresh_env.get("RIM_SPAWN_ROLE_ROUTING_OVERRIDES"))
    tool_prior = _normalize_tool_map(prior.get("RIM_SPAWN_ROLE_TOOL_OVERRIDES"))
    tool_fresh = _normalize_tool_map(fresh_env.get("RIM_SPAWN_ROLE_TOOL_OVERRIDES"))

    blended_env: dict[str, Any] = {
        "RIM_SPAWN_MIN_ROLE_SCORE": round(_clamp_float(min_role_score, 0.4, 3.0), 3),
        "RIM_SPAWN_MAX_SPECIALISTS_DEEP": _clamp_int(
            int(round(max_specialists_deep)),
            1,
            8,
        ),
        "RIM_SPAWN_MAX_SPECIALISTS_FAST": _clamp_int(
            int(round(max_specialists_fast)),
            1,
            4,
        ),
        "RIM_ENABLE_DYNAMIC_SPECIALISTS": 1 if enable_dynamic_score >= 0.5 else 0,
        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": _clamp_int(
            int(round(max_dynamic_specialists)),
            0,
            6,
        ),
    }
    if blended_env["RIM_ENABLE_DYNAMIC_SPECIALISTS"] == 0:
        blended_env["RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"] = 0
    if role_boosts:
        blended_env["RIM_SPAWN_ROLE_BOOSTS"] = role_boosts
    if dynamic_boosts:
        blended_env["RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS"] = dynamic_boosts
    combined_routing = {**routing_prior, **routing_fresh}
    if combined_routing:
        blended_env["RIM_SPAWN_ROLE_ROUTING_OVERRIDES"] = combined_routing
    combined_tools = {**tool_prior, **tool_fresh}
    if combined_tools:
        blended_env["RIM_SPAWN_ROLE_TOOL_OVERRIDES"] = combined_tools
    return blended_env


def _valid_training_reports(
    reports: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    valid: list[dict[str, Any]] = []
    report_ids: list[str] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        if int(report.get("dataset_size", 0)) <= 0:
            continue
        valid.append(report)
        report_ids.append(
            str(report.get("created_at") or f"report-{len(report_ids) + 1}")
        )
    return valid, report_ids


def train_online_depth_and_arbitration_policies(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
    learning_rate: float = 0.35,
    prior_depth_policy_env: dict[str, Any] | None = None,
    prior_specialist_policy_env: dict[str, Any] | None = None,
    prior_arbitration_policy_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_reports, report_ids = _valid_training_reports(reports)
    depth_candidate = train_depth_policy(
        valid_reports,
        target_quality=target_quality,
        target_runtime_sec=target_runtime_sec,
    )
    specialist_candidate = train_specialist_arbitration_policy(
        valid_reports,
        target_quality=target_quality,
        target_runtime_sec=target_runtime_sec,
    )
    arbitration_candidate = train_arbitration_policy(
        valid_reports,
        target_quality=target_quality,
        target_runtime_sec=target_runtime_sec,
    )
    alpha = _clamp_float(float(learning_rate), 0.0, 1.0)
    depth_env = _blend_depth_policy_env(
        prior_env=prior_depth_policy_env,
        fresh_env=dict(depth_candidate.get("policy_env") or {}),
        learning_rate=alpha,
    )
    specialist_env = _blend_specialist_policy_env(
        prior_env=prior_specialist_policy_env,
        fresh_env=dict(specialist_candidate.get("policy_env") or {}),
        learning_rate=alpha,
    )
    arbitration_env = _blend_arbitration_policy_env(
        prior_env=prior_arbitration_policy_env,
        fresh_env=dict(arbitration_candidate.get("policy_env") or {}),
        learning_rate=alpha,
    )
    depth_payload = {
        **depth_candidate,
        "policy_env": depth_env,
        "recommended_exports": calibration_env_exports({"recommended_env": depth_env}),
        "blend": {
            "learning_rate": alpha,
            "prior_policy_env": prior_depth_policy_env or {},
            "candidate_policy_env": depth_candidate.get("policy_env") or {},
        },
    }
    specialist_payload = {
        **specialist_candidate,
        "policy_env": specialist_env,
        "recommended_exports": calibration_env_exports({"recommended_env": specialist_env}),
        "blend": {
            "learning_rate": alpha,
            "prior_policy_env": prior_specialist_policy_env or {},
            "candidate_policy_env": specialist_candidate.get("policy_env") or {},
        },
    }
    arbitration_payload = {
        **arbitration_candidate,
        "policy_env": arbitration_env,
        "recommended_exports": calibration_env_exports({"recommended_env": arbitration_env}),
        "blend": {
            "learning_rate": alpha,
            "prior_policy_env": prior_arbitration_policy_env or {},
            "candidate_policy_env": arbitration_candidate.get("policy_env") or {},
        },
    }
    combined_exports = sorted(
        set(
            list(depth_payload["recommended_exports"])
            + list(specialist_payload["recommended_exports"])
            + list(arbitration_payload["recommended_exports"])
        )
    )
    return {
        "report_count": len(valid_reports),
        "report_ids": report_ids,
        "learning_rate": alpha,
        "target_quality": target_quality,
        "target_runtime_sec": target_runtime_sec,
        "depth_policy": depth_payload,
        "specialist_policy": specialist_payload,
        "arbitration_policy": arbitration_payload,
        "recommended_exports": combined_exports,
    }


def train_online_memory_policy(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
    learning_rate: float = 0.35,
    prior_memory_policy_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_reports, report_ids = _valid_training_reports(reports)
    candidate = train_memory_policy(
        valid_reports,
        target_quality=target_quality,
        target_runtime_sec=target_runtime_sec,
    )
    alpha = _clamp_float(float(learning_rate), 0.0, 1.0)
    memory_env = _blend_memory_policy_env(
        prior_env=prior_memory_policy_env,
        fresh_env=dict(candidate.get("policy_env") or {}),
        learning_rate=alpha,
    )
    payload = {
        **candidate,
        "policy_env": memory_env,
        "recommended_exports": calibration_env_exports({"recommended_env": memory_env}),
        "blend": {
            "learning_rate": alpha,
            "prior_policy_env": prior_memory_policy_env or {},
            "candidate_policy_env": candidate.get("policy_env") or {},
        },
    }
    return {
        "report_count": len(valid_reports),
        "report_ids": report_ids,
        "learning_rate": alpha,
        "target_quality": target_quality,
        "target_runtime_sec": target_runtime_sec,
        "memory_policy": payload,
        "recommended_exports": payload["recommended_exports"],
    }


def train_online_spawn_policy(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
    learning_rate: float = 0.35,
    prior_spawn_policy_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_reports, report_ids = _valid_training_reports(reports)
    candidate = train_spawn_policy(
        valid_reports,
        target_quality=target_quality,
        target_runtime_sec=target_runtime_sec,
    )
    alpha = _clamp_float(float(learning_rate), 0.0, 1.0)
    spawn_env = _blend_spawn_policy_env(
        prior_env=prior_spawn_policy_env,
        fresh_env=dict(candidate.get("policy_env") or {}),
        learning_rate=alpha,
    )
    payload = {
        **candidate,
        "policy_env": spawn_env,
        "recommended_exports": calibration_env_exports({"recommended_env": spawn_env}),
        "blend": {
            "learning_rate": alpha,
            "prior_policy_env": prior_spawn_policy_env or {},
            "candidate_policy_env": candidate.get("policy_env") or {},
        },
    }
    return {
        "report_count": len(valid_reports),
        "report_ids": report_ids,
        "learning_rate": alpha,
        "target_quality": target_quality,
        "target_runtime_sec": target_runtime_sec,
        "spawn_policy": payload,
        "recommended_exports": payload["recommended_exports"],
    }


def _rl_experiences(
    reports: list[dict[str, Any]],
    *,
    target_quality: float,
    target_runtime_sec: float | None,
    runtime_weight: float,
    failure_penalty: float,
) -> list[dict[str, Any]]:
    experiences: list[dict[str, Any]] = []
    runtime_weight_normalized = _clamp_float(float(runtime_weight), 0.0, 2.0)
    failure_penalty_normalized = _clamp_float(float(failure_penalty), 0.0, 4.0)
    for report in reports:
        runs = report.get("runs")
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            status = str(run.get("status") or "").strip().lower()
            quality_payload = run.get("quality")
            quality_score = (
                float(quality_payload.get("quality_score"))
                if isinstance(quality_payload, dict)
                and isinstance(quality_payload.get("quality_score"), (int, float))
                else 0.0
            )
            runtime_sec = _to_float(run.get("runtime_sec"), 0.0)
            runtime_pressure = 0.0
            if target_runtime_sec is not None and float(target_runtime_sec) > 0:
                runtime_pressure = max(
                    0.0,
                    (runtime_sec - float(target_runtime_sec)) / float(target_runtime_sec),
                )
            quality_gap = float(target_quality) - quality_score
            telemetry = run.get("telemetry")
            telemetry_dict = telemetry if isinstance(telemetry, dict) else {}
            disagreement = _to_float(telemetry_dict.get("disagreement_count"), 0.0)
            diversity = _to_float(telemetry_dict.get("diversity_flagged_count"), 0.0)
            depth_cycles = _to_float(telemetry_dict.get("depth_max_cycles_config"), 1.0)
            depth_min_conf = _to_float(telemetry_dict.get("depth_min_confidence_config"), 0.78)
            depth_max_risks = _to_float(telemetry_dict.get("depth_max_residual_risks_config"), 2.0)
            depth_max_high = _to_float(telemetry_dict.get("depth_max_high_findings_config"), 1.0)
            specialist_jobs = _to_float(telemetry_dict.get("specialist_max_jobs_config"), 2.0)
            specialist_min_conf = _to_float(
                telemetry_dict.get("specialist_min_confidence_config"),
                0.78,
            )
            specialist_enabled = _to_float(
                1.0 if telemetry_dict.get("specialist_loop_enabled_config") else 0.0,
                0.0,
            )
            arbitration_jobs = _to_float(
                telemetry_dict.get("arbitration_max_jobs_config"),
                _to_float(telemetry_dict.get("arbitration_jobs_requested_config"), 2.0),
            )
            devils_advocate_enabled = _to_float(
                1.0 if telemetry_dict.get("devils_advocate_enabled_config") else 0.0,
                0.0,
            )
            devils_advocate_rounds = _to_float(
                telemetry_dict.get("devils_advocate_rounds_config"),
                1.0,
            )
            devils_advocate_min_conf = _to_float(
                telemetry_dict.get("devils_advocate_min_confidence_config"),
                0.72,
            )
            disagreement_arbitration_enabled = _to_float(
                1.0 if telemetry_dict.get("disagreement_arbitration_enabled_config") else 0.0,
                0.0,
            )
            if depth_cycles <= 0:
                depth_cycles = 1.0
            if depth_min_conf <= 0:
                depth_min_conf = 0.78
            if depth_max_risks < 0:
                depth_max_risks = 2.0
            if specialist_jobs < 0:
                specialist_jobs = 2.0
            if specialist_min_conf <= 0:
                specialist_min_conf = 0.78
            if arbitration_jobs < 0:
                arbitration_jobs = 2.0
            if devils_advocate_rounds < 0:
                devils_advocate_rounds = 0.0
            if devils_advocate_min_conf <= 0:
                devils_advocate_min_conf = 0.72

            reward = quality_score - (runtime_weight_normalized * runtime_pressure)
            if status in {"failed", "canceled"}:
                reward -= failure_penalty_normalized
            if status == "partial":
                reward -= (0.5 * failure_penalty_normalized)
            experiences.append(
                {
                    "run_id": str(run.get("run_id") or run.get("id") or f"run-{len(experiences)+1}"),
                    "status": status,
                    "quality_score": quality_score,
                    "runtime_sec": runtime_sec,
                    "runtime_pressure": runtime_pressure,
                    "quality_gap": quality_gap,
                    "disagreement": disagreement,
                    "diversity": diversity,
                    "reward": reward,
                    "actions": {
                        "depth_cycles": depth_cycles,
                        "depth_min_conf": depth_min_conf,
                        "depth_max_risks": depth_max_risks,
                        "depth_max_high": depth_max_high,
                        "specialist_jobs": specialist_jobs,
                        "specialist_min_conf": specialist_min_conf,
                        "specialist_enabled": specialist_enabled,
                        "arbitration_jobs": arbitration_jobs,
                        "disagreement_arbitration_enabled": disagreement_arbitration_enabled,
                        "devils_advocate_enabled": devils_advocate_enabled,
                        "devils_advocate_rounds": devils_advocate_rounds,
                        "devils_advocate_min_conf": devils_advocate_min_conf,
                    },
                }
            )
    return experiences


def train_rl_depth_and_arbitration_policies(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
    learning_rate: float = 0.18,
    epochs: int = 3,
    reward_runtime_weight: float = 0.35,
    reward_failure_penalty: float = 1.0,
    prior_depth_policy_env: dict[str, Any] | None = None,
    prior_specialist_policy_env: dict[str, Any] | None = None,
    prior_arbitration_policy_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_reports, report_ids = _valid_training_reports(reports)
    experiences = _rl_experiences(
        valid_reports,
        target_quality=target_quality,
        target_runtime_sec=target_runtime_sec,
        runtime_weight=reward_runtime_weight,
        failure_penalty=reward_failure_penalty,
    )
    alpha = _clamp_float(float(learning_rate), 0.01, 1.0)
    epoch_count = _clamp_int(int(epochs), 1, 50)
    runtime_weight_normalized = _clamp_float(float(reward_runtime_weight), 0.0, 2.0)
    failure_penalty_normalized = _clamp_float(float(reward_failure_penalty), 0.0, 4.0)

    depth = {
        "min_conf": _to_float((prior_depth_policy_env or {}).get("RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE"), 0.78),
        "max_risks": _to_float((prior_depth_policy_env or {}).get("RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS"), 2.0),
        "max_high": _to_float((prior_depth_policy_env or {}).get("RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS"), 1.0),
        "max_cycles": _to_float((prior_depth_policy_env or {}).get("RIM_MAX_ANALYSIS_CYCLES"), 1.0),
    }
    specialist = {
        "enabled_score": _to_float(
            (prior_specialist_policy_env or {}).get("RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP"),
            1.0,
        ),
        "jobs": _to_float(
            (prior_specialist_policy_env or {}).get("RIM_SPECIALIST_ARBITRATION_MAX_JOBS"),
            2.0,
        ),
        "min_conf": _to_float(
            (prior_specialist_policy_env or {}).get("RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE"),
            0.78,
        ),
    }
    arbitration = {
        "enabled_score": _to_float(
            (prior_arbitration_policy_env or {}).get("RIM_ENABLE_DISAGREEMENT_ARBITRATION"),
            1.0,
        ),
        "max_jobs": _to_float(
            (prior_arbitration_policy_env or {}).get("RIM_ARBITRATION_MAX_JOBS"),
            2.0,
        ),
        "devils_enabled_score": _to_float(
            (prior_arbitration_policy_env or {}).get(
                "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION"
            ),
            1.0,
        ),
        "devils_rounds": _to_float(
            (prior_arbitration_policy_env or {}).get("RIM_DEVILS_ADVOCATE_ROUNDS"),
            1.0,
        ),
        "devils_min_conf": _to_float(
            (prior_arbitration_policy_env or {}).get("RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE"),
            0.72,
        ),
    }
    if not experiences:
        depth_env = _blend_depth_policy_env(
            prior_env=prior_depth_policy_env,
            fresh_env={},
            learning_rate=0.0,
        )
        specialist_env = _blend_specialist_policy_env(
            prior_env=prior_specialist_policy_env,
            fresh_env={},
            learning_rate=0.0,
        )
        arbitration_env = _blend_arbitration_policy_env(
            prior_env=prior_arbitration_policy_env,
            fresh_env={},
            learning_rate=0.0,
        )
        return {
            "optimizer": "rl_credit_assignment_v1",
            "report_count": len(valid_reports),
            "report_ids": report_ids,
            "experience_count": 0,
            "learning_rate": alpha,
            "epochs": epoch_count,
            "target_quality": target_quality,
            "target_runtime_sec": target_runtime_sec,
            "depth_policy": {
                "policy_env": depth_env,
                "recommended_exports": calibration_env_exports({"recommended_env": depth_env}),
                "rationale": ["No eligible run experiences; returning prior/default depth policy."],
            },
            "specialist_policy": {
                "policy_env": specialist_env,
                "recommended_exports": calibration_env_exports({"recommended_env": specialist_env}),
                "rationale": ["No eligible run experiences; returning prior/default specialist policy."],
            },
            "arbitration_policy": {
                "policy_env": arbitration_env,
                "recommended_exports": calibration_env_exports({"recommended_env": arbitration_env}),
                "rationale": ["No eligible run experiences; returning prior/default arbitration policy."],
            },
            "credit_assignment": {
                "depth": {"positive": 0.0, "negative": 0.0, "top_runs": []},
                "specialist": {"positive": 0.0, "negative": 0.0, "top_runs": []},
                "arbitration": {"positive": 0.0, "negative": 0.0, "top_runs": []},
            },
            "recommended_exports": sorted(
                set(
                    calibration_env_exports({"recommended_env": depth_env})
                    + calibration_env_exports({"recommended_env": specialist_env})
                    + calibration_env_exports({"recommended_env": arbitration_env})
                )
            ),
        }

    rewards = [float(item["reward"]) for item in experiences]
    reward_baseline = sum(rewards) / len(rewards)
    depth_credits: list[dict[str, Any]] = []
    specialist_credits: list[dict[str, Any]] = []
    arbitration_credits: list[dict[str, Any]] = []

    for _epoch in range(epoch_count):
        for exp in experiences:
            reward = float(exp["reward"])
            advantage = reward - reward_baseline
            runtime_pressure = _to_float(exp.get("runtime_pressure"), 0.0)
            quality_gap = _to_float(exp.get("quality_gap"), 0.0)
            disagreement = _to_float(exp.get("disagreement"), 0.0)
            diversity = _to_float(exp.get("diversity"), 0.0)
            actions = exp.get("actions")
            action_payload = actions if isinstance(actions, dict) else {}
            depth_cycles_action = _to_float(action_payload.get("depth_cycles"), 1.0)
            depth_min_conf_action = _to_float(action_payload.get("depth_min_conf"), 0.78)
            specialist_jobs_action = _to_float(action_payload.get("specialist_jobs"), 2.0)
            specialist_enabled_action = _to_float(action_payload.get("specialist_enabled"), 1.0)
            arbitration_jobs_action = _to_float(action_payload.get("arbitration_jobs"), 2.0)
            disagreement_arbitration_enabled_action = _to_float(
                action_payload.get("disagreement_arbitration_enabled"),
                1.0,
            )
            devils_advocate_enabled_action = _to_float(
                action_payload.get("devils_advocate_enabled"),
                1.0,
            )
            devils_advocate_rounds_action = _to_float(
                action_payload.get("devils_advocate_rounds"),
                1.0,
            )
            devils_advocate_min_conf_action = _to_float(
                action_payload.get("devils_advocate_min_conf"),
                0.72,
            )

            depth_signal = quality_gap + (0.25 * disagreement) - (0.70 * runtime_pressure)
            specialist_signal = (0.9 * disagreement) + (0.5 * diversity) - (0.6 * runtime_pressure)
            arbitration_signal = (1.0 * disagreement) + (0.35 * diversity) + (0.4 * quality_gap) - (
                0.75 * runtime_pressure
            )
            depth_action_intensity = max(
                0.0,
                (0.35 * max(depth_cycles_action - 1.0, 0.0))
                + (0.25 * max(depth_min_conf_action - 0.78, 0.0)),
            )
            specialist_action_intensity = max(
                0.0,
                (0.45 * specialist_enabled_action)
                + (0.2 * max(specialist_jobs_action - 1.0, 0.0)),
            )
            arbitration_action_intensity = max(
                0.0,
                (0.4 * disagreement_arbitration_enabled_action)
                + (0.2 * max(arbitration_jobs_action - 1.0, 0.0))
                + (0.2 * devils_advocate_enabled_action)
                + (0.16 * max(devils_advocate_rounds_action, 0.0))
                + (0.1 * max(0.75 - devils_advocate_min_conf_action, 0.0)),
            )
            depth_credit = advantage * depth_signal * (1.0 + (0.2 * depth_action_intensity))
            specialist_credit = (
                advantage
                * specialist_signal
                * (1.0 + (0.2 * specialist_action_intensity))
            )
            arbitration_credit = (
                advantage
                * arbitration_signal
                * (1.0 + (0.2 * arbitration_action_intensity))
            )

            depth["max_cycles"] += alpha * depth_credit
            depth["min_conf"] += alpha * (0.12 * depth_credit)
            depth["max_risks"] += alpha * (-0.28 * depth_credit)
            depth["max_high"] += alpha * (-0.20 * depth_credit)

            specialist["jobs"] += alpha * (0.45 * specialist_credit)
            specialist["min_conf"] += alpha * (0.08 * specialist_credit)
            specialist["enabled_score"] += alpha * (0.35 * specialist_credit)

            arbitration["max_jobs"] += alpha * (0.52 * arbitration_credit)
            arbitration["enabled_score"] += alpha * (0.34 * arbitration_credit)
            arbitration["devils_enabled_score"] += alpha * (0.36 * arbitration_credit)
            arbitration["devils_rounds"] += alpha * (0.35 * arbitration_credit)
            arbitration["devils_min_conf"] += alpha * (-0.08 * arbitration_credit)

            depth["max_cycles"] = _clamp_float(depth["max_cycles"], 1.0, 4.0)
            depth["min_conf"] = _clamp_float(depth["min_conf"], 0.65, 0.93)
            depth["max_risks"] = _clamp_float(depth["max_risks"], 0.0, 4.0)
            depth["max_high"] = _clamp_float(depth["max_high"], 0.0, 3.0)

            specialist["jobs"] = _clamp_float(specialist["jobs"], 0.0, 6.0)
            specialist["min_conf"] = _clamp_float(specialist["min_conf"], 0.6, 0.95)
            specialist["enabled_score"] = _clamp_float(specialist["enabled_score"], 0.0, 1.0)
            arbitration["max_jobs"] = _clamp_float(arbitration["max_jobs"], 0.0, 6.0)
            arbitration["enabled_score"] = _clamp_float(arbitration["enabled_score"], 0.0, 1.0)
            arbitration["devils_enabled_score"] = _clamp_float(
                arbitration["devils_enabled_score"],
                0.0,
                1.0,
            )
            arbitration["devils_rounds"] = _clamp_float(arbitration["devils_rounds"], 0.0, 3.0)
            arbitration["devils_min_conf"] = _clamp_float(arbitration["devils_min_conf"], 0.55, 0.95)

            depth_credits.append(
                {
                    "run_id": exp["run_id"],
                    "credit": round(depth_credit, 6),
                    "advantage": round(advantage, 6),
                    "signal": round(depth_signal, 6),
                    "action_intensity": round(depth_action_intensity, 6),
                }
            )
            specialist_credits.append(
                {
                    "run_id": exp["run_id"],
                    "credit": round(specialist_credit, 6),
                    "advantage": round(advantage, 6),
                    "signal": round(specialist_signal, 6),
                    "action_intensity": round(specialist_action_intensity, 6),
                }
            )
            arbitration_credits.append(
                {
                    "run_id": exp["run_id"],
                    "credit": round(arbitration_credit, 6),
                    "advantage": round(advantage, 6),
                    "signal": round(arbitration_signal, 6),
                    "action_intensity": round(arbitration_action_intensity, 6),
                }
            )

    depth_env = {
        "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": round(depth["min_conf"], 3),
        "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": _clamp_int(int(round(depth["max_risks"])), 0, 4),
        "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": _clamp_int(int(round(depth["max_high"])), 0, 3),
        "RIM_MAX_ANALYSIS_CYCLES": _clamp_int(int(round(depth["max_cycles"])), 1, 4),
    }
    specialist_enable = 1 if specialist["enabled_score"] >= 0.5 else 0
    specialist_jobs = _clamp_int(int(round(specialist["jobs"])), 0, 6)
    if specialist_enable == 0:
        specialist_jobs = 0
    specialist_env = {
        "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP": specialist_enable,
        "RIM_SPECIALIST_ARBITRATION_MAX_JOBS": specialist_jobs,
        "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE": round(specialist["min_conf"], 3),
    }
    arbitration_enable = 1 if arbitration["enabled_score"] >= 0.5 else 0
    devils_enable = 1 if arbitration["devils_enabled_score"] >= 0.5 else 0
    arbitration_jobs = _clamp_int(int(round(arbitration["max_jobs"])), 0, 6)
    devils_rounds = _clamp_int(int(round(arbitration["devils_rounds"])), 0, 3)
    if arbitration_enable == 0:
        arbitration_jobs = 0
        devils_enable = 0
        devils_rounds = 0
    if devils_enable == 0:
        devils_rounds = 0
    arbitration_env = {
        "RIM_ENABLE_DISAGREEMENT_ARBITRATION": arbitration_enable,
        "RIM_ARBITRATION_MAX_JOBS": arbitration_jobs,
        "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": devils_enable,
        "RIM_DEVILS_ADVOCATE_ROUNDS": devils_rounds,
        "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": round(arbitration["devils_min_conf"], 3),
    }

    def _credit_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        positive = sum(float(item["credit"]) for item in items if float(item["credit"]) > 0.0)
        negative = sum(float(item["credit"]) for item in items if float(item["credit"]) < 0.0)
        ranked = sorted(items, key=lambda item: abs(float(item["credit"])), reverse=True)[:8]
        return {
            "positive": round(positive, 6),
            "negative": round(negative, 6),
            "top_runs": ranked,
        }

    depth_payload = {
        "policy_env": depth_env,
        "recommended_exports": calibration_env_exports({"recommended_env": depth_env}),
        "rationale": [
            "Depth policy updated with reward/advantage credit assignment from recent runs.",
        ],
        "optimizer": "rl_credit_assignment_v1",
        "epochs": epoch_count,
    }
    specialist_payload = {
        "policy_env": specialist_env,
        "recommended_exports": calibration_env_exports({"recommended_env": specialist_env}),
        "rationale": [
            "Specialist arbitration policy updated with disagreement-weighted reward credits.",
        ],
        "optimizer": "rl_credit_assignment_v1",
        "epochs": epoch_count,
    }
    arbitration_payload = {
        "policy_env": arbitration_env,
        "recommended_exports": calibration_env_exports({"recommended_env": arbitration_env}),
        "rationale": [
            "Arbitration policy updated with disagreement/runtime reward credits.",
        ],
        "optimizer": "rl_credit_assignment_v1",
        "epochs": epoch_count,
    }
    return {
        "optimizer": "rl_credit_assignment_v1",
        "report_count": len(valid_reports),
        "report_ids": report_ids,
        "experience_count": len(experiences),
        "learning_rate": alpha,
        "epochs": epoch_count,
        "target_quality": target_quality,
        "target_runtime_sec": target_runtime_sec,
        "reward_summary": {
            "mean_reward": round(reward_baseline, 6),
            "max_reward": round(max(rewards), 6),
            "min_reward": round(min(rewards), 6),
            "runtime_weight": runtime_weight_normalized,
            "failure_penalty": failure_penalty_normalized,
        },
        "depth_policy": depth_payload,
        "specialist_policy": specialist_payload,
        "arbitration_policy": arbitration_payload,
        "credit_assignment": {
            "depth": _credit_summary(depth_credits),
            "specialist": _credit_summary(specialist_credits),
            "arbitration": _credit_summary(arbitration_credits),
        },
        "recommended_exports": sorted(
            set(
                depth_payload["recommended_exports"]
                + specialist_payload["recommended_exports"]
                + arbitration_payload["recommended_exports"]
            )
        ),
    }


def _rl_spawn_experiences(
    reports: list[dict[str, Any]],
    *,
    target_quality: float,
    target_runtime_sec: float | None,
    runtime_weight: float,
    failure_penalty: float,
) -> list[dict[str, Any]]:
    experiences: list[dict[str, Any]] = []
    runtime_weight_normalized = _clamp_float(float(runtime_weight), 0.0, 2.0)
    failure_penalty_normalized = _clamp_float(float(failure_penalty), 0.0, 4.0)
    for report in reports:
        runs = report.get("runs")
        if not isinstance(runs, list):
            continue
        mode_default = str(report.get("mode") or "").strip().lower()
        for run in runs:
            if not isinstance(run, dict):
                continue
            status = str(run.get("status") or "").strip().lower()
            quality_payload = run.get("quality")
            quality_score = (
                float(quality_payload.get("quality_score"))
                if isinstance(quality_payload, dict)
                and isinstance(quality_payload.get("quality_score"), (int, float))
                else 0.0
            )
            runtime_sec = _to_float(run.get("runtime_sec"), 0.0)
            runtime_pressure = 0.0
            if target_runtime_sec is not None and float(target_runtime_sec) > 0:
                runtime_pressure = max(
                    0.0,
                    (runtime_sec - float(target_runtime_sec)) / float(target_runtime_sec),
                )
            mode = str(run.get("mode") or mode_default).strip().lower()
            telemetry = run.get("telemetry")
            telemetry_dict = telemetry if isinstance(telemetry, dict) else {}
            disagreement = _to_float(telemetry_dict.get("disagreement_count"), 0.0)
            selected_count = _to_float(telemetry_dict.get("spawn_selected_count"), 0.0)
            dynamic_count = _to_float(telemetry_dict.get("spawn_dynamic_count"), 0.0)
            min_role_score = _to_float(
                telemetry_dict.get("spawn_min_role_score_config"),
                1.0,
            )
            if min_role_score <= 0:
                min_role_score = 1.0
            max_specialists = _to_float(
                telemetry_dict.get("spawn_max_specialists_config"),
                3.0 if mode == "deep" else 1.0,
            )
            max_dynamic_specialists = _to_float(
                telemetry_dict.get("spawn_max_dynamic_specialists_config"),
                2.0,
            )
            dynamic_enabled = _to_float(
                1.0 if telemetry_dict.get("spawn_dynamic_enabled_config") else 0.0,
                0.0,
            )
            selected_roles_raw = telemetry_dict.get("spawn_selected_roles")
            selected_roles = [
                _normalize_role(item)
                for item in list(selected_roles_raw or [])
                if _normalize_role(item)
            ]
            dynamic_roles_raw = telemetry_dict.get("spawn_dynamic_roles")
            dynamic_roles = [
                _normalize_role(item)
                for item in list(dynamic_roles_raw or [])
                if _normalize_role(item)
            ]
            role_routing = _normalize_string_map(telemetry_dict.get("spawn_role_routing"))
            role_tools = _normalize_tool_map(telemetry_dict.get("spawn_role_tools"))

            reward = quality_score - (runtime_weight_normalized * runtime_pressure)
            if status in {"failed", "canceled"}:
                reward -= failure_penalty_normalized
            if status == "partial":
                reward -= (0.5 * failure_penalty_normalized)
            experiences.append(
                {
                    "run_id": str(run.get("run_id") or run.get("id") or f"run-{len(experiences)+1}"),
                    "status": status,
                    "mode": mode if mode in {"deep", "fast"} else "deep",
                    "quality_score": quality_score,
                    "runtime_sec": runtime_sec,
                    "runtime_pressure": runtime_pressure,
                    "quality_gap": float(target_quality) - quality_score,
                    "disagreement": disagreement,
                    "reward": reward,
                    "selected_roles": selected_roles,
                    "dynamic_roles": dynamic_roles,
                    "role_routing": role_routing,
                    "role_tools": role_tools,
                    "actions": {
                        "selected_count": selected_count,
                        "dynamic_count": dynamic_count,
                        "min_role_score": min_role_score,
                        "max_specialists": max_specialists,
                        "max_dynamic_specialists": max_dynamic_specialists,
                        "dynamic_enabled": dynamic_enabled,
                    },
                }
            )
    return experiences


def train_rl_spawn_policy(
    reports: list[dict[str, Any]],
    *,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
    learning_rate: float = 0.18,
    epochs: int = 3,
    reward_runtime_weight: float = 0.35,
    reward_failure_penalty: float = 1.0,
    prior_spawn_policy_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_reports, report_ids = _valid_training_reports(reports)
    experiences = _rl_spawn_experiences(
        valid_reports,
        target_quality=target_quality,
        target_runtime_sec=target_runtime_sec,
        runtime_weight=reward_runtime_weight,
        failure_penalty=reward_failure_penalty,
    )
    alpha = _clamp_float(float(learning_rate), 0.01, 1.0)
    epoch_count = _clamp_int(int(epochs), 1, 50)
    runtime_weight_normalized = _clamp_float(float(reward_runtime_weight), 0.0, 2.0)
    failure_penalty_normalized = _clamp_float(float(reward_failure_penalty), 0.0, 4.0)
    prior = prior_spawn_policy_env or {}

    spawn = {
        "min_role_score": _to_float(prior.get("RIM_SPAWN_MIN_ROLE_SCORE"), 1.0),
        "max_specialists_deep": _to_float(prior.get("RIM_SPAWN_MAX_SPECIALISTS_DEEP"), 3.0),
        "max_specialists_fast": _to_float(prior.get("RIM_SPAWN_MAX_SPECIALISTS_FAST"), 1.0),
        "max_dynamic_specialists": _to_float(
            prior.get("RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"),
            2.0,
        ),
        "dynamic_enabled_score": _to_float(prior.get("RIM_ENABLE_DYNAMIC_SPECIALISTS"), 1.0),
    }
    role_boosts = _normalize_float_map(prior.get("RIM_SPAWN_ROLE_BOOSTS"))
    dynamic_token_boosts = _normalize_float_map(prior.get("RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS"))
    routing_overrides = _normalize_string_map(prior.get("RIM_SPAWN_ROLE_ROUTING_OVERRIDES"))
    tool_overrides = _normalize_tool_map(prior.get("RIM_SPAWN_ROLE_TOOL_OVERRIDES"))

    if not experiences:
        spawn_env = _blend_spawn_policy_env(
            prior_env=prior_spawn_policy_env,
            fresh_env={},
            learning_rate=0.0,
        )
        payload = {
            "policy_env": spawn_env,
            "recommended_exports": calibration_env_exports({"recommended_env": spawn_env}),
            "rationale": ["No eligible run experiences; returning prior/default spawn policy."],
            "optimizer": "rl_spawn_credit_assignment_v1",
            "epochs": epoch_count,
        }
        return {
            "optimizer": "rl_spawn_credit_assignment_v1",
            "report_count": len(valid_reports),
            "report_ids": report_ids,
            "experience_count": 0,
            "learning_rate": alpha,
            "epochs": epoch_count,
            "target_quality": target_quality,
            "target_runtime_sec": target_runtime_sec,
            "spawn_policy": payload,
            "credit_assignment": {
                "spawn": {"positive": 0.0, "negative": 0.0, "top_runs": []},
                "role_boosts": [],
                "dynamic_token_boosts": [],
            },
            "recommended_exports": payload["recommended_exports"],
        }

    rewards = [float(item["reward"]) for item in experiences]
    reward_baseline = sum(rewards) / len(rewards)
    spawn_credits: list[dict[str, Any]] = []
    role_credit_totals: dict[str, float] = {}
    dynamic_token_credit_totals: dict[str, float] = {}
    routing_votes: dict[str, dict[str, float]] = {}
    tool_votes: dict[str, dict[str, float]] = {}

    for _epoch in range(epoch_count):
        for exp in experiences:
            reward = _to_float(exp.get("reward"), 0.0)
            advantage = reward - reward_baseline
            runtime_pressure = _to_float(exp.get("runtime_pressure"), 0.0)
            quality_gap = _to_float(exp.get("quality_gap"), 0.0)
            disagreement = _to_float(exp.get("disagreement"), 0.0)
            actions = exp.get("actions")
            action_payload = actions if isinstance(actions, dict) else {}
            selected_count = _to_float(action_payload.get("selected_count"), 0.0)
            dynamic_count = _to_float(action_payload.get("dynamic_count"), 0.0)
            min_role_score_action = _to_float(action_payload.get("min_role_score"), 1.0)
            max_specialists_action = _to_float(action_payload.get("max_specialists"), 1.0)
            max_dynamic_action = _to_float(
                action_payload.get("max_dynamic_specialists"),
                0.0,
            )
            dynamic_enabled_action = _to_float(action_payload.get("dynamic_enabled"), 0.0)
            action_intensity = max(
                0.0,
                (0.22 * max(max_specialists_action - 1.0, 0.0))
                + (0.16 * max(max_dynamic_action, 0.0))
                + (0.12 * max(dynamic_enabled_action, 0.0))
                + (0.14 * max(1.0 - min_role_score_action, 0.0)),
            )
            spawn_signal = quality_gap + (0.35 * disagreement) - (0.75 * runtime_pressure)
            spawn_credit = advantage * spawn_signal * (1.0 + action_intensity)

            spawn["min_role_score"] += alpha * (-0.28 * spawn_credit)
            spawn["max_specialists_deep"] += alpha * (0.95 * spawn_credit)
            spawn["max_specialists_fast"] += alpha * (0.45 * spawn_credit)
            spawn["max_dynamic_specialists"] += alpha * (0.65 * spawn_credit)
            spawn["dynamic_enabled_score"] += alpha * (0.32 * spawn_credit)

            spawn["min_role_score"] = _clamp_float(spawn["min_role_score"], 0.4, 3.0)
            spawn["max_specialists_deep"] = _clamp_float(
                spawn["max_specialists_deep"],
                1.0,
                8.0,
            )
            spawn["max_specialists_fast"] = _clamp_float(
                spawn["max_specialists_fast"],
                1.0,
                4.0,
            )
            spawn["max_dynamic_specialists"] = _clamp_float(
                spawn["max_dynamic_specialists"],
                0.0,
                6.0,
            )
            spawn["dynamic_enabled_score"] = _clamp_float(
                spawn["dynamic_enabled_score"],
                0.0,
                1.0,
            )

            selected_roles = list(exp.get("selected_roles") or [])
            role_credit_base = advantage * (1.0 - (0.45 * runtime_pressure))
            role_credit_each = role_credit_base / max(1.0, float(len(selected_roles)))
            role_routing = _normalize_string_map(exp.get("role_routing"))
            role_tools = _normalize_tool_map(exp.get("role_tools"))
            for role in selected_roles:
                normalized_role = _normalize_role(role)
                if not normalized_role:
                    continue
                if normalized_role.startswith("dynamic_"):
                    token = normalized_role[len("dynamic_") :]
                    if token:
                        dynamic_token_boosts[token] = round(
                            _clamp_float(
                                _to_float(dynamic_token_boosts.get(token), 0.0)
                                + (alpha * 0.35 * role_credit_each),
                                -4.0,
                                4.0,
                            ),
                            4,
                        )
                        dynamic_token_credit_totals[token] = _to_float(
                            dynamic_token_credit_totals.get(token),
                            0.0,
                        ) + role_credit_each
                    continue
                role_boosts[normalized_role] = round(
                    _clamp_float(
                        _to_float(role_boosts.get(normalized_role), 0.0)
                        + (alpha * 0.35 * role_credit_each),
                        -4.0,
                        4.0,
                    ),
                    4,
                )
                role_credit_totals[normalized_role] = _to_float(
                    role_credit_totals.get(normalized_role),
                    0.0,
                ) + role_credit_each
                route = role_routing.get(normalized_role)
                if route and role_credit_each > 0.0:
                    role_votes = routing_votes.setdefault(normalized_role, {})
                    role_votes[route] = _to_float(role_votes.get(route), 0.0) + role_credit_each
                tools = role_tools.get(normalized_role)
                if tools and role_credit_each > 0.0:
                    signature = "||".join(tools[:8])
                    role_tool_votes = tool_votes.setdefault(normalized_role, {})
                    role_tool_votes[signature] = _to_float(
                        role_tool_votes.get(signature),
                        0.0,
                    ) + role_credit_each

            spawn_credits.append(
                {
                    "run_id": exp["run_id"],
                    "credit": round(spawn_credit, 6),
                    "advantage": round(advantage, 6),
                    "signal": round(spawn_signal, 6),
                    "action_intensity": round(action_intensity, 6),
                }
            )

    role_boosts = {
        key: value
        for key, value in sorted(
            role_boosts.items(),
            key=lambda item: (abs(float(item[1])), item[0]),
            reverse=True,
        )[:24]
        if abs(_to_float(value, 0.0)) >= 0.05
    }
    dynamic_token_boosts = {
        key: value
        for key, value in sorted(
            dynamic_token_boosts.items(),
            key=lambda item: (abs(float(item[1])), item[0]),
            reverse=True,
        )[:32]
        if abs(_to_float(value, 0.0)) >= 0.05
    }
    for role, votes in routing_votes.items():
        if not votes:
            continue
        winner = max(votes.items(), key=lambda item: item[1])
        if winner[1] > 0:
            routing_overrides[role] = winner[0]
    for role, votes in tool_votes.items():
        if not votes:
            continue
        winner = max(votes.items(), key=lambda item: item[1])
        if winner[1] > 0 and winner[0]:
            tools = [item for item in winner[0].split("||") if item]
            if tools:
                tool_overrides[role] = tools

    spawn_enable_dynamic = 1 if spawn["dynamic_enabled_score"] >= 0.5 else 0
    spawn_env: dict[str, Any] = {
        "RIM_SPAWN_MIN_ROLE_SCORE": round(spawn["min_role_score"], 3),
        "RIM_SPAWN_MAX_SPECIALISTS_DEEP": _clamp_int(
            int(round(spawn["max_specialists_deep"])),
            1,
            8,
        ),
        "RIM_SPAWN_MAX_SPECIALISTS_FAST": _clamp_int(
            int(round(spawn["max_specialists_fast"])),
            1,
            4,
        ),
        "RIM_ENABLE_DYNAMIC_SPECIALISTS": spawn_enable_dynamic,
        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": _clamp_int(
            int(round(spawn["max_dynamic_specialists"])),
            0,
            6,
        ),
    }
    if spawn_enable_dynamic == 0:
        spawn_env["RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"] = 0
    if role_boosts:
        spawn_env["RIM_SPAWN_ROLE_BOOSTS"] = role_boosts
    if dynamic_token_boosts:
        spawn_env["RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS"] = dynamic_token_boosts
    if routing_overrides:
        spawn_env["RIM_SPAWN_ROLE_ROUTING_OVERRIDES"] = routing_overrides
    if tool_overrides:
        spawn_env["RIM_SPAWN_ROLE_TOOL_OVERRIDES"] = tool_overrides

    def _credit_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        positive = sum(float(item["credit"]) for item in items if float(item["credit"]) > 0.0)
        negative = sum(float(item["credit"]) for item in items if float(item["credit"]) < 0.0)
        ranked = sorted(items, key=lambda item: abs(float(item["credit"])), reverse=True)[:8]
        return {
            "positive": round(positive, 6),
            "negative": round(negative, 6),
            "top_runs": ranked,
        }

    role_credit_ranked = sorted(
        (
            {"role": key, "credit": round(value, 6)}
            for key, value in role_credit_totals.items()
        ),
        key=lambda item: abs(float(item["credit"])),
        reverse=True,
    )[:12]
    token_credit_ranked = sorted(
        (
            {"token": key, "credit": round(value, 6)}
            for key, value in dynamic_token_credit_totals.items()
        ),
        key=lambda item: abs(float(item["credit"])),
        reverse=True,
    )[:12]

    spawn_payload = {
        "policy_env": spawn_env,
        "recommended_exports": calibration_env_exports({"recommended_env": spawn_env}),
        "rationale": [
            "Spawn policy updated with reward/advantage credits for role breadth and routing.",
        ],
        "optimizer": "rl_spawn_credit_assignment_v1",
        "epochs": epoch_count,
    }
    return {
        "optimizer": "rl_spawn_credit_assignment_v1",
        "report_count": len(valid_reports),
        "report_ids": report_ids,
        "experience_count": len(experiences),
        "learning_rate": alpha,
        "epochs": epoch_count,
        "target_quality": target_quality,
        "target_runtime_sec": target_runtime_sec,
        "reward_summary": {
            "mean_reward": round(reward_baseline, 6),
            "max_reward": round(max(rewards), 6),
            "min_reward": round(min(rewards), 6),
            "runtime_weight": runtime_weight_normalized,
            "failure_penalty": failure_penalty_normalized,
        },
        "spawn_policy": spawn_payload,
        "credit_assignment": {
            "spawn": _credit_summary(spawn_credits),
            "role_boosts": role_credit_ranked,
            "dynamic_token_boosts": token_credit_ranked,
        },
        "recommended_exports": spawn_payload["recommended_exports"],
    }


def _load_recent_reports(
    *,
    reports_dir: Path,
    lookback_reports: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if lookback_reports <= 0:
        return [], []
    paths = list_reports(reports_dir)
    selected_paths = paths[-lookback_reports:]
    reports: list[dict[str, Any]] = []
    used_paths: list[str] = []
    for path in selected_paths:
        try:
            report = load_report(path)
        except Exception:  # noqa: BLE001
            continue
        if int(report.get("dataset_size", 0)) <= 0:
            continue
        reports.append(report)
        used_paths.append(str(path))
    return reports, used_paths


async def run_online_depth_arbitration_learning_loop(
    orchestrator: RimOrchestrator,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mode: str = "deep",
    limit: int | None = None,
    iterations: int = 2,
    lookback_reports: int = 6,
    target_quality: float = 0.65,
    target_runtime_sec: float | None = None,
    learning_rate: float = 0.35,
    optimizer: str = "blend",
    rl_epochs: int = 3,
    rl_reward_runtime_weight: float = 0.35,
    rl_reward_failure_penalty: float = 1.0,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    depth_policy_path: Path = DEFAULT_POLICIES_DIR / "depth_policy.json",
    specialist_policy_path: Path = DEFAULT_POLICIES_DIR / "specialist_policy.json",
    arbitration_policy_path: Path = DEFAULT_POLICIES_DIR / "arbitration_policy.json",
    spawn_policy_path: Path = DEFAULT_POLICIES_DIR / "spawn_policy.json",
    memory_policy_path: Path = DEFAULT_POLICIES_DIR / "memory_policy.json",
) -> dict[str, Any]:
    cycles = _clamp_int(int(iterations), 1, 24)
    lookback = _clamp_int(int(lookback_reports), 1, 60)
    alpha = _clamp_float(float(learning_rate), 0.05, 1.0)
    optimizer_name = str(optimizer or "blend").strip().lower()
    if optimizer_name not in {"blend", "rl"}:
        optimizer_name = "blend"
    normalized_rl_epochs = _clamp_int(int(rl_epochs), 1, 50)
    normalized_rl_runtime_weight = _clamp_float(float(rl_reward_runtime_weight), 0.0, 2.0)
    normalized_rl_failure_penalty = _clamp_float(float(rl_reward_failure_penalty), 0.0, 4.0)
    reports_dir.mkdir(parents=True, exist_ok=True)
    depth_policy_path.parent.mkdir(parents=True, exist_ok=True)
    specialist_policy_path.parent.mkdir(parents=True, exist_ok=True)
    arbitration_policy_path.parent.mkdir(parents=True, exist_ok=True)
    spawn_policy_path.parent.mkdir(parents=True, exist_ok=True)
    memory_policy_path.parent.mkdir(parents=True, exist_ok=True)

    previous_depth_path = os.getenv("RIM_DEPTH_POLICY_PATH")
    previous_specialist_path = os.getenv("RIM_SPECIALIST_POLICY_PATH")
    previous_arbitration_path = os.getenv("RIM_ARBITRATION_POLICY_PATH")
    previous_spawn_path = os.getenv("RIM_SPAWN_POLICY_PATH")
    previous_memory_path = os.getenv("RIM_MEMORY_POLICY_PATH")
    os.environ["RIM_DEPTH_POLICY_PATH"] = str(depth_policy_path)
    os.environ["RIM_SPECIALIST_POLICY_PATH"] = str(specialist_policy_path)
    os.environ["RIM_ARBITRATION_POLICY_PATH"] = str(arbitration_policy_path)
    os.environ["RIM_SPAWN_POLICY_PATH"] = str(spawn_policy_path)
    os.environ["RIM_MEMORY_POLICY_PATH"] = str(memory_policy_path)
    cycle_outputs: list[dict[str, Any]] = []
    try:
        for cycle in range(1, cycles + 1):
            report = await run_benchmark(
                orchestrator=orchestrator,
                dataset_path=dataset_path,
                mode=mode,
                limit=limit,
            )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            report_path = reports_dir / f"autolearn_{stamp}_cycle{cycle:02d}.json"
            save_report(report, report_path)

            training_reports, training_paths = _load_recent_reports(
                reports_dir=reports_dir,
                lookback_reports=lookback,
            )
            prior_depth_env = load_policy_env(depth_policy_path)
            prior_specialist_env = load_policy_env(specialist_policy_path)
            prior_arbitration_env = load_policy_env(arbitration_policy_path)
            if optimizer_name == "rl":
                learning = train_rl_depth_and_arbitration_policies(
                    training_reports,
                    target_quality=target_quality,
                    target_runtime_sec=target_runtime_sec,
                    learning_rate=alpha,
                    epochs=normalized_rl_epochs,
                    reward_runtime_weight=normalized_rl_runtime_weight,
                    reward_failure_penalty=normalized_rl_failure_penalty,
                    prior_depth_policy_env=prior_depth_env,
                    prior_specialist_policy_env=prior_specialist_env,
                    prior_arbitration_policy_env=prior_arbitration_env,
                )
            else:
                learning = train_online_depth_and_arbitration_policies(
                    training_reports,
                    target_quality=target_quality,
                    target_runtime_sec=target_runtime_sec,
                    learning_rate=alpha,
                    prior_depth_policy_env=prior_depth_env,
                    prior_specialist_policy_env=prior_specialist_env,
                    prior_arbitration_policy_env=prior_arbitration_env,
                )
            prior_spawn_env = load_policy_env(spawn_policy_path)
            if optimizer_name == "rl":
                spawn_learning = train_rl_spawn_policy(
                    training_reports,
                    target_quality=target_quality,
                    target_runtime_sec=target_runtime_sec,
                    learning_rate=alpha,
                    epochs=normalized_rl_epochs,
                    reward_runtime_weight=normalized_rl_runtime_weight,
                    reward_failure_penalty=normalized_rl_failure_penalty,
                    prior_spawn_policy_env=prior_spawn_env,
                )
            else:
                spawn_learning = train_online_spawn_policy(
                    training_reports,
                    target_quality=target_quality,
                    target_runtime_sec=target_runtime_sec,
                    learning_rate=alpha,
                    prior_spawn_policy_env=prior_spawn_env,
                )
            prior_memory_env = load_policy_env(memory_policy_path)
            memory_learning = train_online_memory_policy(
                training_reports,
                target_quality=target_quality,
                target_runtime_sec=target_runtime_sec,
                learning_rate=alpha,
                prior_memory_policy_env=prior_memory_env,
            )
            depth_save_path = save_policy_artifact(
                learning["depth_policy"],
                policy_kind="depth",
                source_reports=training_paths,
                output_path=depth_policy_path,
                learning_meta={
                    "cycle": cycle,
                    "iterations": cycles,
                    "learning_rate": alpha,
                    "report_path": str(report_path),
                },
            )
            specialist_save_path = save_policy_artifact(
                learning["specialist_policy"],
                policy_kind="specialist_arbitration",
                source_reports=training_paths,
                output_path=specialist_policy_path,
                learning_meta={
                    "cycle": cycle,
                    "iterations": cycles,
                    "learning_rate": alpha,
                    "report_path": str(report_path),
                },
            )
            arbitration_save_path = save_policy_artifact(
                learning["arbitration_policy"],
                policy_kind="arbitration",
                source_reports=training_paths,
                output_path=arbitration_policy_path,
                learning_meta={
                    "cycle": cycle,
                    "iterations": cycles,
                    "learning_rate": alpha,
                    "report_path": str(report_path),
                },
            )
            spawn_save_path = save_policy_artifact(
                spawn_learning["spawn_policy"],
                policy_kind="spawn",
                source_reports=training_paths,
                output_path=spawn_policy_path,
                learning_meta={
                    "cycle": cycle,
                    "iterations": cycles,
                    "learning_rate": alpha,
                    "report_path": str(report_path),
                },
            )
            memory_save_path = save_policy_artifact(
                memory_learning["memory_policy"],
                policy_kind="memory",
                source_reports=training_paths,
                output_path=memory_policy_path,
                learning_meta={
                    "cycle": cycle,
                    "iterations": cycles,
                    "learning_rate": alpha,
                    "report_path": str(report_path),
                },
            )
            cycle_outputs.append(
                {
                    "cycle": cycle,
                    "report_path": str(report_path),
                    "report_summary": {
                        "average_quality_score": report.get("average_quality_score"),
                        "average_runtime_sec": report.get("average_runtime_sec"),
                        "dataset_size": report.get("dataset_size"),
                        "failure_count": report.get("failure_count"),
                    },
                    "training_report_count": len(training_reports),
                    "optimizer": learning.get("optimizer", optimizer_name),
                    "depth_policy_path": str(depth_save_path),
                    "specialist_policy_path": str(specialist_save_path),
                    "arbitration_policy_path": str(arbitration_save_path),
                    "spawn_policy_path": str(spawn_save_path),
                    "memory_policy_path": str(memory_save_path),
                    "depth_policy_env": learning["depth_policy"]["policy_env"],
                    "specialist_policy_env": learning["specialist_policy"]["policy_env"],
                    "arbitration_policy_env": learning["arbitration_policy"]["policy_env"],
                    "spawn_policy_env": spawn_learning["spawn_policy"]["policy_env"],
                    "memory_policy_env": memory_learning["memory_policy"]["policy_env"],
                }
            )
    finally:
        if previous_depth_path is None:
            os.environ.pop("RIM_DEPTH_POLICY_PATH", None)
        else:
            os.environ["RIM_DEPTH_POLICY_PATH"] = previous_depth_path
        if previous_specialist_path is None:
            os.environ.pop("RIM_SPECIALIST_POLICY_PATH", None)
        else:
            os.environ["RIM_SPECIALIST_POLICY_PATH"] = previous_specialist_path
        if previous_arbitration_path is None:
            os.environ.pop("RIM_ARBITRATION_POLICY_PATH", None)
        else:
            os.environ["RIM_ARBITRATION_POLICY_PATH"] = previous_arbitration_path
        if previous_spawn_path is None:
            os.environ.pop("RIM_SPAWN_POLICY_PATH", None)
        else:
            os.environ["RIM_SPAWN_POLICY_PATH"] = previous_spawn_path
        if previous_memory_path is None:
            os.environ.pop("RIM_MEMORY_POLICY_PATH", None)
        else:
            os.environ["RIM_MEMORY_POLICY_PATH"] = previous_memory_path

    return {
        "mode": mode,
        "dataset_path": str(dataset_path),
        "iterations": cycles,
        "lookback_reports": lookback,
        "learning_rate": alpha,
        "optimizer": optimizer_name,
        "rl_epochs": normalized_rl_epochs,
        "rl_reward_runtime_weight": normalized_rl_runtime_weight,
        "rl_reward_failure_penalty": normalized_rl_failure_penalty,
        "target_quality": target_quality,
        "target_runtime_sec": target_runtime_sec,
        "depth_policy_path": str(depth_policy_path),
        "specialist_policy_path": str(specialist_policy_path),
        "arbitration_policy_path": str(arbitration_policy_path),
        "spawn_policy_path": str(spawn_policy_path),
        "memory_policy_path": str(memory_policy_path),
        "recommended_exports": [
            f"export RIM_DEPTH_POLICY_PATH={depth_policy_path}",
            f"export RIM_SPECIALIST_POLICY_PATH={specialist_policy_path}",
            f"export RIM_ARBITRATION_POLICY_PATH={arbitration_policy_path}",
            f"export RIM_SPAWN_POLICY_PATH={spawn_policy_path}",
            f"export RIM_MEMORY_POLICY_PATH={memory_policy_path}",
        ],
        "cycles": cycle_outputs,
        "final": cycle_outputs[-1] if cycle_outputs else None,
    }
