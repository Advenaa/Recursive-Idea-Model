from __future__ import annotations

import json
import os
import time
from uuid import uuid4

from rim.agents.critics import run_critics
from rim.agents.decomposer import decompose_idea
from rim.agents.arbitrator import run_arbitration
from rim.agents.executable_verifier import run_executable_verification
from rim.agents.advanced_verifier import run_advanced_verification
from rim.agents.reconciliation import reconcile_findings
from rim.agents.spawner import build_spawn_plan
from rim.agents.synthesizer import synthesize_idea
from rim.agents.verification import verify_synthesis
from rim.core.depth_allocator import decide_next_cycle, severity_counts
from rim.core.memory_folding import fold_cycle_memory, fold_to_memory_entries
from rim.core.modes import get_mode_settings
from rim.core.schemas import (
    AnalyzeRequest,
    AnalyzeResult,
    RunError,
    RunListResponse,
    RunSummary,
    AnalyzeRunResponse,
    CriticFinding,
    DecompositionNode,
    RunLogsResponse,
    StageLogEntry,
)
from rim.providers.base import BudgetExceededError, StageExecutionError
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def _memory_entries_from_run(
    findings: list[CriticFinding],
    changes_summary: list[str],
    residual_risks: list[str],
    domain: str | None,
) -> list[dict]:
    entries: list[dict] = []

    for change in changes_summary[:5]:
        entries.append(
            {
                "entry_type": "insight",
                "entry_text": f"Synthesis change: {change}",
                "domain": domain,
                "severity": "medium",
                "score": 0.75,
            }
        )
    for risk in residual_risks[:5]:
        entries.append(
            {
                "entry_type": "failure",
                "entry_text": f"Residual risk: {risk}",
                "domain": domain,
                "severity": "high",
                "score": 0.7,
            }
        )

    severity_score = {"low": 0.3, "medium": 0.55, "high": 0.75, "critical": 0.9}
    for finding in findings:
        if finding.severity in {"high", "critical"}:
            entries.append(
                {
                    "entry_type": "pattern",
                    "entry_text": f"{finding.critic_type} finding: {finding.issue}",
                    "domain": domain,
                    "severity": finding.severity,
                    "score": max(
                        severity_score.get(finding.severity, 0.5),
                        float(finding.confidence),
                    ),
                }
            )
    return entries[:20]


def _structured_error(
    *,
    stage: str,
    message: str,
    provider: str | None = None,
    retryable: bool = False,
) -> dict:
    return {
        "stage": stage,
        "provider": provider,
        "message": message,
        "retryable": retryable,
    }


def _error_from_exception(exc: Exception, default_stage: str) -> dict:
    if isinstance(exc, StageExecutionError):
        return exc.to_dict()
    if isinstance(exc, BudgetExceededError):
        return _structured_error(
            stage=default_stage,
            message=str(exc),
            provider=None,
            retryable=False,
        )
    return _structured_error(
        stage=default_stage,
        message=str(exc),
        provider=None,
        retryable=False,
    )


def _fallback_synthesis(idea: str, error_message: str) -> dict[str, object]:
    return {
        "synthesized_idea": idea.strip(),
        "changes_summary": [],
        "residual_risks": [f"Synthesis stage failed: {error_message}"],
        "next_experiments": [
            "Retry this run after checking provider health and stage logs.",
            "Run in fast mode once to isolate failure surface.",
            "Reduce constraints and rerun to validate baseline output path.",
        ],
        "confidence_score": 0.25,
    }


def _parse_int_env(
    name: str,
    default: int,
    *,
    lower: int,
    upper: int,
) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw)) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(lower, min(upper, value))


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _parse_float_env(
    name: str,
    default: float,
    *,
    lower: float,
    upper: float,
) -> float:
    raw = os.getenv(name)
    try:
        value = float(str(raw)) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(lower, min(upper, value))


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _coerce_int(value: object, default: int, *, lower: int, upper: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(lower, min(upper, parsed))


def _coerce_float(value: object, default: float, *, lower: float, upper: float) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        parsed = float(default)
    return max(lower, min(upper, parsed))


def _extract_policy_env(payload: object) -> dict[str, object]:
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


def _load_specialist_policy_env(path_value: str) -> tuple[dict[str, object], str | None]:
    path = str(path_value or "").strip()
    if not path:
        return {}, None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.loads(handle.read())
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)
    env = _extract_policy_env(payload)
    allowed = {
        "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP",
        "RIM_SPECIALIST_ARBITRATION_MAX_JOBS",
        "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE",
    }
    filtered = {key: value for key, value in env.items() if key in allowed}
    if not filtered:
        return {}, "No specialist arbitration keys found in policy file."
    return filtered, None


def _load_memory_policy_env(path_value: str) -> tuple[dict[str, object], str | None]:
    path = str(path_value or "").strip()
    if not path:
        return {}, None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.loads(handle.read())
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)
    env = _extract_policy_env(payload)
    allowed = {
        "RIM_ENABLE_MEMORY_FOLDING",
        "RIM_MEMORY_FOLD_MAX_ENTRIES",
        "RIM_MEMORY_FOLD_NOVELTY_FLOOR",
        "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO",
    }
    filtered = {key: value for key, value in env.items() if key in allowed}
    if not filtered:
        return {}, "No memory-fold keys found in policy file."
    return filtered, None


def _next_cycle_memory_context(
    current_memory: list[str],
    synthesis: dict[str, object],
    findings: list[CriticFinding],
) -> list[str]:
    additions: list[str] = []
    for change in list(synthesis.get("changes_summary") or [])[:2]:
        text = str(change).strip()
        if text:
            additions.append(f"Current-run change: {text}")
    for risk in list(synthesis.get("residual_risks") or [])[:2]:
        text = str(risk).strip()
        if text:
            additions.append(f"Current-run risk: {text}")
    for finding in findings:
        if finding.severity not in {"high", "critical"}:
            continue
        issue = str(finding.issue).strip()
        if issue:
            additions.append(f"Current-run finding ({finding.critic_type}): {issue}")
        if len(additions) >= 6:
            break

    deduped: list[str] = []
    seen: set[str] = set()
    for entry in [*current_memory, *additions]:
        normalized = str(entry).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:12]


class RimOrchestrator:
    def __init__(self, repository: RunRepository, router: ProviderRouter) -> None:
        self.repository = repository
        self.router = router

    def create_run(
        self,
        request: AnalyzeRequest,
        status: str = "queued",
        run_id: str | None = None,
    ) -> str:
        run_id = str(run_id or uuid4())
        self.repository.create_run_with_request(
            run_id=run_id,
            mode=request.mode,
            input_idea=request.idea,
            request_json=request.model_dump_json(),
            status=status,
        )
        return run_id

    async def execute_run(self, run_id: str, request: AnalyzeRequest) -> AnalyzeResult:
        settings = get_mode_settings(request.mode)
        provider_session = self.router.create_session(run_id)

        try:
            default_min_severity = "low" if request.mode == "deep" else "medium"
            memory_min_severity = os.getenv("RIM_MEMORY_MIN_SEVERITY", default_min_severity)
            memory_max_age_days = _parse_int_env(
                "RIM_MEMORY_MAX_AGE_DAYS",
                120,
                lower=1,
                upper=3650,
            )
            max_cycles = _parse_int_env(
                "RIM_MAX_ANALYSIS_CYCLES",
                1,
                lower=1,
                upper=6,
            )
            min_confidence_to_stop = _parse_float_env(
                "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE",
                0.78,
                lower=0.0,
                upper=1.0,
            )
            max_residual_risks_to_stop = _parse_int_env(
                "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS",
                2,
                lower=0,
                upper=20,
            )
            max_high_findings_to_stop = _parse_int_env(
                "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS",
                1,
                lower=0,
                upper=20,
            )
            consensus_min_agents = _parse_int_env(
                "RIM_RECONCILE_CONSENSUS_MIN_AGENTS",
                3,
                lower=2,
                upper=8,
            )
            consensus_min_confidence = _parse_float_env(
                "RIM_RECONCILE_CONSENSUS_MIN_CONFIDENCE",
                0.7,
                lower=0.0,
                upper=1.0,
            )
            reconcile_min_unique_critics = _parse_int_env(
                "RIM_RECONCILE_MIN_UNIQUE_CRITICS",
                3,
                lower=1,
                upper=8,
            )
            reconcile_max_single_critic_share = _parse_float_env(
                "RIM_RECONCILE_MAX_SINGLE_CRITIC_SHARE",
                0.7,
                lower=0.0,
                upper=1.0,
            )
            verification_default = request.mode == "deep"
            verification_enabled = _parse_bool_env(
                "RIM_ENABLE_VERIFICATION",
                verification_default,
            )
            verification_min_constraint_overlap = _parse_float_env(
                "RIM_VERIFY_MIN_CONSTRAINT_OVERLAP",
                0.6,
                lower=0.0,
                upper=1.0,
            )
            verification_min_finding_overlap = _parse_float_env(
                "RIM_VERIFY_MIN_FINDING_OVERLAP",
                0.35,
                lower=0.0,
                upper=1.0,
            )
            executable_verification_enabled = _parse_bool_env(
                "RIM_ENABLE_EXECUTABLE_VERIFICATION",
                request.mode == "deep",
            )
            executable_verification_max_checks = _parse_int_env(
                "RIM_EXEC_VERIFY_MAX_CHECKS",
                5,
                lower=1,
                upper=20,
            )
            python_exec_checks_enabled = _parse_bool_env(
                "RIM_ENABLE_PYTHON_EXEC_CHECKS",
                False,
            )
            python_exec_timeout_sec = _parse_int_env(
                "RIM_PYTHON_EXEC_TIMEOUT_SEC",
                2,
                lower=1,
                upper=15,
            )
            advanced_verification_enabled = _parse_bool_env(
                "RIM_ENABLE_ADVANCED_VERIFICATION",
                request.mode == "deep",
            )
            advanced_verification_max_checks = _parse_int_env(
                "RIM_ADV_VERIFY_MAX_CHECKS",
                4,
                lower=1,
                upper=20,
            )
            advanced_verification_sim_trials = _parse_int_env(
                "RIM_ADV_VERIFY_SIMULATION_TRIALS",
                200,
                lower=10,
                upper=5000,
            )
            advanced_verification_sim_min_pass_rate = _parse_float_env(
                "RIM_ADV_VERIFY_SIMULATION_MIN_PASS_RATE",
                0.7,
                lower=0.0,
                upper=1.0,
            )
            advanced_verification_data_path = os.getenv(
                "RIM_ADV_VERIFY_DATA_PATH",
                "rim/eval/data/benchmark_ideas.jsonl",
            )
            advanced_verification_seed = _parse_int_env(
                "RIM_ADV_VERIFY_SIMULATION_SEED",
                42,
                lower=0,
                upper=2_000_000_000,
            )
            advanced_verify_external_timeout_sec = _parse_int_env(
                "RIM_ADV_VERIFY_EXTERNAL_TIMEOUT_SEC",
                8,
                lower=1,
                upper=120,
            )
            advanced_verify_external_solver_cmd = os.getenv(
                "RIM_ADV_VERIFY_EXTERNAL_SOLVER_CMD"
            )
            advanced_verify_external_simulation_cmd = os.getenv(
                "RIM_ADV_VERIFY_EXTERNAL_SIMULATION_CMD"
            )
            advanced_verify_external_data_cmd = os.getenv(
                "RIM_ADV_VERIFY_EXTERNAL_DATA_CMD"
            )
            memory_policy_path = str(os.getenv("RIM_MEMORY_POLICY_PATH", "")).strip()
            memory_policy_env: dict[str, object] = {}
            memory_policy_error: str | None = None
            if memory_policy_path:
                memory_policy_env, memory_policy_error = _load_memory_policy_env(
                    memory_policy_path
                )
            memory_folding_default = request.mode == "deep"
            memory_fold_max_entries_default = 12
            memory_fold_novelty_floor_default = 0.35
            memory_fold_max_duplicate_ratio_default = 0.5
            if memory_policy_env:
                memory_folding_default = _coerce_bool(
                    memory_policy_env.get("RIM_ENABLE_MEMORY_FOLDING"),
                    memory_folding_default,
                )
                memory_fold_max_entries_default = _coerce_int(
                    memory_policy_env.get("RIM_MEMORY_FOLD_MAX_ENTRIES"),
                    memory_fold_max_entries_default,
                    lower=6,
                    upper=40,
                )
                memory_fold_novelty_floor_default = _coerce_float(
                    memory_policy_env.get("RIM_MEMORY_FOLD_NOVELTY_FLOOR"),
                    memory_fold_novelty_floor_default,
                    lower=0.0,
                    upper=1.0,
                )
                memory_fold_max_duplicate_ratio_default = _coerce_float(
                    memory_policy_env.get("RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO"),
                    memory_fold_max_duplicate_ratio_default,
                    lower=0.0,
                    upper=1.0,
                )
            enable_memory_folding = _parse_bool_env(
                "RIM_ENABLE_MEMORY_FOLDING",
                memory_folding_default,
            )
            memory_fold_max_entries = _parse_int_env(
                "RIM_MEMORY_FOLD_MAX_ENTRIES",
                memory_fold_max_entries_default,
                lower=6,
                upper=40,
            )
            memory_fold_novelty_floor = _parse_float_env(
                "RIM_MEMORY_FOLD_NOVELTY_FLOOR",
                memory_fold_novelty_floor_default,
                lower=0.0,
                upper=1.0,
            )
            memory_fold_max_duplicate_ratio = _parse_float_env(
                "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO",
                memory_fold_max_duplicate_ratio_default,
                lower=0.0,
                upper=1.0,
            )
            enable_arbitration = _parse_bool_env(
                "RIM_ENABLE_DISAGREEMENT_ARBITRATION",
                request.mode == "deep",
            )
            arbitration_max_jobs = _parse_int_env(
                "RIM_ARBITRATION_MAX_JOBS",
                2,
                lower=0,
                upper=6,
            )
            enable_devils_advocate_arbitration = _parse_bool_env(
                "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION",
                request.mode == "deep",
            )
            devils_advocate_rounds = _parse_int_env(
                "RIM_DEVILS_ADVOCATE_ROUNDS",
                1,
                lower=0,
                upper=3,
            )
            devils_advocate_min_confidence = _parse_float_env(
                "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE",
                0.72,
                lower=0.0,
                upper=1.0,
            )
            specialist_policy_path = str(os.getenv("RIM_SPECIALIST_POLICY_PATH", "")).strip()
            specialist_policy_env: dict[str, object] = {}
            specialist_policy_error: str | None = None
            if specialist_policy_path:
                specialist_policy_env, specialist_policy_error = _load_specialist_policy_env(
                    specialist_policy_path
                )
            specialist_loop_default = request.mode == "deep"
            specialist_max_jobs_default = 2
            specialist_min_conf_default = 0.78
            if specialist_policy_env:
                specialist_loop_default = _coerce_bool(
                    specialist_policy_env.get("RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP"),
                    specialist_loop_default,
                )
                specialist_max_jobs_default = _coerce_int(
                    specialist_policy_env.get("RIM_SPECIALIST_ARBITRATION_MAX_JOBS"),
                    specialist_max_jobs_default,
                    lower=0,
                    upper=6,
                )
                specialist_min_conf_default = _coerce_float(
                    specialist_policy_env.get("RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE"),
                    specialist_min_conf_default,
                    lower=0.0,
                    upper=1.0,
                )
            enable_specialist_arbitration_loop = _parse_bool_env(
                "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP",
                specialist_loop_default,
            )
            specialist_arbitration_max_jobs = _parse_int_env(
                "RIM_SPECIALIST_ARBITRATION_MAX_JOBS",
                specialist_max_jobs_default,
                lower=0,
                upper=6,
            )
            specialist_arbitration_min_confidence = _parse_float_env(
                "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE",
                specialist_min_conf_default,
                lower=0.0,
                upper=1.0,
            )
            self.repository.mark_run_status(run_id=run_id, status="running")
            self.repository.log_stage(
                run_id=run_id,
                stage="queue",
                status="completed",
                meta={
                    "mode": request.mode,
                    "max_cycles": max_cycles,
                    "memory_policy_applied": bool(memory_policy_env),
                    "memory_policy_path": memory_policy_path or None,
                    "memory_policy_error": memory_policy_error,
                    "specialist_policy_applied": bool(specialist_policy_env),
                    "specialist_policy_path": specialist_policy_path or None,
                    "specialist_policy_error": specialist_policy_error,
                },
            )
            memory_context = self.repository.get_memory_context(
                limit=8,
                domain=request.domain,
                max_age_days=memory_max_age_days,
                min_severity=memory_min_severity,
            )
            self.repository.log_stage(
                run_id=run_id,
                stage="memory_read",
                status="completed",
                meta={
                    "entries": len(memory_context),
                    "domain": request.domain,
                    "max_age_days": memory_max_age_days,
                    "min_severity": memory_min_severity,
                },
            )
            spawn_plan = build_spawn_plan(
                mode=request.mode,
                domain=request.domain,
                constraints=request.constraints,
                memory_context=memory_context,
            )
            spawned_extra_critics = [
                (str(item["stage"]), str(item["critic_type"]))
                for item in list(spawn_plan.get("extra_critics") or [])
                if isinstance(item, dict)
                and str(item.get("stage") or "").strip()
                and str(item.get("critic_type") or "").strip()
            ]
            self.repository.log_stage(
                run_id=run_id,
                stage="specialization_spawn",
                status="completed",
                meta=spawn_plan,
            )
            current_idea = request.idea
            working_memory_context = list(memory_context)
            previous_confidence: float | None = None
            cycles_completed = 0
            partial_error: dict | None = None
            nodes: list[DecompositionNode] = []
            findings: list[CriticFinding] = []
            folded_memory_entries: list[dict] = []
            synthesis: dict[str, object] = _fallback_synthesis(
                request.idea,
                "pipeline did not produce synthesis output",
            )

            for cycle in range(1, max_cycles + 1):
                decompose_started = time.perf_counter()
                nodes, decompose_provider, decompose_meta = await decompose_idea(
                    provider_session,
                    current_idea,
                    settings,
                    domain=request.domain,
                    constraints=request.constraints,
                    memory_context=working_memory_context,
                )
                self.repository.log_stage(
                    run_id=run_id,
                    stage="decompose",
                    status="completed",
                    provider=decompose_provider,
                    latency_ms=int((time.perf_counter() - decompose_started) * 1000),
                    meta={"cycle": cycle, "node_count": len(nodes), **decompose_meta},
                )

                challenge_started = time.perf_counter()
                findings = await run_critics(
                    provider_session,
                    nodes,
                    settings,
                    domain=request.domain,
                    extra_critics=spawned_extra_critics,
                )
                self.repository.log_stage(
                    run_id=run_id,
                    stage="challenge_parallel",
                    status="completed",
                    latency_ms=int((time.perf_counter() - challenge_started) * 1000),
                    meta={"cycle": cycle, "finding_count": len(findings)},
                )
                reconciliation = reconcile_findings(
                    findings,
                    consensus_min_agents=consensus_min_agents,
                    consensus_min_confidence=consensus_min_confidence,
                    min_unique_critics_per_node=reconcile_min_unique_critics,
                    max_single_critic_share=reconcile_max_single_critic_share,
                )
                diversity_summary = reconciliation.get("diversity_guardrails", {}).get("summary", {})
                self.repository.log_stage(
                    run_id=run_id,
                    stage="challenge_reconciliation",
                    status="completed",
                    meta={
                        "cycle": cycle,
                        **reconciliation["summary"],
                        "diversity_flagged_count": diversity_summary.get("flagged_count", 0),
                        "diversity_nodes_evaluated": diversity_summary.get("nodes_evaluated", 0),
                        "diversity_avg_unique_critics_per_node": diversity_summary.get(
                            "avg_unique_critics_per_node",
                            0.0,
                        ),
                    },
                )
                arbitrations: list[dict[str, object]] = []
                if enable_arbitration and reconciliation["summary"]["disagreement_count"] > 0:
                    arbitration_started = time.perf_counter()
                    try:
                        arbitrations, arbitration_providers = await run_arbitration(
                            provider_session,
                            reconciliation=reconciliation,
                            findings=findings,
                            max_jobs=arbitration_max_jobs,
                            devils_advocate_rounds=(
                                devils_advocate_rounds
                                if enable_devils_advocate_arbitration
                                else 0
                            ),
                            devils_advocate_min_confidence=devils_advocate_min_confidence,
                            specialist_loop_enabled=enable_specialist_arbitration_loop,
                            specialist_max_jobs=specialist_arbitration_max_jobs,
                            specialist_min_confidence=specialist_arbitration_min_confidence,
                        )
                        devils_advocate_count = len(
                            [
                                item
                                for item in arbitrations
                                if str(item.get("round", "")).startswith("devil_")
                            ]
                        )
                        specialist_count = len(
                            [
                                item
                                for item in arbitrations
                                if str(item.get("round", "")) == "specialist"
                            ]
                        )
                        self.repository.log_stage(
                            run_id=run_id,
                            stage="challenge_arbitration",
                            status="completed",
                            provider=",".join(arbitration_providers) if arbitration_providers else None,
                            latency_ms=int((time.perf_counter() - arbitration_started) * 1000),
                            meta={
                                "cycle": cycle,
                                "jobs_requested": min(
                                    arbitration_max_jobs,
                                    int(reconciliation["summary"]["disagreement_count"]),
                                ),
                                "resolved_count": len(arbitrations),
                                "devils_advocate_enabled": enable_devils_advocate_arbitration,
                                "devils_advocate_rounds": (
                                    devils_advocate_rounds
                                    if enable_devils_advocate_arbitration
                                    else 0
                                ),
                                "devils_advocate_min_confidence": devils_advocate_min_confidence,
                                "devils_advocate_count": devils_advocate_count,
                                "specialist_loop_enabled": enable_specialist_arbitration_loop,
                                "specialist_max_jobs": specialist_arbitration_max_jobs,
                                "specialist_min_confidence": specialist_arbitration_min_confidence,
                                "specialist_count": specialist_count,
                                "specialist_policy_applied": bool(specialist_policy_env),
                                "specialist_policy_path": specialist_policy_path or None,
                                "specialist_policy_error": specialist_policy_error,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.repository.log_stage(
                            run_id=run_id,
                            stage="challenge_arbitration",
                            status="failed",
                            latency_ms=int((time.perf_counter() - arbitration_started) * 1000),
                            meta={
                                "cycle": cycle,
                                "devils_advocate_enabled": enable_devils_advocate_arbitration,
                                "devils_advocate_rounds": (
                                    devils_advocate_rounds
                                    if enable_devils_advocate_arbitration
                                    else 0
                                ),
                                "devils_advocate_min_confidence": devils_advocate_min_confidence,
                                "specialist_loop_enabled": enable_specialist_arbitration_loop,
                                "specialist_max_jobs": specialist_arbitration_max_jobs,
                                "specialist_min_confidence": specialist_arbitration_min_confidence,
                                "specialist_policy_applied": bool(specialist_policy_env),
                                "specialist_policy_path": specialist_policy_path or None,
                                "specialist_policy_error": specialist_policy_error,
                                "error": _error_from_exception(exc, default_stage="challenge_arbitration"),
                            },
                        )
                        arbitrations = []

                synth_started = time.perf_counter()
                try:
                    synthesis, synthesis_providers = await synthesize_idea(
                        provider_session,
                        current_idea,
                        nodes,
                        findings,
                        settings,
                        memory_context=working_memory_context,
                        reconciliation=reconciliation,
                        arbitrations=arbitrations,
                    )
                    self.repository.log_stage(
                        run_id=run_id,
                        stage="synthesis",
                        status="completed",
                        provider=",".join(synthesis_providers),
                        latency_ms=int((time.perf_counter() - synth_started) * 1000),
                        meta={"cycle": cycle},
                    )
                except Exception as exc:  # noqa: BLE001
                    partial_error = _error_from_exception(exc, default_stage="synthesis")
                    synthesis = _fallback_synthesis(
                        current_idea,
                        str(partial_error["message"]),
                    )
                    self.repository.log_stage(
                        run_id=run_id,
                        stage="synthesis",
                        status="failed",
                        provider=partial_error.get("provider"),
                        latency_ms=int((time.perf_counter() - synth_started) * 1000),
                        meta={"cycle": cycle, "error": partial_error},
                    )
                    cycles_completed = cycle
                    break

                if verification_enabled:
                    verification = verify_synthesis(
                        synthesis=synthesis,
                        findings=findings,
                        constraints=request.constraints,
                        min_constraint_overlap=verification_min_constraint_overlap,
                        min_finding_overlap=verification_min_finding_overlap,
                    )
                    self.repository.log_stage(
                        run_id=run_id,
                        stage="verification",
                        status="completed",
                        meta={"cycle": cycle, **verification["summary"]},
                    )
                    failed_checks = [
                        check
                        for check in list(verification.get("checks") or [])
                        if not bool(check.get("passed"))
                    ]
                    if failed_checks:
                        risks = [
                            str(item)
                            for item in list(synthesis.get("residual_risks") or [])
                            if str(item).strip()
                        ]
                        for check in failed_checks[:3]:
                            detail = str(check.get("description") or "unknown issue").strip()
                            check_type = str(check.get("check_type") or "verification").strip()
                            message = f"Verification check failed ({check_type}): {detail}"
                            if message not in risks:
                                risks.append(message)
                        synthesis["residual_risks"] = risks[:8]
                        penalty = min(0.3, 0.05 * len(failed_checks))
                        adjusted_confidence = max(
                            0.0,
                            min(
                                1.0,
                                float(synthesis.get("confidence_score", 0.5)) - penalty,
                            ),
                        )
                        synthesis["confidence_score"] = adjusted_confidence

                if executable_verification_enabled:
                    exec_verification = run_executable_verification(
                        constraints=request.constraints,
                        synthesis=synthesis,
                        findings=findings,
                        max_checks=executable_verification_max_checks,
                        enable_python_exec=python_exec_checks_enabled,
                        python_exec_timeout_sec=python_exec_timeout_sec,
                    )
                    self.repository.log_stage(
                        run_id=run_id,
                        stage="verification_executable",
                        status="completed",
                        meta={
                            "cycle": cycle,
                            "python_exec_enabled": python_exec_checks_enabled,
                            **exec_verification["summary"],
                        },
                    )
                    failed_exec_checks = [
                        check
                        for check in list(exec_verification.get("checks") or [])
                        if not bool(check.get("passed"))
                    ]
                    if failed_exec_checks:
                        risks = [
                            str(item)
                            for item in list(synthesis.get("residual_risks") or [])
                            if str(item).strip()
                        ]
                        for check in failed_exec_checks[:3]:
                            expression = str(check.get("expression") or "unknown expression").strip()
                            message = f"Executable verification failed: {expression}"
                            if message not in risks:
                                risks.append(message)
                        synthesis["residual_risks"] = risks[:8]
                        synthesis["confidence_score"] = max(
                            0.0,
                            min(
                                1.0,
                                float(synthesis.get("confidence_score", 0.5))
                                - min(0.4, 0.08 * len(failed_exec_checks)),
                            ),
                        )

                if advanced_verification_enabled:
                    advanced_verification = run_advanced_verification(
                        constraints=request.constraints,
                        synthesis=synthesis,
                        findings=findings,
                        max_checks=advanced_verification_max_checks,
                        simulation_trials=advanced_verification_sim_trials,
                        simulation_min_pass_rate=advanced_verification_sim_min_pass_rate,
                        data_reference_path=advanced_verification_data_path,
                        simulation_seed=advanced_verification_seed,
                        external_solver_cmd=advanced_verify_external_solver_cmd,
                        external_simulation_cmd=advanced_verify_external_simulation_cmd,
                        external_data_cmd=advanced_verify_external_data_cmd,
                        external_timeout_sec=advanced_verify_external_timeout_sec,
                    )
                    self.repository.log_stage(
                        run_id=run_id,
                        stage="verification_advanced",
                        status="completed",
                        meta={
                            "cycle": cycle,
                            "simulation_trials": advanced_verification_sim_trials,
                            "simulation_min_pass_rate": advanced_verification_sim_min_pass_rate,
                            "data_reference_path": advanced_verification_data_path,
                            "simulation_seed": advanced_verification_seed,
                            "external_timeout_sec": advanced_verify_external_timeout_sec,
                            "external_solver_enabled": bool(
                                str(advanced_verify_external_solver_cmd or "").strip()
                            ),
                            "external_simulation_enabled": bool(
                                str(advanced_verify_external_simulation_cmd or "").strip()
                            ),
                            "external_data_enabled": bool(
                                str(advanced_verify_external_data_cmd or "").strip()
                            ),
                            **advanced_verification["summary"],
                        },
                    )
                    failed_adv_checks = [
                        check
                        for check in list(advanced_verification.get("checks") or [])
                        if not bool(check.get("passed"))
                    ]
                    if failed_adv_checks:
                        risks = [
                            str(item)
                            for item in list(synthesis.get("residual_risks") or [])
                            if str(item).strip()
                        ]
                        for check in failed_adv_checks[:3]:
                            check_type = str(check.get("check_type") or "advanced").strip()
                            detail = (
                                str(check.get("expression") or "")
                                or ", ".join(list(check.get("terms") or [])[:3])
                                or "constraint"
                            )
                            detail = detail.strip() or "constraint"
                            message = f"Advanced verification failed ({check_type}): {detail}"
                            if message not in risks:
                                risks.append(message)
                        synthesis["residual_risks"] = risks[:8]
                        synthesis["confidence_score"] = max(
                            0.0,
                            min(
                                1.0,
                                float(synthesis.get("confidence_score", 0.5))
                                - min(0.35, 0.06 * len(failed_adv_checks)),
                            ),
                        )

                high_findings, critical_findings = severity_counts(findings)
                decision = decide_next_cycle(
                    cycle=cycle,
                    max_cycles=max_cycles,
                    confidence_score=float(synthesis["confidence_score"]),
                    residual_risk_count=len(
                        list(synthesis["residual_risks"])
                        if isinstance(synthesis.get("residual_risks"), list)
                        else []
                    ),
                    high_severity_findings=high_findings,
                    critical_findings=critical_findings,
                    previous_confidence=previous_confidence,
                    min_confidence_to_stop=min_confidence_to_stop,
                    max_residual_risks_to_stop=max_residual_risks_to_stop,
                    max_high_findings_to_stop=max_high_findings_to_stop,
                )
                self.repository.log_stage(
                    run_id=run_id,
                    stage="depth_allocator",
                    status="completed",
                    meta={
                        "cycle": cycle,
                        "decision": decision.__dict__,
                    },
                )
                cycles_completed = cycle
                if not decision.recurse:
                    break
                previous_confidence = float(synthesis["confidence_score"])
                current_idea = str(synthesis["synthesized_idea"])
                if enable_memory_folding:
                    fold_payload = fold_cycle_memory(
                        cycle=cycle,
                        prior_context=working_memory_context,
                        synthesis=synthesis,
                        findings=findings,
                        max_entries=memory_fold_max_entries,
                        novelty_floor=memory_fold_novelty_floor,
                        max_duplicate_ratio=memory_fold_max_duplicate_ratio,
                    )
                    working_memory_context = list(fold_payload["folded_context"])
                    cycle_fold_entries = fold_to_memory_entries(
                        fold_payload,
                        domain=request.domain,
                    )
                    folded_memory_entries.extend(cycle_fold_entries)
                    fold_quality = fold_payload.get("quality", {})
                    self.repository.log_stage(
                        run_id=run_id,
                        stage="memory_fold",
                        status="completed",
                        meta={
                            "cycle": cycle,
                            "fold_version": fold_payload.get("fold_version", "v1"),
                            "folded_context_entries": len(working_memory_context),
                            "episodic_entries": len(list(fold_payload["episodic"])),
                            "working_entries": len(list(fold_payload["working"])),
                            "tool_entries": len(list(fold_payload["tool"])),
                            "persisted_entries": len(cycle_fold_entries),
                            "degradation_detected": bool(
                                fold_quality.get("degradation_detected", False)
                            ),
                            "degradation_reasons": list(
                                fold_quality.get("degradation_reasons") or []
                            )[:4],
                            "novelty_ratio": fold_quality.get("novelty_ratio", 0.0),
                            "duplicate_ratio": fold_quality.get("duplicate_ratio", 0.0),
                            "memory_policy_applied": bool(memory_policy_env),
                            "memory_policy_path": memory_policy_path or None,
                            "memory_policy_error": memory_policy_error,
                        },
                    )
                else:
                    working_memory_context = _next_cycle_memory_context(
                        working_memory_context,
                        synthesis,
                        findings,
                    )

            self.repository.save_nodes(run_id, nodes)
            self.repository.save_findings(run_id, findings)
            self.repository.save_synthesis(
                run_id=run_id,
                synthesized_idea=str(synthesis["synthesized_idea"]),
                changes_summary=list(synthesis["changes_summary"]),
                residual_risks=list(synthesis["residual_risks"]),
                next_experiments=list(synthesis["next_experiments"]),
            )
            memory_entries = _memory_entries_from_run(
                findings=findings,
                changes_summary=list(synthesis["changes_summary"]),
                residual_risks=list(synthesis["residual_risks"]),
                domain=request.domain,
            )
            memory_entries.extend(folded_memory_entries)
            memory_entries = memory_entries[:32]
            memory_write_started = time.perf_counter()
            self.repository.save_memory_entries(run_id, memory_entries)
            self.repository.log_stage(
                run_id=run_id,
                stage="memory_write",
                status="completed",
                latency_ms=int((time.perf_counter() - memory_write_started) * 1000),
                meta={"entries": len(memory_entries), "cycles_completed": cycles_completed},
            )
            provider_budget_meta = provider_session.get_usage_meta()
            provider_budget_meta["cycles_completed"] = cycles_completed
            self.repository.log_stage(
                run_id=run_id,
                stage="provider_budget",
                status="completed",
                meta=provider_budget_meta,
            )

            result = AnalyzeResult(
                run_id=run_id,
                mode=request.mode,
                input_idea=request.idea,
                decomposition=nodes,
                critic_findings=findings,
                synthesized_idea=str(synthesis["synthesized_idea"]),
                changes_summary=list(synthesis["changes_summary"]),
                residual_risks=list(synthesis["residual_risks"]),
                next_experiments=list(synthesis["next_experiments"]),
                confidence_score=float(synthesis["confidence_score"]),
            )
            if partial_error is not None:
                self.repository.mark_run_status(
                    run_id=run_id,
                    status="partial",
                    confidence_score=result.confidence_score,
                    error_summary=json.dumps(partial_error),
                )
                return result
            self.repository.mark_run_status(
                run_id=run_id,
                status="completed",
                confidence_score=result.confidence_score,
            )
            return result
        except BudgetExceededError as exc:
            error = _error_from_exception(exc, default_stage="provider_budget")
            self.repository.log_stage(
                run_id=run_id,
                stage="provider_budget",
                status="failed",
                meta={"error": error, **provider_session.get_usage_meta()},
            )
            self.repository.mark_run_status(
                run_id=run_id,
                status="failed",
                error_summary=json.dumps(error),
            )
            raise
        except Exception as exc:  # noqa: BLE001
            error = _error_from_exception(exc, default_stage="pipeline")
            self.repository.mark_run_status(
                run_id=run_id,
                status="failed",
                error_summary=json.dumps(error),
            )
            self.repository.log_stage(
                run_id=run_id,
                stage=str(error["stage"]),
                status="failed",
                provider=error.get("provider"),
                meta={"error": error},
            )
            raise

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResult:
        run_id = self.create_run(request, status="running")
        return await self.execute_run(run_id, request)

    def get_run_request(self, run_id: str) -> AnalyzeRequest | None:
        payload = self.repository.get_run_request(run_id)
        if payload is None:
            return None
        try:
            return AnalyzeRequest.model_validate(payload)
        except Exception:  # noqa: BLE001
            return None

    def get_run(self, run_id: str) -> AnalyzeRunResponse | None:
        payload = self.repository.get_run(run_id)
        if payload is None:
            return None
        result = (
            AnalyzeResult.model_validate(payload["result"])
            if payload.get("result")
            else None
        )
        error = None
        if isinstance(payload.get("error"), dict):
            try:
                error = RunError.model_validate(payload["error"])
            except Exception:  # noqa: BLE001
                error = None
        return AnalyzeRunResponse(
            run_id=payload["run_id"],
            status=payload["status"],
            error_summary=payload.get("error_summary"),
            error=error,
            result=result,
        )

    def list_runs(
        self,
        *,
        status: str | None = None,
        mode: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> RunListResponse:
        rows = self.repository.list_runs(
            status=status,
            mode=mode,
            limit=limit,
            offset=offset,
        )
        items = [
            RunSummary(
                run_id=str(row["id"]),
                mode=str(row["mode"]),
                input_idea=str(row["input_idea"]),
                status=str(row["status"]),
                created_at=str(row["created_at"]),
                completed_at=row.get("completed_at"),
                confidence_score=row.get("confidence_score"),
                error_summary=row.get("error_summary"),
            )
            for row in rows
        ]
        return RunListResponse(
            runs=items,
            count=len(items),
            limit=max(1, min(int(limit), 200)),
            offset=max(0, int(offset)),
            status=status,
            mode=mode,
        )

    def get_run_logs(self, run_id: str) -> RunLogsResponse:
        logs = self.repository.get_stage_logs(run_id)
        return RunLogsResponse(
            run_id=run_id,
            logs=[StageLogEntry.model_validate(item) for item in logs],
        )

    def submit_feedback(
        self,
        run_id: str,
        verdict: str,
        notes: str | None = None,
    ) -> dict:
        feedback = self.repository.submit_run_feedback(
            run_id=run_id,
            verdict=verdict,
            notes=notes,
        )
        self.repository.log_stage(
            run_id=run_id,
            stage="feedback",
            status="completed",
            meta={
                "verdict": feedback["verdict"],
                "updated_memory_entries": feedback["updated_memory_entries"],
            },
        )
        return feedback
