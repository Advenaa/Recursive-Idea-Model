from __future__ import annotations

import json
import os
import time
from uuid import uuid4

from rim.agents.critics import run_critics
from rim.agents.decomposer import decompose_idea
from rim.agents.reconciliation import reconcile_findings
from rim.agents.synthesizer import synthesize_idea
from rim.agents.verification import verify_synthesis
from rim.core.depth_allocator import decide_next_cycle, severity_counts
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
            self.repository.mark_run_status(run_id=run_id, status="running")
            self.repository.log_stage(
                run_id=run_id,
                stage="queue",
                status="completed",
                meta={"mode": request.mode, "max_cycles": max_cycles},
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
            current_idea = request.idea
            working_memory_context = list(memory_context)
            previous_confidence: float | None = None
            cycles_completed = 0
            partial_error: dict | None = None
            nodes: list[DecompositionNode] = []
            findings: list[CriticFinding] = []
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
                findings = await run_critics(provider_session, nodes, settings)
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
                )
                self.repository.log_stage(
                    run_id=run_id,
                    stage="challenge_reconciliation",
                    status="completed",
                    meta={"cycle": cycle, **reconciliation["summary"]},
                )

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
