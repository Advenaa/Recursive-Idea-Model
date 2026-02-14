from __future__ import annotations

import time
from uuid import uuid4

from rim.agents.critics import run_critics
from rim.agents.decomposer import decompose_idea
from rim.agents.synthesizer import synthesize_idea
from rim.core.modes import get_mode_settings
from rim.core.schemas import (
    AnalyzeRequest,
    AnalyzeResult,
    AnalyzeRunResponse,
    CriticFinding,
    RunLogsResponse,
    StageLogEntry,
)
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def _memory_entries_from_run(
    findings: list[CriticFinding],
    changes_summary: list[str],
    residual_risks: list[str],
) -> list[dict]:
    entries: list[dict] = []

    for change in changes_summary[:5]:
        entries.append(
            {
                "entry_type": "insight",
                "entry_text": f"Synthesis change: {change}",
                "score": 0.75,
            }
        )
    for risk in residual_risks[:5]:
        entries.append(
            {
                "entry_type": "failure",
                "entry_text": f"Residual risk: {risk}",
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
                    "score": max(
                        severity_score.get(finding.severity, 0.5),
                        float(finding.confidence),
                    ),
                }
            )
    return entries[:20]


class RimOrchestrator:
    def __init__(self, repository: RunRepository, router: ProviderRouter) -> None:
        self.repository = repository
        self.router = router

    def create_run(self, request: AnalyzeRequest) -> str:
        run_id = str(uuid4())
        self.repository.create_run(run_id, request.mode, request.idea)
        return run_id

    async def execute_run(self, run_id: str, request: AnalyzeRequest) -> AnalyzeResult:
        settings = get_mode_settings(request.mode)

        try:
            memory_context = self.repository.get_memory_context(limit=8)
            self.repository.log_stage(
                run_id=run_id,
                stage="memory_read",
                status="completed",
                meta={"entries": len(memory_context)},
            )
            decompose_started = time.perf_counter()
            nodes, decompose_provider, decompose_meta = await decompose_idea(
                self.router,
                request.idea,
                settings,
                domain=request.domain,
                constraints=request.constraints,
                memory_context=memory_context,
            )
            self.repository.log_stage(
                run_id=run_id,
                stage="decompose",
                status="completed",
                provider=decompose_provider,
                latency_ms=int((time.perf_counter() - decompose_started) * 1000),
                meta={"node_count": len(nodes), **decompose_meta},
            )
            self.repository.save_nodes(run_id, nodes)

            challenge_started = time.perf_counter()
            findings = await run_critics(self.router, nodes, settings)
            self.repository.log_stage(
                run_id=run_id,
                stage="challenge_parallel",
                status="completed",
                latency_ms=int((time.perf_counter() - challenge_started) * 1000),
                meta={"finding_count": len(findings)},
            )
            self.repository.save_findings(run_id, findings)

            synth_started = time.perf_counter()
            synthesis, synthesis_providers = await synthesize_idea(
                self.router,
                request.idea,
                nodes,
                findings,
                settings,
                memory_context=memory_context,
            )
            self.repository.log_stage(
                run_id=run_id,
                stage="synthesis",
                status="completed",
                provider=",".join(synthesis_providers),
                latency_ms=int((time.perf_counter() - synth_started) * 1000),
            )
            self.repository.save_synthesis(
                run_id=run_id,
                synthesized_idea=synthesis["synthesized_idea"],
                changes_summary=synthesis["changes_summary"],
                residual_risks=synthesis["residual_risks"],
                next_experiments=synthesis["next_experiments"],
            )
            memory_entries = _memory_entries_from_run(
                findings=findings,
                changes_summary=synthesis["changes_summary"],
                residual_risks=synthesis["residual_risks"],
            )
            memory_write_started = time.perf_counter()
            self.repository.save_memory_entries(run_id, memory_entries)
            self.repository.log_stage(
                run_id=run_id,
                stage="memory_write",
                status="completed",
                latency_ms=int((time.perf_counter() - memory_write_started) * 1000),
                meta={"entries": len(memory_entries)},
            )

            result = AnalyzeResult(
                run_id=run_id,
                mode=request.mode,
                input_idea=request.idea,
                decomposition=nodes,
                critic_findings=findings,
                synthesized_idea=synthesis["synthesized_idea"],
                changes_summary=synthesis["changes_summary"],
                residual_risks=synthesis["residual_risks"],
                next_experiments=synthesis["next_experiments"],
                confidence_score=synthesis["confidence_score"],
            )
            self.repository.mark_run_status(
                run_id=run_id,
                status="completed",
                confidence_score=result.confidence_score,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            self.repository.mark_run_status(
                run_id=run_id,
                status="failed",
                error_summary=str(exc),
            )
            self.repository.log_stage(
                run_id=run_id,
                stage="pipeline",
                status="failed",
                meta={"error": str(exc)},
            )
            raise

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResult:
        run_id = self.create_run(request)
        return await self.execute_run(run_id, request)

    def get_run(self, run_id: str) -> AnalyzeRunResponse | None:
        payload = self.repository.get_run(run_id)
        if payload is None:
            return None
        result = (
            AnalyzeResult.model_validate(payload["result"])
            if payload.get("result")
            else None
        )
        return AnalyzeRunResponse(
            run_id=payload["run_id"],
            status=payload["status"],
            error_summary=payload.get("error_summary"),
            result=result,
        )

    def get_run_logs(self, run_id: str) -> RunLogsResponse:
        logs = self.repository.get_stage_logs(run_id)
        return RunLogsResponse(
            run_id=run_id,
            logs=[StageLogEntry.model_validate(item) for item in logs],
        )
