from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from rim.agents.advanced_verifier import run_advanced_verification
from rim.agents.arbitrator import run_arbitration
from rim.agents.critics import run_critics
from rim.agents.decomposer import decompose_idea
from rim.agents.executable_verifier import run_executable_verification
from rim.agents.reconciliation import reconcile_findings
from rim.agents.spawner import build_spawn_plan
from rim.agents.synthesizer import synthesize_idea
from rim.agents.verification import verify_synthesis
from rim.core.depth_allocator import decide_next_cycle, severity_counts
from rim.core.engine_runtime import EngineAgents, RimExecutionEngine
from rim.core.memory_folding import fold_cycle_memory, fold_to_memory_entries
from rim.core.schemas import (
    AnalyzeRequest,
    AnalyzeResult,
    AnalyzeRunResponse,
    RunError,
    RunListResponse,
    RunLogsResponse,
    RunSummary,
    StageLogEntry,
)
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def _delegate_module_global(name: str) -> Callable[..., Any]:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        return globals()[name](*args, **kwargs)

    return _wrapped


def _default_engine_agents() -> EngineAgents:
    return EngineAgents(
        decompose=_delegate_module_global("decompose_idea"),
        run_critics=_delegate_module_global("run_critics"),
        reconcile_findings=_delegate_module_global("reconcile_findings"),
        run_arbitration=_delegate_module_global("run_arbitration"),
        synthesize_idea=_delegate_module_global("synthesize_idea"),
        verify_synthesis=_delegate_module_global("verify_synthesis"),
        run_executable_verification=_delegate_module_global("run_executable_verification"),
        run_advanced_verification=_delegate_module_global("run_advanced_verification"),
        decide_next_cycle=_delegate_module_global("decide_next_cycle"),
        severity_counts=_delegate_module_global("severity_counts"),
        fold_cycle_memory=_delegate_module_global("fold_cycle_memory"),
        fold_to_memory_entries=_delegate_module_global("fold_to_memory_entries"),
        build_spawn_plan=_delegate_module_global("build_spawn_plan"),
    )


class RimOrchestrator:
    def __init__(
        self,
        repository: RunRepository,
        router: ProviderRouter,
        engine: RimExecutionEngine | None = None,
    ) -> None:
        self.repository = repository
        self.router = router
        self.engine = engine or RimExecutionEngine(
            repository=repository,
            router=router,
            agents=_default_engine_agents(),
        )

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
        return await self.engine.execute_run(run_id, request)

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
