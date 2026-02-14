from __future__ import annotations

from uuid import uuid4

from rim.agents.critics import run_critics
from rim.agents.decomposer import decompose_idea
from rim.agents.synthesizer import synthesize_idea
from rim.core.modes import get_mode_settings
from rim.core.schemas import AnalyzeRequest, AnalyzeResult, AnalyzeRunResponse
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


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
            nodes, decompose_provider = await decompose_idea(
                self.router,
                request.idea,
                settings,
                domain=request.domain,
                constraints=request.constraints,
            )
            self.repository.log_stage(
                run_id=run_id,
                stage="decompose",
                status="completed",
                provider=decompose_provider,
                meta={"node_count": len(nodes)},
            )
            self.repository.save_nodes(run_id, nodes)

            findings = await run_critics(self.router, nodes, settings)
            self.repository.log_stage(
                run_id=run_id,
                stage="challenge_parallel",
                status="completed",
                meta={"finding_count": len(findings)},
            )
            self.repository.save_findings(run_id, findings)

            synthesis, synthesis_providers = await synthesize_idea(
                self.router,
                request.idea,
                nodes,
                findings,
                settings,
            )
            self.repository.log_stage(
                run_id=run_id,
                stage="synthesis",
                status="completed",
                provider=",".join(synthesis_providers),
            )
            self.repository.save_synthesis(
                run_id=run_id,
                synthesized_idea=synthesis["synthesized_idea"],
                changes_summary=synthesis["changes_summary"],
                residual_risks=synthesis["residual_risks"],
                next_experiments=synthesis["next_experiments"],
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
