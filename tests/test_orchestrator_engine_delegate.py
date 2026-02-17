from __future__ import annotations

import asyncio
from pathlib import Path

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, AnalyzeResult
from rim.storage.repo import RunRepository


class StubEngine:
    def __init__(self) -> None:
        self.called_with: tuple[str, AnalyzeRequest] | None = None

    async def execute_run(self, run_id: str, request: AnalyzeRequest) -> AnalyzeResult:
        self.called_with = (run_id, request)
        return AnalyzeResult(
            run_id=run_id,
            mode=request.mode,
            input_idea=request.idea,
            decomposition=[],
            critic_findings=[],
            synthesized_idea="stubbed",
            changes_summary=[],
            residual_risks=[],
            next_experiments=[],
            confidence_score=0.5,
        )


class DummyRouter:
    def create_session(self, run_id: str):  # noqa: ANN001, ANN201
        raise AssertionError("custom engine path should not call router")


def test_orchestrator_delegates_execution_to_injected_engine(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_delegate.db")
    engine = StubEngine()
    orchestrator = RimOrchestrator(
        repository=repo,
        router=DummyRouter(),  # type: ignore[arg-type]
        engine=engine,  # type: ignore[arg-type]
    )
    request = AnalyzeRequest(idea="delegate", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "stubbed"
    assert engine.called_with is not None
    assert engine.called_with[0] == run_id
    assert engine.called_with[1].idea == "delegate"
