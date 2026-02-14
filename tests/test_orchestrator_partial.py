from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import rim.core.orchestrator as orchestrator_module
from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, CriticFinding, DecompositionNode
from rim.providers.base import StageExecutionError
from rim.storage.repo import RunRepository


class DummyProviderSession:
    def get_usage_meta(self) -> dict:
        return {
            "run_id": "dummy",
            "usage": {
                "calls": 0,
                "latency_ms": 0,
                "tokens": 0,
                "estimated_cost_usd": 0.0,
            },
            "budget": {
                "max_calls": 0,
                "max_latency_ms": 0,
                "max_tokens": 0,
                "max_estimated_cost_usd": 0.0,
            },
        }


class DummyRouter:
    def create_session(self, run_id: str) -> DummyProviderSession:
        return DummyProviderSession()


def test_orchestrator_marks_partial_on_synthesis_failure(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    root = DecompositionNode(
        depth=0,
        component_text="Idea A",
        node_type="claim",
        confidence=0.4,
    )

    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        return [root], "codex", {"stop_reason": "max_depth"}

    async def fake_critics(*args, **kwargs):  # noqa: ANN001, ANN202
        return [
            CriticFinding(
                node_id=root.id,
                critic_type="logic",
                issue="test issue",
                severity="high",
                confidence=0.8,
                suggested_fix="test fix",
                provider="codex",
            )
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        raise StageExecutionError(
            stage="synthesis",
            provider="claude",
            message="provider timeout",
            retryable=True,
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea A", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.run_id == run_id
    assert result.synthesized_idea == "Idea A"
    assert result.confidence_score == 0.25
    assert result.residual_risks
    assert "Synthesis stage failed" in result.residual_risks[0]

    run = orchestrator.get_run(run_id)
    assert run is not None
    assert run.status == "partial"
    assert run.error is not None
    assert run.error.stage == "synthesis"
    assert run.error.provider == "claude"
    assert run.error.retryable is True
    assert run.error_summary == "provider timeout"

    logs = orchestrator.get_run_logs(run_id).logs
    synth_log = next(item for item in logs if item.stage == "synthesis")
    assert synth_log.status == "failed"
    assert synth_log.meta["error"]["retryable"] is True


def test_orchestrator_failed_run_keeps_structured_error(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def failing_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        raise StageExecutionError(
            stage="decompose",
            provider="codex",
            message="temporary timeout",
            retryable=True,
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", failing_decompose)

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_fail.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea B", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    with pytest.raises(StageExecutionError):
        asyncio.run(orchestrator.execute_run(run_id, request))

    run = orchestrator.get_run(run_id)
    assert run is not None
    assert run.status == "failed"
    assert run.error is not None
    assert run.error.stage == "decompose"
    assert run.error.provider == "codex"
    assert run.error.retryable is True
    assert run.error_summary == "temporary timeout"
