from __future__ import annotations

import asyncio
from pathlib import Path

import rim.core.orchestrator as orchestrator_module
from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, CriticFinding, DecompositionNode
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


def test_orchestrator_recurse_when_depth_allocator_signals_continue(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    state = {"cycle": 0}

    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        state["cycle"] += 1
        root = DecompositionNode(
            depth=0,
            component_text=str(kwargs.get("idea") or args[1]),
            node_type="claim",
            confidence=0.4,
        )
        return [root], "codex", {"stop_reason": "max_depth"}

    async def fake_critics(*args, **kwargs):  # noqa: ANN001, ANN202
        nodes = kwargs.get("nodes") or args[1]
        severity = "critical" if state["cycle"] == 1 else "medium"
        return [
            CriticFinding(
                node_id=nodes[0].id,
                critic_type="logic",
                issue=f"issue cycle {state['cycle']}",
                severity=severity,
                confidence=0.8,
                suggested_fix="fix",
                provider="codex",
            )
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        if state["cycle"] == 1:
            return (
                {
                    "synthesized_idea": "Idea refined once",
                    "changes_summary": ["narrow scope"],
                    "residual_risks": ["open risk"],
                    "next_experiments": ["run pilot"],
                    "confidence_score": 0.55,
                },
                ["claude"],
            )
        return (
            {
                "synthesized_idea": "Idea refined twice",
                "changes_summary": ["close open risk"],
                "residual_risks": [],
                "next_experiments": ["ship controlled rollout"],
                "confidence_score": 0.91,
            },
            ["claude"],
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "2")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_cycles.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea A", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert state["cycle"] == 2
    assert result.synthesized_idea == "Idea refined twice"
    assert result.confidence_score == 0.91

    depth_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "depth_allocator"
    ]
    assert len(depth_logs) == 2
    assert depth_logs[0].meta["decision"]["recurse"] is True
    assert depth_logs[1].meta["decision"]["reason"] == "max_cycles_reached"


def test_orchestrator_defaults_to_single_cycle(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    state = {"cycle": 0}

    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        state["cycle"] += 1
        root = DecompositionNode(
            depth=0,
            component_text="Idea B",
            node_type="claim",
            confidence=0.4,
        )
        return [root], "codex", {"stop_reason": "max_depth"}

    async def fake_critics(*args, **kwargs):  # noqa: ANN001, ANN202
        nodes = kwargs.get("nodes") or args[1]
        return [
            CriticFinding(
                node_id=nodes[0].id,
                critic_type="logic",
                issue="still risky",
                severity="critical",
                confidence=0.8,
                suggested_fix="fix",
                provider="codex",
            )
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea B refined",
                "changes_summary": ["change"],
                "residual_risks": ["remaining risk"],
                "next_experiments": ["test"],
                "confidence_score": 0.4,
            },
            ["claude"],
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.delenv("RIM_MAX_ANALYSIS_CYCLES", raising=False)

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_single_cycle.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea B", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert state["cycle"] == 1
    assert result.synthesized_idea == "Idea B refined"


def test_orchestrator_verification_penalizes_uncovered_output(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea C",
            node_type="claim",
            confidence=0.4,
        )
        return [root], "codex", {"stop_reason": "max_depth"}

    async def fake_critics(*args, **kwargs):  # noqa: ANN001, ANN202
        nodes = kwargs.get("nodes") or args[1]
        return [
            CriticFinding(
                node_id=nodes[0].id,
                critic_type="evidence",
                issue="Missing regulatory compliance controls",
                severity="critical",
                confidence=0.85,
                suggested_fix="Add compliance controls",
                provider="claude",
            )
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Run a very small pilot first.",
                "changes_summary": ["Reduced rollout size."],
                "residual_risks": [],
                "next_experiments": ["Recruit ten users."],
                "confidence_score": 0.9,
            },
            ["claude"],
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ENABLE_VERIFICATION", "1")
    monkeypatch.setenv("RIM_VERIFY_MIN_CONSTRAINT_OVERLAP", "0.9")
    monkeypatch.setenv("RIM_VERIFY_MIN_FINDING_OVERLAP", "0.7")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_verification.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(
        idea="Idea C",
        mode="deep",
        constraints=["Include regulatory compliance controls"],
    )
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.confidence_score < 0.9
    assert any("Verification check failed" in risk for risk in result.residual_risks)

    verification_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "verification"
    ]
    assert len(verification_logs) == 1
    assert verification_logs[0].meta["failed_checks"] >= 1
