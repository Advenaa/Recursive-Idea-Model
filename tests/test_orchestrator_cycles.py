from __future__ import annotations

import asyncio
from pathlib import Path

import rim.core.orchestrator as orchestrator_module
from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, CriticFinding, DecompositionNode
from rim.storage.repo import RunRepository


class DummyProviderSession:
    async def invoke_json(self, stage: str, prompt: str, json_schema=None):  # noqa: ANN001, ANN201
        if stage not in {"critic_arbitration", "critic_arbitration_devil"}:
            raise AssertionError("unexpected stage")
        if stage == "critic_arbitration_devil":
            return (
                {
                    "node_id": "n1",
                    "resolved_issue": "Merge disagreement into one prioritized blocker.",
                    "rationale": "Devil pass confirms overlap between concerns.",
                    "action": "merge",
                    "confidence": 0.91,
                },
                "claude",
            )
        return (
            {
                "node_id": "n1",
                "resolved_issue": "Merge disagreements into one prioritized risk.",
                "rationale": "Concerns overlap and should be addressed together.",
                "action": "merge",
                "confidence": 0.8,
            },
            "codex",
        )

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
    fold_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "memory_fold"
    ]
    assert len(fold_logs) == 1
    assert fold_logs[0].meta["cycle"] == 1


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


def test_orchestrator_runs_disagreement_arbitration(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea D",
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
                issue="No rollout criteria",
                severity="high",
                confidence=0.8,
                suggested_fix="Define criteria",
                provider="codex",
            ),
            CriticFinding(
                node_id=nodes[0].id,
                critic_type="execution",
                issue="Unclear deployment gating",
                severity="high",
                confidence=0.7,
                suggested_fix="Add gates",
                provider="claude",
            ),
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea D refined",
                "changes_summary": ["Added rollout gating rules."],
                "residual_risks": [],
                "next_experiments": ["Test rollout gates on staging."],
                "confidence_score": 0.82,
            },
            ["claude"],
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ENABLE_DISAGREEMENT_ARBITRATION", "1")
    monkeypatch.setenv("RIM_ARBITRATION_MAX_JOBS", "2")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_arbitration.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea D", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea D refined"

    arbitration_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "challenge_arbitration"
    ]
    assert len(arbitration_logs) == 1
    assert arbitration_logs[0].status == "completed"
    assert arbitration_logs[0].meta["resolved_count"] >= 1
    reconciliation_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "challenge_reconciliation"
    ]
    assert len(reconciliation_logs) == 1
    assert "diversity_flagged_count" in reconciliation_logs[0].meta


def test_orchestrator_runs_devils_advocate_arbitration_round(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea D2",
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
                issue="No rollout criteria",
                severity="high",
                confidence=0.8,
                suggested_fix="Define criteria",
                provider="codex",
            ),
            CriticFinding(
                node_id=nodes[0].id,
                critic_type="execution",
                issue="Unclear deployment gating",
                severity="high",
                confidence=0.7,
                suggested_fix="Add gates",
                provider="claude",
            ),
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea D2 refined",
                "changes_summary": ["Added rollout gating rules."],
                "residual_risks": [],
                "next_experiments": ["Test rollout gates on staging."],
                "confidence_score": 0.82,
            },
            ["claude"],
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ENABLE_DISAGREEMENT_ARBITRATION", "1")
    monkeypatch.setenv("RIM_ARBITRATION_MAX_JOBS", "2")
    monkeypatch.setenv("RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION", "1")
    monkeypatch.setenv("RIM_DEVILS_ADVOCATE_ROUNDS", "1")
    monkeypatch.setenv("RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE", "0.95")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_devil_arbitration.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea D2", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea D2 refined"

    arbitration_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "challenge_arbitration"
    ]
    assert len(arbitration_logs) == 1
    assert arbitration_logs[0].status == "completed"
    assert arbitration_logs[0].meta["devils_advocate_enabled"] is True
    assert arbitration_logs[0].meta["devils_advocate_rounds"] == 1
    assert arbitration_logs[0].meta["devils_advocate_count"] >= 1


def test_orchestrator_runs_executable_verification(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea E",
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
                issue="Missing concrete plan",
                severity="high",
                confidence=0.75,
                suggested_fix="Add plan",
                provider="codex",
            ),
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea E refined",
                "changes_summary": ["Added one change."],
                "residual_risks": [],
                "next_experiments": ["Run one pilot."],
                "confidence_score": 0.7,
            },
            ["claude"],
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ENABLE_VERIFICATION", "0")
    monkeypatch.setenv("RIM_ENABLE_EXECUTABLE_VERIFICATION", "1")
    monkeypatch.setenv("RIM_EXEC_VERIFY_MAX_CHECKS", "3")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_exec_verify.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(
        idea="Idea E",
        mode="deep",
        constraints=["python: change_count >= 2"],
    )
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.confidence_score < 0.7
    assert any("Executable verification failed" in risk for risk in result.residual_risks)

    logs = orchestrator.get_run_logs(run_id).logs
    exec_logs = [log for log in logs if log.stage == "verification_executable"]
    assert len(exec_logs) == 1
    assert exec_logs[0].meta["failed_checks"] == 1


def test_orchestrator_runs_python_exec_checks_when_enabled(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea F",
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
                issue="Need at least one change",
                severity="medium",
                confidence=0.7,
                suggested_fix="Add change",
                provider="codex",
            ),
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea F refined",
                "changes_summary": ["Added one change"],
                "residual_risks": [],
                "next_experiments": ["Run pilot."],
                "confidence_score": 0.66,
            },
            ["claude"],
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ENABLE_VERIFICATION", "0")
    monkeypatch.setenv("RIM_ENABLE_EXECUTABLE_VERIFICATION", "1")
    monkeypatch.setenv("RIM_ENABLE_PYTHON_EXEC_CHECKS", "1")
    monkeypatch.setenv("RIM_EXEC_VERIFY_MAX_CHECKS", "3")
    monkeypatch.setenv("RIM_PYTHON_EXEC_TIMEOUT_SEC", "2")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_python_exec.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(
        idea="Idea F",
        mode="deep",
        constraints=["python_exec: passed = context['change_count'] >= 1"],
    )
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.confidence_score == 0.66

    logs = orchestrator.get_run_logs(run_id).logs
    exec_logs = [log for log in logs if log.stage == "verification_executable"]
    assert len(exec_logs) == 1
    assert exec_logs[0].meta["failed_checks"] == 0
    assert exec_logs[0].meta["python_exec_enabled"] is True


def test_orchestrator_logs_specialization_spawn_plan(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    captured_extra: list[tuple[str, str]] = []

    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea G",
            node_type="claim",
            confidence=0.4,
        )
        return [root], "codex", {"stop_reason": "max_depth"}

    async def fake_critics(*args, **kwargs):  # noqa: ANN001, ANN202
        captured_extra.extend(list(kwargs.get("extra_critics") or []))
        nodes = kwargs.get("nodes") or args[1]
        return [
            CriticFinding(
                node_id=nodes[0].id,
                critic_type="logic",
                issue="base issue",
                severity="medium",
                confidence=0.7,
                suggested_fix="fix",
                provider="codex",
            )
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea G refined",
                "changes_summary": ["change"],
                "residual_risks": [],
                "next_experiments": ["test"],
                "confidence_score": 0.75,
            },
            ["claude"],
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ENABLE_VERIFICATION", "0")
    monkeypatch.setenv("RIM_ENABLE_EXECUTABLE_VERIFICATION", "0")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_spawn_plan.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(
        idea="Idea G",
        mode="deep",
        constraints=["Need security and low latency"],
    )
    run_id = orchestrator.create_run(request, status="running")

    _result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert any(stage in {"critic_security", "critic_scalability"} for stage, _ in captured_extra)

    logs = orchestrator.get_run_logs(run_id).logs
    spawn_logs = [log for log in logs if log.stage == "specialization_spawn"]
    assert len(spawn_logs) == 1
    assert spawn_logs[0].meta["selected_count"] >= 1
