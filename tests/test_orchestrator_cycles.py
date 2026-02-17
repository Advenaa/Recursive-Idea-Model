from __future__ import annotations

import asyncio
import json
from pathlib import Path

import rim.core.orchestrator as orchestrator_module
from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, CriticFinding, DecompositionNode
from rim.storage.repo import RunRepository


class DummyProviderSession:
    async def invoke_json(self, stage: str, prompt: str, json_schema=None):  # noqa: ANN001, ANN201
        if stage not in {
            "critic_arbitration",
            "critic_arbitration_devil",
            "critic_arbitration_specialist",
        }:
            raise AssertionError("unexpected stage")
        if stage == "critic_arbitration_devil":
            return (
                {
                    "node_id": "",
                    "resolved_issue": "Merge disagreement into one prioritized blocker.",
                    "rationale": "Devil pass confirms overlap between concerns.",
                    "action": "merge",
                    "confidence": 0.91,
                },
                "claude",
            )
        if stage == "critic_arbitration_specialist":
            return (
                {
                    "node_id": "",
                    "resolved_issue": "Specialist review confirms rollout blocker",
                    "rationale": "Specialist loop adds stronger domain-aware arbitration.",
                    "action": "merge",
                    "confidence": 0.94,
                },
                "codex",
            )
        return (
            {
                "node_id": "",
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
    assert fold_logs[0].meta["fold_version"] == "v2"
    assert "degradation_detected" in fold_logs[0].meta


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


def test_orchestrator_runs_specialist_arbitration_loop_on_diversity_flags(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea D3",
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
                "synthesized_idea": "Idea D3 refined",
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
    monkeypatch.setenv("RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION", "0")
    monkeypatch.setenv("RIM_RECONCILE_MIN_UNIQUE_CRITICS", "3")
    monkeypatch.setenv("RIM_RECONCILE_MAX_SINGLE_CRITIC_SHARE", "0.7")
    monkeypatch.setenv("RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP", "1")
    monkeypatch.setenv("RIM_SPECIALIST_ARBITRATION_MAX_JOBS", "2")
    monkeypatch.setenv("RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE", "0.95")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_specialist_arbitration.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea D3", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea D3 refined"

    arbitration_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "challenge_arbitration"
    ]
    assert len(arbitration_logs) == 1
    assert arbitration_logs[0].status == "completed"
    assert arbitration_logs[0].meta["specialist_loop_enabled"] is True
    assert arbitration_logs[0].meta["specialist_count"] >= 1


def test_orchestrator_applies_specialist_policy_file(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    captured: dict[str, object] = {}

    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea D4",
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

    async def fake_run_arbitration(  # noqa: ANN001, ANN202
        _router,
        *,
        reconciliation,
        findings,
        max_jobs,
        devils_advocate_rounds,
        devils_advocate_min_confidence,
        specialist_loop_enabled,
        specialist_max_jobs,
        specialist_min_confidence,
        specialist_contracts,
    ):
        captured["max_jobs"] = max_jobs
        captured["devils_advocate_rounds"] = devils_advocate_rounds
        captured["devils_advocate_min_confidence"] = devils_advocate_min_confidence
        captured["specialist_loop_enabled"] = specialist_loop_enabled
        captured["specialist_max_jobs"] = specialist_max_jobs
        captured["specialist_min_confidence"] = specialist_min_confidence
        captured["specialist_contract_count"] = len(list(specialist_contracts or []))
        captured["disagreement_count"] = int(reconciliation.get("summary", {}).get("disagreement_count", 0))
        captured["finding_count"] = len(findings)
        return (
            [
                {
                    "node_id": findings[0].node_id,
                    "resolved_issue": "Specialist policy-driven arbitration",
                    "rationale": "Policy selected specialist review depth.",
                    "action": "merge",
                    "confidence": 0.91,
                    "round": "specialist",
                }
            ],
            ["codex"],
        )

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea D4 refined",
                "changes_summary": ["Added rollout gating rules."],
                "residual_risks": [],
                "next_experiments": ["Test rollout gates on staging."],
                "confidence_score": 0.82,
            },
            ["claude"],
        )

    policy_path = tmp_path / "specialist_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy": {
                    "policy_env": {
                        "RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP": 1,
                        "RIM_SPECIALIST_ARBITRATION_MAX_JOBS": 4,
                        "RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE": 0.91,
                        "RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER": 1,
                        "RIM_SPECIALIST_CONTRACT_LOOKBACK_RUNS": 18,
                        "RIM_SPECIALIST_CONTRACT_MIN_ROUNDS": 5,
                        "RIM_SPECIALIST_CONTRACT_MIN_ROLE_SAMPLES": 3,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "run_arbitration", fake_run_arbitration)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ENABLE_DISAGREEMENT_ARBITRATION", "1")
    monkeypatch.setenv("RIM_ARBITRATION_MAX_JOBS", "2")
    monkeypatch.setenv("RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION", "0")
    monkeypatch.setenv("RIM_RECONCILE_MIN_UNIQUE_CRITICS", "3")
    monkeypatch.setenv("RIM_RECONCILE_MAX_SINGLE_CRITIC_SHARE", "0.7")
    monkeypatch.setenv("RIM_SPECIALIST_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP", raising=False)
    monkeypatch.delenv("RIM_SPECIALIST_ARBITRATION_MAX_JOBS", raising=False)
    monkeypatch.delenv("RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER", raising=False)
    monkeypatch.delenv("RIM_SPECIALIST_CONTRACT_LOOKBACK_RUNS", raising=False)
    monkeypatch.delenv("RIM_SPECIALIST_CONTRACT_MIN_ROUNDS", raising=False)
    monkeypatch.delenv("RIM_SPECIALIST_CONTRACT_MIN_ROLE_SAMPLES", raising=False)

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_specialist_policy.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea D4", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea D4 refined"
    assert captured["specialist_loop_enabled"] is True
    assert captured["specialist_max_jobs"] == 4
    assert captured["specialist_min_confidence"] == 0.91

    arbitration_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "challenge_arbitration"
    ]
    assert len(arbitration_logs) == 1
    assert arbitration_logs[0].meta["specialist_policy_applied"] is True
    assert arbitration_logs[0].meta["specialist_policy_path"] == str(policy_path)
    queue_logs = [log for log in orchestrator.get_run_logs(run_id).logs if log.stage == "queue"]
    assert len(queue_logs) == 1
    assert queue_logs[0].meta["specialist_contract_controller_enabled"] is True
    assert queue_logs[0].meta["specialist_contract_lookback_runs"] == 18
    assert queue_logs[0].meta["specialist_contract_min_rounds"] == 5
    assert queue_logs[0].meta["specialist_contract_min_role_samples"] == 3


def test_orchestrator_applies_arbitration_policy_file(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    captured: dict[str, object] = {}

    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea D4A",
            node_type="claim",
            confidence=0.35,
        )
        return [root], "codex", {"stop_reason": "max_depth"}

    async def fake_critics(*args, **kwargs):  # noqa: ANN001, ANN202
        nodes = kwargs.get("nodes") or args[1]
        return [
            CriticFinding(
                node_id=nodes[0].id,
                critic_type="logic",
                issue="No rollback criteria",
                severity="high",
                confidence=0.85,
                suggested_fix="Add rollback gates",
                provider="codex",
            ),
            CriticFinding(
                node_id=nodes[0].id,
                critic_type="evidence",
                issue="No pilot evidence",
                severity="high",
                confidence=0.82,
                suggested_fix="Run pilot",
                provider="claude",
            ),
        ]

    async def fake_run_arbitration(  # noqa: ANN001, ANN202
        _router,
        *,
        reconciliation,
        findings,
        max_jobs,
        devils_advocate_rounds,
        devils_advocate_min_confidence,
        specialist_loop_enabled,
        specialist_max_jobs,
        specialist_min_confidence,
        specialist_contracts,
    ):
        captured["max_jobs"] = max_jobs
        captured["devils_advocate_rounds"] = devils_advocate_rounds
        captured["devils_advocate_min_confidence"] = devils_advocate_min_confidence
        captured["specialist_loop_enabled"] = specialist_loop_enabled
        captured["specialist_contract_count"] = len(list(specialist_contracts or []))
        captured["disagreement_count"] = int(reconciliation.get("summary", {}).get("disagreement_count", 0))
        captured["finding_count"] = len(findings)
        return (
            [
                {
                    "node_id": findings[0].node_id,
                    "resolved_issue": "Policy-driven arbitration",
                    "rationale": "Arbitration policy requested deeper challenge loop.",
                    "action": "revise",
                    "confidence": 0.87,
                    "round": "devil_1",
                }
            ],
            ["codex"],
        )

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea D4A refined",
                "changes_summary": ["Added rollback guardrails."],
                "residual_risks": [],
                "next_experiments": ["Dry-run rollback on staging."],
                "confidence_score": 0.79,
            },
            ["claude"],
        )

    policy_path = tmp_path / "arbitration_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy": {
                    "policy_env": {
                        "RIM_ENABLE_DISAGREEMENT_ARBITRATION": 1,
                        "RIM_ARBITRATION_MAX_JOBS": 5,
                        "RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION": 1,
                        "RIM_DEVILS_ADVOCATE_ROUNDS": 2,
                        "RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE": 0.66,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "run_arbitration", fake_run_arbitration)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ARBITRATION_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("RIM_ENABLE_DISAGREEMENT_ARBITRATION", raising=False)
    monkeypatch.delenv("RIM_ARBITRATION_MAX_JOBS", raising=False)
    monkeypatch.delenv("RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION", raising=False)
    monkeypatch.delenv("RIM_DEVILS_ADVOCATE_ROUNDS", raising=False)
    monkeypatch.delenv("RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE", raising=False)
    monkeypatch.setenv("RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP", "0")
    monkeypatch.setenv("RIM_RECONCILE_MIN_UNIQUE_CRITICS", "3")
    monkeypatch.setenv("RIM_RECONCILE_MAX_SINGLE_CRITIC_SHARE", "0.7")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_arbitration_policy.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea D4A", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea D4A refined"
    assert captured["max_jobs"] == 5
    assert captured["devils_advocate_rounds"] == 2
    assert captured["devils_advocate_min_confidence"] == 0.66

    logs = orchestrator.get_run_logs(run_id).logs
    queue_logs = [log for log in logs if log.stage == "queue"]
    assert len(queue_logs) == 1
    assert queue_logs[0].meta["arbitration_policy_applied"] is True
    assert queue_logs[0].meta["arbitration_policy_path"] == str(policy_path)
    arbitration_logs = [log for log in logs if log.stage == "challenge_arbitration"]
    assert len(arbitration_logs) == 1
    assert arbitration_logs[0].meta["arbitration_policy_applied"] is True
    assert arbitration_logs[0].meta["arbitration_policy_path"] == str(policy_path)


def test_orchestrator_applies_depth_policy_file(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea D5",
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
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea D5 refined",
                "changes_summary": ["Added rollout gating rules."],
                "residual_risks": [],
                "next_experiments": ["Test rollout gates on staging."],
                "confidence_score": 0.82,
            },
            ["claude"],
        )

    policy_path = tmp_path / "depth_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy": {
                    "policy_env": {
                        "RIM_MAX_ANALYSIS_CYCLES": 2,
                        "RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE": 0.89,
                        "RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS": 1,
                        "RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setenv("RIM_DEPTH_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("RIM_MAX_ANALYSIS_CYCLES", raising=False)
    monkeypatch.delenv("RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS", raising=False)
    monkeypatch.delenv("RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS", raising=False)

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_depth_policy.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea D5", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea D5 refined"

    depth_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "depth_allocator"
    ]
    assert len(depth_logs) >= 1
    latest = depth_logs[-1]
    assert latest.meta["depth_policy_applied"] is True
    assert latest.meta["depth_policy_path"] == str(policy_path)
    assert latest.meta["max_cycles"] == 2
    assert latest.meta["min_confidence_to_stop"] == 0.89
    assert latest.meta["max_residual_risks_to_stop"] == 1
    assert latest.meta["max_high_findings_to_stop"] == 0


def test_orchestrator_applies_memory_policy_file(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    state = {"cycle": 0}
    captured: dict[str, object] = {}

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
                    "synthesized_idea": "Idea M refined once",
                    "changes_summary": ["narrow scope"],
                    "residual_risks": ["open risk"],
                    "next_experiments": ["run pilot"],
                    "confidence_score": 0.55,
                },
                ["claude"],
            )
        return (
            {
                "synthesized_idea": "Idea M refined twice",
                "changes_summary": ["close open risk"],
                "residual_risks": [],
                "next_experiments": ["ship controlled rollout"],
                "confidence_score": 0.91,
            },
            ["claude"],
        )

    def fake_fold_cycle_memory(*args, **kwargs):  # noqa: ANN001, ANN202
        captured["max_entries"] = kwargs["max_entries"]
        captured["novelty_floor"] = kwargs["novelty_floor"]
        captured["max_duplicate_ratio"] = kwargs["max_duplicate_ratio"]
        return {
            "fold_version": "v2",
            "folded_context": ["Folded context A"],
            "episodic": ["Episodic A"],
            "working": ["Working A"],
            "tool": ["Tool A"],
            "quality": {
                "degradation_detected": False,
                "degradation_reasons": [],
                "novelty_ratio": 0.45,
                "duplicate_ratio": 0.2,
            },
        }

    policy_path = tmp_path / "memory_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy": {
                    "policy_env": {
                        "RIM_ENABLE_MEMORY_FOLDING": 1,
                        "RIM_MEMORY_FOLD_MAX_ENTRIES": 20,
                        "RIM_MEMORY_FOLD_NOVELTY_FLOOR": 0.55,
                        "RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO": 0.3,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setattr(orchestrator_module, "fold_cycle_memory", fake_fold_cycle_memory)
    monkeypatch.setattr(orchestrator_module, "fold_to_memory_entries", lambda *args, **kwargs: [])
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "2")
    monkeypatch.setenv("RIM_MEMORY_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("RIM_ENABLE_MEMORY_FOLDING", raising=False)
    monkeypatch.delenv("RIM_MEMORY_FOLD_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("RIM_MEMORY_FOLD_NOVELTY_FLOOR", raising=False)
    monkeypatch.delenv("RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO", raising=False)

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_memory_policy.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea M", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea M refined twice"
    assert captured["max_entries"] == 20
    assert captured["novelty_floor"] == 0.55
    assert captured["max_duplicate_ratio"] == 0.3

    fold_logs = [
        log
        for log in orchestrator.get_run_logs(run_id).logs
        if log.stage == "memory_fold"
    ]
    assert len(fold_logs) == 1
    assert fold_logs[0].meta["memory_policy_applied"] is True
    assert fold_logs[0].meta["memory_policy_path"] == str(policy_path)


def test_orchestrator_memory_quality_controller_tightens_fold_parameters(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    captured: dict[str, float] = {}
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
                    "synthesized_idea": "Idea Q refined once",
                    "changes_summary": ["narrow scope"],
                    "residual_risks": ["open risk"],
                    "next_experiments": ["run pilot"],
                    "confidence_score": 0.55,
                },
                ["claude"],
            )
        return (
            {
                "synthesized_idea": "Idea Q refined twice",
                "changes_summary": ["close open risk"],
                "residual_risks": [],
                "next_experiments": ["ship controlled rollout"],
                "confidence_score": 0.91,
            },
            ["claude"],
        )

    def fake_fold_cycle_memory(*args, **kwargs):  # noqa: ANN001, ANN202
        captured["max_entries"] = float(kwargs["max_entries"])
        captured["novelty_floor"] = float(kwargs["novelty_floor"])
        captured["max_duplicate_ratio"] = float(kwargs["max_duplicate_ratio"])
        return {
            "fold_version": "v2",
            "folded_context": ["Folded context Q"],
            "episodic": ["Episodic Q"],
            "working": ["Working Q"],
            "tool": ["Tool Q"],
            "quality": {
                "degradation_detected": False,
                "degradation_reasons": [],
                "novelty_ratio": 0.5,
                "duplicate_ratio": 0.2,
            },
        }

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setattr(orchestrator_module, "fold_cycle_memory", fake_fold_cycle_memory)
    monkeypatch.setattr(orchestrator_module, "fold_to_memory_entries", lambda *args, **kwargs: [])
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "2")
    monkeypatch.setenv("RIM_ENABLE_MEMORY_QUALITY_CONTROLLER", "1")
    monkeypatch.setenv("RIM_MEMORY_QUALITY_LOOKBACK_RUNS", "8")
    monkeypatch.setenv("RIM_MEMORY_QUALITY_MIN_FOLDS", "2")
    monkeypatch.delenv("RIM_MEMORY_POLICY_PATH", raising=False)
    monkeypatch.delenv("RIM_ENABLE_MEMORY_FOLDING", raising=False)
    monkeypatch.delenv("RIM_MEMORY_FOLD_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("RIM_MEMORY_FOLD_NOVELTY_FLOOR", raising=False)
    monkeypatch.delenv("RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO", raising=False)

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_memory_quality.db")
    for index in range(3):
        historical_run = f"hist-{index + 1}"
        repo.create_run_with_request(
            run_id=historical_run,
            mode="deep",
            input_idea=f"historical-{index + 1}",
            request_json=json.dumps(
                {"idea": f"historical-{index + 1}", "mode": "deep"},
            ),
            status="completed",
        )
        repo.log_stage(
            run_id=historical_run,
            stage="memory_fold",
            status="completed",
            meta={
                "degradation_detected": True,
                "novelty_ratio": 0.2,
                "duplicate_ratio": 0.75,
            },
        )

    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea Q", mode="deep")
    run_id = orchestrator.create_run(request, status="running")
    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea Q refined twice"

    assert captured["max_entries"] < 12.0
    assert captured["novelty_floor"] > 0.35
    assert captured["max_duplicate_ratio"] < 0.5

    queue_logs = [log for log in orchestrator.get_run_logs(run_id).logs if log.stage == "queue"]
    assert len(queue_logs) == 1
    assert queue_logs[0].meta["memory_quality_controller_enabled"] is True
    assert queue_logs[0].meta["memory_quality_controller_applied"] is True
    assert queue_logs[0].meta["memory_quality_fold_count"] >= 3


def test_orchestrator_specialist_contract_controller_adjusts_spawn_boosts(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    captured: dict[str, object] = {}

    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea SC",
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
                issue="Need stronger rollout checklist",
                severity="medium",
                confidence=0.7,
                suggested_fix="Add checklist",
                provider="codex",
            ),
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea SC refined",
                "changes_summary": ["Added checklist"],
                "residual_risks": [],
                "next_experiments": ["Run checklist pilot"],
                "confidence_score": 0.82,
            },
            ["claude"],
        )

    from rim.agents.spawner import build_spawn_plan as real_build_spawn_plan

    def fake_build_spawn_plan(*, mode, domain, constraints, memory_context, adaptive_role_boosts=None, adaptive_meta=None):  # noqa: ANN001, ANN202
        captured["adaptive_role_boosts"] = dict(adaptive_role_boosts or {})
        captured["adaptive_meta"] = dict(adaptive_meta or {})
        return real_build_spawn_plan(
            mode=mode,
            domain=domain,
            constraints=constraints,
            memory_context=memory_context,
            adaptive_role_boosts=adaptive_role_boosts,
            adaptive_meta=adaptive_meta,
        )

    monkeypatch.setattr(orchestrator_module, "decompose_idea", fake_decompose)
    monkeypatch.setattr(orchestrator_module, "run_critics", fake_critics)
    monkeypatch.setattr(orchestrator_module, "synthesize_idea", fake_synthesize)
    monkeypatch.setattr(orchestrator_module, "build_spawn_plan", fake_build_spawn_plan)
    monkeypatch.setenv("RIM_MAX_ANALYSIS_CYCLES", "1")
    monkeypatch.setenv("RIM_ENABLE_DISAGREEMENT_ARBITRATION", "0")
    monkeypatch.setenv("RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER", "1")
    monkeypatch.setenv("RIM_SPECIALIST_CONTRACT_LOOKBACK_RUNS", "8")
    monkeypatch.setenv("RIM_SPECIALIST_CONTRACT_MIN_ROUNDS", "2")
    monkeypatch.setenv("RIM_SPECIALIST_CONTRACT_MIN_ROLE_SAMPLES", "2")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_specialist_contract_controller.db")
    for index, confidence in enumerate([0.9, 0.82], start=1):
        historical_run = f"hist-specialist-{index}"
        repo.create_run_with_request(
            run_id=historical_run,
            mode="deep",
            input_idea=f"historical-specialist-{index}",
            request_json=json.dumps(
                {"idea": f"historical-specialist-{index}", "mode": "deep"},
            ),
            status="completed",
        )
        repo.mark_run_status(
            run_id=historical_run,
            status="completed",
            confidence_score=confidence,
        )
        selected_roles = ["security", "security"] if index == 1 else ["security"]
        repo.log_stage(
            run_id=historical_run,
            stage="challenge_arbitration",
            status="completed",
            meta={
                "specialist_selected_roles": selected_roles,
                "specialist_role_action_counts": {
                    "security": {
                        "merge": len(selected_roles),
                        "escalate": 0,
                        "drop": 0,
                        "total": len(selected_roles),
                    }
                },
                "specialist_role_avg_match_score": {"security": 1.6},
            },
        )

    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(idea="Idea SC", mode="deep")
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.synthesized_idea == "Idea SC refined"
    role_boosts = captured.get("adaptive_role_boosts")
    assert isinstance(role_boosts, dict)
    assert float(role_boosts.get("security", 0.0)) > 0.0

    queue_logs = [log for log in orchestrator.get_run_logs(run_id).logs if log.stage == "queue"]
    assert len(queue_logs) == 1
    assert queue_logs[0].meta["specialist_contract_controller_enabled"] is True
    assert queue_logs[0].meta["specialist_contract_controller_applied"] is True
    assert "security" in queue_logs[0].meta["specialist_contract_roles_adjusted"]

    spawn_logs = [log for log in orchestrator.get_run_logs(run_id).logs if log.stage == "specialization_spawn"]
    assert len(spawn_logs) == 1
    assert spawn_logs[0].meta["adaptive_boosts_applied"] is True
    assert float(spawn_logs[0].meta["role_boosts"]["security"]) > 0.0


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


def test_orchestrator_runs_advanced_verification(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    async def fake_decompose(*args, **kwargs):  # noqa: ANN001, ANN202
        root = DecompositionNode(
            depth=0,
            component_text="Idea H",
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
                issue="Need stronger confidence evidence",
                severity="high",
                confidence=0.75,
                suggested_fix="Add stronger evidence",
                provider="codex",
            ),
        ]

    async def fake_synthesize(*args, **kwargs):  # noqa: ANN001, ANN202
        return (
            {
                "synthesized_idea": "Idea H refined",
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
    monkeypatch.setenv("RIM_ENABLE_EXECUTABLE_VERIFICATION", "0")
    monkeypatch.setenv("RIM_ENABLE_ADVANCED_VERIFICATION", "1")
    monkeypatch.setenv("RIM_ADV_VERIFY_MAX_CHECKS", "3")
    monkeypatch.setenv("RIM_ADV_VERIFY_SIMULATION_TRIALS", "80")
    monkeypatch.setenv("RIM_ADV_VERIFY_SIMULATION_MIN_PASS_RATE", "0.7")

    repo = RunRepository(db_path=tmp_path / "rim_orchestrator_advanced_verify.db")
    orchestrator = RimOrchestrator(repository=repo, router=DummyRouter())  # type: ignore[arg-type]
    request = AnalyzeRequest(
        idea="Idea H",
        mode="deep",
        constraints=["solver: confidence_score >= 0.9"],
    )
    run_id = orchestrator.create_run(request, status="running")

    result = asyncio.run(orchestrator.execute_run(run_id, request))
    assert result.confidence_score < 0.7
    assert any("Advanced verification failed" in risk for risk in result.residual_risks)

    logs = orchestrator.get_run_logs(run_id).logs
    advanced_logs = [log for log in logs if log.stage == "verification_advanced"]
    assert len(advanced_logs) == 1
    assert advanced_logs[0].meta["total_checks"] == 1
    assert advanced_logs[0].meta["failed_checks"] == 1


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
