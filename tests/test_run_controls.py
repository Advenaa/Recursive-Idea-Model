from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import HTTPException, Response

import rim.api.app as api_app
from rim.api.job_queue import RunJobQueue
from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import CriticFinding, DecompositionNode
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def _count_rows(repo: RunRepository, table: str, run_id: str) -> int:
    row = repo.conn.execute(
        f"SELECT COUNT(*) AS count FROM {table} WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return int(row["count"])


def test_queue_cancel_marks_run_canceled(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_run_cancel.db")
    repo.create_run_with_request(
        run_id="run-cancel",
        mode="deep",
        input_idea="idea",
        request_json='{"idea":"idea","mode":"deep"}',
        status="queued",
    )
    queue = RunJobQueue(
        orchestrator=RimOrchestrator(repository=repo, router=ProviderRouter()),
        repository=repo,
    )

    run = asyncio.run(queue.cancel("run-cancel"))
    assert run is not None
    assert run.status == "canceled"
    assert run.error_summary == "Run canceled by user request."
    assert run.error is not None
    assert run.error.retryable is True

    logs = repo.get_stage_logs("run-cancel")
    assert any(item["stage"] == "queue" and item["status"] == "canceled" for item in logs)


def test_queue_retry_resets_run_artifacts_and_requeues(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_run_retry.db")
    run_id = "run-retry"
    repo.create_run_with_request(
        run_id=run_id,
        mode="deep",
        input_idea="idea",
        request_json='{"idea":"idea","mode":"deep"}',
        status="failed",
    )

    node = DecompositionNode(depth=0, component_text="root")
    finding = CriticFinding(
        node_id=node.id,
        critic_type="logic",
        issue="bad assumption",
        suggested_fix="replace assumption",
    )
    repo.save_nodes(run_id, [node])
    repo.save_findings(run_id, [finding])
    repo.save_synthesis(
        run_id=run_id,
        synthesized_idea="old synthesis",
        changes_summary=["a"],
        residual_risks=["b"],
        next_experiments=["c"],
    )
    repo.save_memory_entries(
        run_id=run_id,
        entries=[{"entry_type": "insight", "entry_text": "memo", "score": 0.7}],
    )
    repo.submit_run_feedback(run_id=run_id, verdict="accept", notes="nice")
    repo.log_stage(run_id=run_id, stage="pipeline", status="failed")
    repo.mark_run_status(run_id=run_id, status="failed", error_summary="broken")

    queue = RunJobQueue(
        orchestrator=RimOrchestrator(repository=repo, router=ProviderRouter()),
        repository=repo,
    )
    run = asyncio.run(queue.retry(run_id))
    assert run is not None
    assert run.status == "queued"
    assert run.result is None
    assert run.error_summary is None
    assert run_id in queue.pending_ids
    assert _count_rows(repo, "nodes", run_id) == 0
    assert _count_rows(repo, "critic_findings", run_id) == 0
    assert _count_rows(repo, "synthesis_outputs", run_id) == 0
    assert _count_rows(repo, "memory_entries", run_id) == 0
    assert _count_rows(repo, "run_feedback", run_id) == 0
    assert _count_rows(repo, "stage_logs", run_id) == 0


def test_queue_retry_rejects_non_retryable_status(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_run_retry_reject.db")
    repo.create_run_with_request(
        run_id="run-done",
        mode="deep",
        input_idea="idea",
        request_json='{"idea":"idea","mode":"deep"}',
        status="completed",
    )
    queue = RunJobQueue(
        orchestrator=RimOrchestrator(repository=repo, router=ProviderRouter()),
        repository=repo,
    )
    try:
        asyncio.run(queue.retry("run-done"))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "not retryable" in str(exc)


def test_queue_retry_rejects_missing_request_payload(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_run_retry_missing_request.db")
    repo.create_run_with_request(
        run_id="run-missing-request",
        mode="deep",
        input_idea="idea",
        request_json=None,
        status="failed",
    )
    queue = RunJobQueue(
        orchestrator=RimOrchestrator(repository=repo, router=ProviderRouter()),
        repository=repo,
    )
    try:
        asyncio.run(queue.retry("run-missing-request"))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "request payload is missing or invalid" in str(exc)


def test_api_cancel_and_retry_controls(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_api_run_controls.db")
    repo.create_run_with_request(
        run_id="run-api-control",
        mode="deep",
        input_idea="idea",
        request_json='{"idea":"idea","mode":"deep"}',
        status="queued",
    )
    orchestrator = RimOrchestrator(repository=repo, router=ProviderRouter())
    queue = RunJobQueue(orchestrator=orchestrator, repository=repo)

    previous_orchestrator = api_app.orchestrator
    previous_job_queue = api_app.job_queue
    api_app.orchestrator = orchestrator
    api_app.job_queue = queue
    try:
        canceled = asyncio.run(api_app.cancel_run("run-api-control"))
        assert canceled.status == "canceled"

        response = Response()
        retried = asyncio.run(api_app.retry_run("run-api-control", response=response))
        assert response.status_code == 202
        assert retried.status == "queued"
    finally:
        api_app.orchestrator = previous_orchestrator
        api_app.job_queue = previous_job_queue


def test_api_retry_rejects_completed_run(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_api_run_controls_reject.db")
    repo.create_run_with_request(
        run_id="run-completed",
        mode="deep",
        input_idea="idea",
        request_json='{"idea":"idea","mode":"deep"}',
        status="completed",
    )
    orchestrator = RimOrchestrator(repository=repo, router=ProviderRouter())
    queue = RunJobQueue(orchestrator=orchestrator, repository=repo)

    previous_orchestrator = api_app.orchestrator
    previous_job_queue = api_app.job_queue
    api_app.orchestrator = orchestrator
    api_app.job_queue = queue
    try:
        try:
            asyncio.run(api_app.retry_run("run-completed", response=Response()))
            raise AssertionError("expected HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 409
            assert "not retryable" in str(exc.detail)
    finally:
        api_app.orchestrator = previous_orchestrator
        api_app.job_queue = previous_job_queue
