from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import HTTPException, Response

import rim.api.app as api_app
from rim.api.job_queue import RunJobQueue
from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def test_analyze_idempotent_run_id_returns_existing_state(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_api_idempotent.db")
    orchestrator = RimOrchestrator(repository=repo, router=ProviderRouter())
    queue = RunJobQueue(orchestrator=orchestrator, repository=repo)

    previous_orchestrator = api_app.orchestrator
    previous_job_queue = api_app.job_queue
    api_app.orchestrator = orchestrator
    api_app.job_queue = queue
    try:
        request = AnalyzeRequest(idea="idempotent idea", mode="deep")
        first_response = Response()
        first = asyncio.run(
            api_app.analyze(
                request=request,
                response=first_response,
                wait=False,
                run_id="fixed-run-1",
            )
        )
        assert first_response.status_code == 202
        assert first.run_id == "fixed-run-1"
        assert first.status == "queued"

        second_response = Response()
        second = asyncio.run(
            api_app.analyze(
                request=request,
                response=second_response,
                wait=False,
                run_id="fixed-run-1",
            )
        )
        assert second_response.status_code == 200
        assert second.run_id == "fixed-run-1"
        assert second.status == "queued"

        runs = repo.list_runs(limit=10, offset=0)
        assert len(runs) == 1
        assert runs[0]["id"] == "fixed-run-1"
    finally:
        api_app.orchestrator = previous_orchestrator
        api_app.job_queue = previous_job_queue


def test_analyze_idempotent_run_id_rejects_request_mismatch(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_api_idempotent_conflict.db")
    orchestrator = RimOrchestrator(repository=repo, router=ProviderRouter())
    queue = RunJobQueue(orchestrator=orchestrator, repository=repo)

    previous_orchestrator = api_app.orchestrator
    previous_job_queue = api_app.job_queue
    api_app.orchestrator = orchestrator
    api_app.job_queue = queue
    try:
        first_request = AnalyzeRequest(idea="idea one", mode="deep")
        asyncio.run(
            api_app.analyze(
                request=first_request,
                response=Response(),
                wait=False,
                run_id="fixed-run-2",
            )
        )
        conflicting_request = AnalyzeRequest(idea="idea two", mode="deep")
        try:
            asyncio.run(
                api_app.analyze(
                    request=conflicting_request,
                    response=Response(),
                    wait=False,
                    run_id="fixed-run-2",
                )
            )
            raise AssertionError("expected HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 409
            assert "different request payload" in str(exc.detail)
    finally:
        api_app.orchestrator = previous_orchestrator
        api_app.job_queue = previous_job_queue
