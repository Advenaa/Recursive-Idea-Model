from __future__ import annotations

import asyncio
from pathlib import Path

import rim.api.app as api_app
from rim.core.orchestrator import RimOrchestrator
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def test_list_runs_endpoint_filters_and_paginates(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_api.db")
    repo.create_run_with_request(
        run_id="run-1",
        mode="deep",
        input_idea="idea one",
        request_json='{"idea":"idea one","mode":"deep"}',
        status="completed",
    )
    repo.create_run_with_request(
        run_id="run-2",
        mode="fast",
        input_idea="idea two",
        request_json='{"idea":"idea two","mode":"fast"}',
        status="failed",
    )
    repo.create_run_with_request(
        run_id="run-3",
        mode="deep",
        input_idea="idea three",
        request_json='{"idea":"idea three","mode":"deep"}',
        status="queued",
    )

    previous_orchestrator = api_app.orchestrator
    api_app.orchestrator = RimOrchestrator(repository=repo, router=ProviderRouter())
    try:
        filtered = asyncio.run(
            api_app.list_runs(
                status="completed",
                mode="deep",
                limit=10,
                offset=0,
            )
        )
        assert filtered.count == 1
        assert len(filtered.runs) == 1
        assert filtered.runs[0].run_id == "run-1"

        paged = asyncio.run(
            api_app.list_runs(
                status=None,
                mode=None,
                limit=2,
                offset=0,
            )
        )
        assert paged.count == 2
        assert [item.run_id for item in paged.runs] == ["run-3", "run-2"]
    finally:
        api_app.orchestrator = previous_orchestrator
