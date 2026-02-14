from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, AnalyzeResult, AnalyzeRunResponse, HealthResponse
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository

app = FastAPI(title="RIM MVP", version="0.1.0")

repository = RunRepository()
router = ProviderRouter()
orchestrator = RimOrchestrator(repository=repository, router=router)
in_flight_runs: dict[str, asyncio.Task[AnalyzeResult]] = {}


def _register_task(run_id: str, task: asyncio.Task[AnalyzeResult]) -> None:
    in_flight_runs[run_id] = task

    def _cleanup(_task: asyncio.Task[Any]) -> None:
        in_flight_runs.pop(run_id, None)

    task.add_done_callback(_cleanup)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    provider_status = await router.healthcheck()
    db_ok = repository.healthcheck()
    ok = db_ok and all(provider_status.values())
    return HealthResponse(ok=ok, db=db_ok, providers=provider_status)


@app.post("/analyze", response_model=AnalyzeRunResponse)
async def analyze(
    request: AnalyzeRequest,
    response: Response,
    wait: bool = Query(default=False),
) -> AnalyzeRunResponse:
    run_id = orchestrator.create_run(request)
    task = asyncio.create_task(orchestrator.execute_run(run_id, request))
    _register_task(run_id, task)

    if wait:
        try:
            await task
        except Exception:  # noqa: BLE001
            pass
        run = orchestrator.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=500, detail="Run state missing after execution")
        response.status_code = 200
        return run

    response.status_code = 202
    return AnalyzeRunResponse(run_id=run_id, status="running", result=None, error_summary=None)


@app.get("/runs/{run_id}", response_model=AnalyzeRunResponse)
async def get_run(run_id: str) -> AnalyzeRunResponse:
    run = orchestrator.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
