from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Response

from rim.api.job_queue import RunJobQueue
from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import (
    AnalyzeRequest,
    AnalyzeRunResponse,
    HealthResponse,
    RunFeedbackRequest,
    RunFeedbackResponse,
    RunListResponse,
    RunLogsResponse,
)
from rim.engine import build_orchestrator as build_embedded_orchestrator
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository

repository = RunRepository()
router = ProviderRouter()
orchestrator = build_embedded_orchestrator(repository=repository, router=router)
job_queue = RunJobQueue(orchestrator=orchestrator, repository=repository)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await job_queue.start()
    try:
        yield
    finally:
        await job_queue.stop()


app = FastAPI(title="RIM MVP", version="0.1.0", lifespan=_lifespan)


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
    run_id: str | None = Query(default=None),
) -> AnalyzeRunResponse:
    try:
        resolved_run_id, created = await job_queue.submit(request, run_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if wait:
        run = await job_queue.wait_for(resolved_run_id)
        if run is None:
            raise HTTPException(status_code=500, detail="Run state missing after execution")
        response.status_code = 200
        return run

    run = orchestrator.get_run(resolved_run_id)
    if run is None:
        response.status_code = 202
        return AnalyzeRunResponse(
            run_id=resolved_run_id,
            status="queued",
            result=None,
            error_summary=None,
            error=None,
        )
    response.status_code = 202 if created else 200
    return run


@app.get("/runs", response_model=RunListResponse)
async def list_runs(
    status: Literal["queued", "running", "completed", "failed", "partial", "canceled"] | None = Query(
        default=None
    ),
    mode: Literal["deep", "fast"] | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> RunListResponse:
    return orchestrator.list_runs(
        status=status,
        mode=mode,
        limit=limit,
        offset=offset,
    )


@app.get("/runs/{run_id}", response_model=AnalyzeRunResponse)
async def get_run(run_id: str) -> AnalyzeRunResponse:
    run = orchestrator.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/runs/{run_id}/cancel", response_model=AnalyzeRunResponse)
async def cancel_run(run_id: str) -> AnalyzeRunResponse:
    run = await job_queue.cancel(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/runs/{run_id}/retry", response_model=AnalyzeRunResponse)
async def retry_run(run_id: str, response: Response) -> AnalyzeRunResponse:
    try:
        run = await job_queue.retry(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    response.status_code = 202
    return run


@app.get("/runs/{run_id}/logs", response_model=RunLogsResponse)
async def get_run_logs(run_id: str) -> RunLogsResponse:
    run = orchestrator.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return orchestrator.get_run_logs(run_id)


@app.post("/runs/{run_id}/feedback", response_model=RunFeedbackResponse)
async def submit_run_feedback(
    run_id: str,
    request: RunFeedbackRequest,
) -> RunFeedbackResponse:
    run = orchestrator.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    payload = orchestrator.submit_feedback(
        run_id=run_id,
        verdict=request.verdict,
        notes=request.notes,
    )
    return RunFeedbackResponse.model_validate(payload)
