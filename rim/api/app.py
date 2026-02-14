from fastapi import FastAPI, HTTPException

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, AnalyzeResult, AnalyzeRunResponse, HealthResponse
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository

app = FastAPI(title="RIM MVP", version="0.1.0")

repository = RunRepository()
router = ProviderRouter()
orchestrator = RimOrchestrator(repository=repository, router=router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    provider_status = await router.healthcheck()
    db_ok = repository.healthcheck()
    ok = db_ok and all(provider_status.values())
    return HealthResponse(ok=ok, db=db_ok, providers=provider_status)


@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(request: AnalyzeRequest) -> AnalyzeResult:
    return await orchestrator.analyze(request)


@app.get("/runs/{run_id}", response_model=AnalyzeRunResponse)
async def get_run(run_id: str) -> AnalyzeRunResponse:
    run = orchestrator.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
