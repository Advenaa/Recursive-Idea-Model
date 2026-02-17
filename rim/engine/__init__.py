from __future__ import annotations

"""Public engine embedding API for external projects."""

from typing import Any

from rim.core.engine_runtime import EngineAgents, RimExecutionEngine
from rim.core.orchestrator import RimOrchestrator
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def build_engine(
    *,
    repository: RunRepository | None = None,
    router: ProviderRouter | None = None,
    agents: EngineAgents | None = None,
) -> RimExecutionEngine:
    resolved_repository = repository or RunRepository()
    resolved_router = router or ProviderRouter()
    return RimExecutionEngine(
        repository=resolved_repository,
        router=resolved_router,
        agents=agents,
    )


def build_orchestrator(
    *,
    repository: RunRepository | None = None,
    router: ProviderRouter | None = None,
    engine: RimExecutionEngine | None = None,
    agents: EngineAgents | None = None,
) -> RimOrchestrator:
    resolved_repository = repository or RunRepository()
    resolved_router = router or ProviderRouter()
    resolved_engine = engine or build_engine(
        repository=resolved_repository,
        router=resolved_router,
        agents=agents,
    )
    return RimOrchestrator(
        repository=resolved_repository,
        router=resolved_router,
        engine=resolved_engine,
    )


__all__ = [
    "EngineAgents",
    "RimExecutionEngine",
    "RimOrchestrator",
    "build_engine",
    "build_orchestrator",
]
