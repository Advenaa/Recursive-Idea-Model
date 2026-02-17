from __future__ import annotations

"""Public engine embedding API for external projects."""

from typing import Any

from rim.core.engine_runtime import EngineAgents, RimExecutionEngine
from rim.core.orchestrator import RimOrchestrator
from rim.engine.registry import EngineAgentRegistry, default_agent_registry
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def build_agents(
    *,
    pack: str = "default",
    overrides: dict[str, Any] | None = None,
    registry: EngineAgentRegistry | None = None,
) -> EngineAgents:
    active_registry = registry or default_agent_registry
    return active_registry.build(pack=pack, overrides=overrides)


def build_engine(
    *,
    repository: RunRepository | None = None,
    router: ProviderRouter | None = None,
    agents: EngineAgents | None = None,
    agent_pack: str = "default",
    agent_overrides: dict[str, Any] | None = None,
    registry: EngineAgentRegistry | None = None,
) -> RimExecutionEngine:
    resolved_repository = repository or RunRepository()
    resolved_router = router or ProviderRouter()
    resolved_agents = agents or build_agents(
        pack=agent_pack,
        overrides=agent_overrides,
        registry=registry,
    )
    return RimExecutionEngine(
        repository=resolved_repository,
        router=resolved_router,
        agents=resolved_agents,
    )


def build_orchestrator(
    *,
    repository: RunRepository | None = None,
    router: ProviderRouter | None = None,
    engine: RimExecutionEngine | None = None,
    agents: EngineAgents | None = None,
    agent_pack: str = "default",
    agent_overrides: dict[str, Any] | None = None,
    registry: EngineAgentRegistry | None = None,
) -> RimOrchestrator:
    resolved_repository = repository or RunRepository()
    resolved_router = router or ProviderRouter()
    resolved_engine = engine or build_engine(
        repository=resolved_repository,
        router=resolved_router,
        agents=agents,
        agent_pack=agent_pack,
        agent_overrides=agent_overrides,
        registry=registry,
    )
    return RimOrchestrator(
        repository=resolved_repository,
        router=resolved_router,
        engine=resolved_engine,
    )


__all__ = [
    "EngineAgentRegistry",
    "EngineAgents",
    "RimExecutionEngine",
    "RimOrchestrator",
    "build_agents",
    "build_engine",
    "build_orchestrator",
    "default_agent_registry",
]
