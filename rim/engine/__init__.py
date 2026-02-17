from __future__ import annotations

"""Public engine embedding API for external projects."""

import os
from typing import Any

from rim.core.engine_runtime import EngineAgents, RimExecutionEngine
from rim.core.orchestrator import RimOrchestrator
from rim.engine.loader import load_agent_packs_config
from rim.engine.registry import EngineAgentRegistry, default_agent_registry
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def _resolve_agent_pack(pack: str | None) -> str:
    if pack is not None and str(pack).strip():
        return str(pack).strip().lower()
    env_pack = str(os.getenv("RIM_AGENT_PACK", "default")).strip().lower()
    return env_pack or "default"


def _resolve_registry(
    *,
    registry: EngineAgentRegistry | None,
    packs_path: str | None,
) -> EngineAgentRegistry:
    resolved_packs_path = str(packs_path or os.getenv("RIM_AGENT_PACKS_PATH", "")).strip()
    if registry is not None:
        active_registry = registry
    elif resolved_packs_path:
        # Use an isolated registry when loading from a config file to avoid
        # mutating the global default registry between calls.
        active_registry = EngineAgentRegistry()
    else:
        active_registry = default_agent_registry
    if resolved_packs_path:
        load_agent_packs_config(
            resolved_packs_path,
            registry=active_registry,
            replace_existing=True,
        )
    return active_registry


def build_agents(
    *,
    pack: str | None = None,
    overrides: dict[str, Any] | None = None,
    registry: EngineAgentRegistry | None = None,
    packs_path: str | None = None,
) -> EngineAgents:
    resolved_pack = _resolve_agent_pack(pack)
    active_registry = _resolve_registry(
        registry=registry,
        packs_path=packs_path,
    )
    return active_registry.build(pack=resolved_pack, overrides=overrides)


def build_engine(
    *,
    repository: RunRepository | None = None,
    router: ProviderRouter | None = None,
    agents: EngineAgents | None = None,
    agent_pack: str | None = None,
    agent_overrides: dict[str, Any] | None = None,
    registry: EngineAgentRegistry | None = None,
    agent_packs_path: str | None = None,
) -> RimExecutionEngine:
    resolved_repository = repository or RunRepository()
    resolved_router = router or ProviderRouter()
    resolved_agents = agents or build_agents(
        pack=agent_pack,
        overrides=agent_overrides,
        registry=registry,
        packs_path=agent_packs_path,
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
    agent_pack: str | None = None,
    agent_overrides: dict[str, Any] | None = None,
    registry: EngineAgentRegistry | None = None,
    agent_packs_path: str | None = None,
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
        agent_packs_path=agent_packs_path,
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
    "load_agent_packs_config",
]
