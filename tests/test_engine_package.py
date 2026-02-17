from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rim.core.engine_runtime import EngineAgents, RimExecutionEngine
from rim.core.orchestrator import RimOrchestrator
from rim.engine import EngineAgentRegistry, build_agents, build_engine, build_orchestrator
from rim.providers.router import ProviderRouter
from rim.storage.repo import RunRepository


def test_build_engine_returns_execution_engine(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_engine_package.db")
    router = ProviderRouter()
    engine = build_engine(repository=repo, router=router)
    assert isinstance(engine, RimExecutionEngine)
    assert engine.repository is repo
    assert engine.router is router


def test_build_orchestrator_returns_orchestrator_with_engine(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_engine_package_orchestrator.db")
    router = ProviderRouter()
    orchestrator = build_orchestrator(repository=repo, router=router)
    assert isinstance(orchestrator, RimOrchestrator)
    assert isinstance(orchestrator.engine, RimExecutionEngine)
    assert orchestrator.repository is repo
    assert orchestrator.router is router


async def _stub_run_critics(*args: Any, **kwargs: Any) -> list[Any]:  # noqa: ANN401
    return []


def test_engine_agent_registry_registers_and_builds_pack() -> None:
    registry = EngineAgentRegistry()
    assert "default" in registry.list_packs()

    custom_pack = replace(
        EngineAgents(),
        run_critics=_stub_run_critics,
    )
    registry.register_pack("custom", custom_pack)
    built = registry.build(pack="custom")
    assert built.run_critics is _stub_run_critics


def test_build_agents_rejects_unknown_override_key() -> None:
    try:
        build_agents(
            overrides={"unknown_stage": _stub_run_critics},
            registry=EngineAgentRegistry(),
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "unknown EngineAgents override keys" in str(exc)


def test_build_engine_accepts_agent_pack_from_registry(tmp_path: Path) -> None:
    repo = RunRepository(db_path=tmp_path / "rim_engine_registry.db")
    router = ProviderRouter()
    registry = EngineAgentRegistry()
    custom_pack = replace(
        EngineAgents(),
        run_critics=_stub_run_critics,
    )
    registry.register_pack("custom", custom_pack)
    engine = build_engine(
        repository=repo,
        router=router,
        agent_pack="custom",
        registry=registry,
    )
    assert isinstance(engine, RimExecutionEngine)
    assert engine.agents.run_critics is _stub_run_critics
