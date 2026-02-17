from __future__ import annotations

from pathlib import Path

from rim.core.engine_runtime import RimExecutionEngine
from rim.core.orchestrator import RimOrchestrator
from rim.engine import build_engine, build_orchestrator
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
