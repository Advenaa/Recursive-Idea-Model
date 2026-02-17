from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from rim.core.engine_runtime import EngineAgents


class EngineAgentRegistry:
    def __init__(self) -> None:
        self._packs: dict[str, EngineAgents] = {"default": EngineAgents()}

    def register_pack(
        self,
        name: str,
        agents: EngineAgents,
        *,
        replace_existing: bool = False,
    ) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("pack name cannot be empty")
        if key in self._packs and not replace_existing:
            raise ValueError(f"agent pack already exists: {key}")
        self._packs[key] = agents

    def list_packs(self) -> list[str]:
        return sorted(self._packs.keys())

    def get_pack(self, name: str = "default") -> EngineAgents:
        key = str(name).strip().lower() or "default"
        if key not in self._packs:
            raise ValueError(f"unknown agent pack: {key}")
        return self._packs[key]

    def build(
        self,
        *,
        pack: str = "default",
        overrides: dict[str, Any] | None = None,
    ) -> EngineAgents:
        base = self.get_pack(pack)
        patch = {
            str(key): value
            for key, value in dict(overrides or {}).items()
            if value is not None
        }
        if not patch:
            return base

        allowed = {field.name for field in fields(EngineAgents)}
        unknown = sorted(key for key in patch.keys() if key not in allowed)
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(f"unknown EngineAgents override keys: {joined}")
        return replace(base, **patch)


default_agent_registry = EngineAgentRegistry()

