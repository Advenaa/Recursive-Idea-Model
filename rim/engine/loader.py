from __future__ import annotations

import importlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from rim.core.engine_runtime import EngineAgents
from rim.engine.registry import EngineAgentRegistry


def _import_symbol(symbol_path: str) -> Any:
    value = str(symbol_path or "").strip()
    module_path, sep, attribute_path = value.partition(":")
    if not sep or not module_path.strip() or not attribute_path.strip():
        raise ValueError(
            f"invalid symbol path '{value}'; expected format 'module.submodule:attribute'",
        )
    try:
        target: Any = importlib.import_module(module_path.strip())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"failed importing module '{module_path}': {exc}") from exc
    for part in [item.strip() for item in attribute_path.split(".") if item.strip()]:
        if not hasattr(target, part):
            raise ValueError(
                f"symbol '{value}' could not resolve attribute '{part}'",
            )
        target = getattr(target, part)
    return target


def _resolve_pack_specs(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("agent pack config must be a JSON object")
    nested = payload.get("packs")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("'packs' must be a JSON object")
        return nested
    return payload


def load_agent_packs_config(
    config_path: str | Path,
    *,
    registry: EngineAgentRegistry | None = None,
    replace_existing: bool = True,
) -> EngineAgentRegistry:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise ValueError(f"agent pack config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"failed parsing agent pack config '{path}': {exc}") from exc

    pack_specs = _resolve_pack_specs(payload)
    if not pack_specs:
        raise ValueError("agent pack config does not define any packs")

    active_registry = registry or EngineAgentRegistry()
    allowed_stage_names = {field.name for field in fields(EngineAgents)}

    for pack_name, raw_spec in pack_specs.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"pack '{pack_name}' must be a JSON object")

        base_pack = str(raw_spec.get("base_pack", "default")).strip() or "default"
        raw_overrides = raw_spec.get("overrides", raw_spec)
        if not isinstance(raw_overrides, dict):
            raise ValueError(f"pack '{pack_name}' overrides must be a JSON object")

        overrides: dict[str, Any] = {}
        for key, value in raw_overrides.items():
            stage_name = str(key).strip()
            if stage_name in {"base_pack", "overrides"}:
                continue
            if stage_name not in allowed_stage_names:
                raise ValueError(
                    f"pack '{pack_name}' has unknown EngineAgents stage '{stage_name}'",
                )
            symbol = _import_symbol(str(value))
            if not callable(symbol):
                raise ValueError(
                    f"pack '{pack_name}' stage '{stage_name}' must resolve to a callable",
                )
            overrides[stage_name] = symbol

        if not overrides:
            raise ValueError(f"pack '{pack_name}' does not define callable overrides")

        agents = active_registry.build(pack=base_pack, overrides=overrides)
        active_registry.register_pack(
            str(pack_name),
            agents,
            replace_existing=replace_existing,
        )

    return active_registry
