from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from rim.core.modes import ModeSettings
from rim.core.schemas import DecompositionNode
from rim.providers.router import ProviderRouter

COMPONENT_DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "children": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "component_text": {"type": "string"},
                    "node_type": {
                        "type": "string",
                        "enum": ["claim", "assumption", "dependency"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["component_text", "node_type", "confidence"],
            },
        }
    },
    "required": ["children"],
}

COMPONENT_DECOMPOSE_PROMPT = """You are a rigorous recursive decomposition engine.
Return STRICT JSON only with:
{{
  "children": [
    {{
      "component_text": "string",
      "node_type": "claim|assumption|dependency",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Produce distinct and non-overlapping child components.
- Keep child components concise and concrete.
- Maximum children: {max_children}.
- Do not include the parent text verbatim.
- Keep confidence between 0 and 1.

Global root idea:
{idea}

Current parent component (depth {depth}):
{component}

Domain:
{domain}

Constraints:
{constraints}

Memory context from prior runs:
{memory_context}
"""


def _clamp_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, parsed))


def _normalize_node_type(value: Any) -> str:
    node_type = str(value or "claim").strip().lower()
    if node_type in {"claim", "assumption", "dependency"}:
        return node_type
    return "claim"


def _normalize_component_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _make_child_node(parent: DecompositionNode, item: dict[str, Any], idx: int) -> DecompositionNode:
    return DecompositionNode(
        id=str(uuid4()),
        parent_node_id=parent.id,
        depth=parent.depth + 1,
        component_text=_normalize_component_text(
            item.get("component_text") or item.get("component") or item.get("text"),
            fallback=f"{parent.component_text} - subcomponent {idx + 1}",
        ),
        node_type=_normalize_node_type(item.get("node_type")),
        confidence=_clamp_confidence(item.get("confidence", 0.5)),
    )


def _memory_context_block(memory_context: list[str] | None) -> str:
    if not memory_context:
        return "none"
    return "\n".join(f"- {entry}" for entry in memory_context[:10])


def _constraints_block(constraints: list[str] | None) -> str:
    if not constraints:
        return "none"
    return ", ".join(str(item).strip() for item in constraints if str(item).strip()) or "none"


async def _decompose_component(
    router: ProviderRouter,
    parent: DecompositionNode,
    idea: str,
    settings: ModeSettings,
    domain: str | None,
    constraints: list[str],
    memory_context: list[str] | None,
) -> tuple[list[DecompositionNode], str]:
    prompt = COMPONENT_DECOMPOSE_PROMPT.format(
        max_children=settings.max_children_per_node,
        idea=idea.strip(),
        depth=parent.depth,
        component=parent.component_text,
        domain=domain or "general",
        constraints=_constraints_block(constraints),
        memory_context=_memory_context_block(memory_context),
    )
    payload, provider = await router.invoke_json(
        "decompose",
        prompt,
        json_schema=COMPONENT_DECOMPOSE_SCHEMA,
    )
    raw_children = payload.get("children") if isinstance(payload, dict) else None
    if not isinstance(raw_children, list) or not raw_children:
        return [], provider

    children = [
        _make_child_node(parent, item, idx)
        for idx, item in enumerate(raw_children[: settings.max_children_per_node])
        if isinstance(item, dict)
    ]
    return children, provider


async def decompose_idea(
    router: ProviderRouter,
    idea: str,
    settings: ModeSettings,
    domain: str | None = None,
    constraints: list[str] | None = None,
    memory_context: list[str] | None = None,
) -> tuple[list[DecompositionNode], str]:
    constraints = constraints or []
    root = DecompositionNode(
        id=str(uuid4()),
        parent_node_id=None,
        depth=0,
        component_text=idea.strip(),
        node_type="claim",
        confidence=0.5,
    )

    all_nodes: list[DecompositionNode] = [root]
    current_level: list[DecompositionNode] = [root]
    provider_order: list[str] = []
    seen_components = {root.component_text.strip().lower()}
    low_gain_levels = 0
    started = time.monotonic()

    for _depth in range(settings.max_depth):
        if not current_level:
            break
        if time.monotonic() - started > settings.runtime_budget_sec:
            break

        next_level: list[DecompositionNode] = []
        level_new_nodes = 0
        for parent in current_level:
            if parent.confidence >= settings.confidence_stop_threshold:
                continue
            children, provider = await _decompose_component(
                router=router,
                parent=parent,
                idea=idea,
                settings=settings,
                domain=domain,
                constraints=constraints,
                memory_context=memory_context,
            )
            provider_order.append(provider)
            for child in children:
                key = child.component_text.strip().lower()
                if not key or key in seen_components:
                    continue
                seen_components.add(key)
                next_level.append(child)
                all_nodes.append(child)
                level_new_nodes += 1

        if level_new_nodes < settings.marginal_gain_min_new_nodes:
            low_gain_levels += 1
        else:
            low_gain_levels = 0
        if low_gain_levels >= settings.marginal_gain_patience:
            break
        current_level = next_level

    provider = provider_order[-1] if provider_order else "none"
    return all_nodes, provider
