from __future__ import annotations

from typing import Any
from uuid import uuid4

from rim.core.modes import ModeSettings
from rim.core.schemas import DecompositionNode
from rim.providers.router import ProviderRouter

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parent_node_id": {"type": ["string", "null"]},
                    "depth": {"type": "integer", "minimum": 0},
                    "component_text": {"type": "string"},
                    "node_type": {
                        "type": "string",
                        "enum": ["claim", "assumption", "dependency"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["depth", "component_text", "node_type", "confidence"],
            },
        }
    },
    "required": ["nodes"],
}

DECOMPOSE_PROMPT = """You are a rigorous idea decomposition engine.
Return STRICT JSON only with this schema:
{{
  "nodes": [
    {{
      "parent_node_id": "string|null",
      "depth": 0,
      "component_text": "string",
      "node_type": "claim|assumption|dependency",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Decompose the idea into concise analyzable components.
- Include at least one root node with depth 0.
- Target depth up to {max_depth}.
- Keep confidence between 0 and 1.

Idea:
{idea}

Domain:
{domain}

Constraints:
{constraints}
"""


def _to_node(item: dict[str, Any], idx: int) -> DecompositionNode:
    node_type = str(item.get("node_type", "claim")).strip().lower()
    if node_type not in {"claim", "assumption", "dependency"}:
        node_type = "claim"
    return DecompositionNode(
        id=str(uuid4()),
        parent_node_id=item.get("parent_node_id"),
        depth=max(0, int(item.get("depth", 0))),
        component_text=str(
            item.get("component_text")
            or item.get("component")
            or item.get("text")
            or f"Component {idx + 1}"
        ).strip(),
        node_type=node_type,
        confidence=float(item.get("confidence", 0.5)),
    )


async def decompose_idea(
    router: ProviderRouter,
    idea: str,
    settings: ModeSettings,
    domain: str | None = None,
    constraints: list[str] | None = None,
) -> tuple[list[DecompositionNode], str]:
    constraints = constraints or []
    prompt = DECOMPOSE_PROMPT.format(
        idea=idea.strip(),
        max_depth=settings.max_depth,
        domain=domain or "general",
        constraints=", ".join(constraints) if constraints else "none",
    )
    payload, provider = await router.invoke_json(
        "decompose",
        prompt,
        json_schema=DECOMPOSE_SCHEMA,
    )
    raw_nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return [
            DecompositionNode(
                id=str(uuid4()),
                parent_node_id=None,
                depth=0,
                component_text=idea.strip(),
                node_type="claim",
                confidence=0.5,
            )
        ], provider

    nodes = [_to_node(item, idx) for idx, item in enumerate(raw_nodes)]
    if not any(node.depth == 0 for node in nodes):
        nodes.insert(
            0,
            DecompositionNode(
                id=str(uuid4()),
                parent_node_id=None,
                depth=0,
                component_text=idea.strip(),
                node_type="claim",
                confidence=0.5,
            ),
        )
    return nodes, provider
