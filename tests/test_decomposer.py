from __future__ import annotations

import asyncio

from rim.agents.decomposer import decompose_idea
from rim.core.modes import ModeSettings


class FakeRouter:
    async def invoke_json(self, stage: str, prompt: str, json_schema=None):  # noqa: ANN001
        if stage != "decompose":
            raise AssertionError("unexpected stage")
        if "Current parent component (depth 0):\nRoot idea" in prompt:
            return (
                {
                    "children": [
                        {"component_text": "A", "node_type": "claim", "confidence": 0.4},
                        {"component_text": "B", "node_type": "assumption", "confidence": 0.95},
                    ]
                },
                "codex",
            )
        if "Current parent component (depth 1):\nA" in prompt:
            return (
                {
                    "children": [
                        {"component_text": "A1", "node_type": "dependency", "confidence": 0.7}
                    ]
                },
                "claude",
            )
        return ({"children": []}, "codex")


def test_recursive_decomposition_with_confidence_stop() -> None:
    settings = ModeSettings(
        mode="deep",
        max_depth=4,
        critics_per_node=4,
        synthesis_passes=2,
        self_critique_pass=True,
        evidence_requirement="strict",
        confidence_stop_threshold=0.85,
        marginal_gain_min_new_nodes=1,
        marginal_gain_patience=2,
        max_children_per_node=4,
        runtime_budget_sec=60,
    )
    nodes, provider = asyncio.run(
        decompose_idea(
            router=FakeRouter(),
            idea="Root idea",
            settings=settings,
            domain="general",
            constraints=[],
            memory_context=[],
        )
    )

    texts = [node.component_text for node in nodes]
    assert texts == ["Root idea", "A", "B", "A1"]
    assert provider in {"codex", "claude"}
