from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import uuid4

from rim.core.modes import ModeSettings
from rim.core.schemas import CriticFinding, DecompositionNode
from rim.providers.router import ProviderRouter

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "issue": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "suggested_fix": {"type": "string"},
    },
    "required": ["issue", "severity", "confidence", "suggested_fix"],
}

CRITIC_PROMPT = """You are the {critic_type} critic for one idea component.
Return STRICT JSON only with:
{{
  "issue": "string",
  "severity": "low|medium|high|critical",
  "confidence": 0.0,
  "suggested_fix": "string"
}}

Evidence requirement: {evidence_requirement}

Component:
{component}
"""

DEEP_CRITICS: list[tuple[str, str]] = [
    ("critic_logic", "logic"),
    ("critic_evidence", "evidence"),
    ("critic_execution", "execution"),
    ("critic_adversarial", "adversarial"),
]
FAST_CRITICS: list[tuple[str, str]] = [
    ("critic_logic", "logic"),
    ("critic_execution", "execution"),
]


def _valid_severity(value: str) -> str:
    value = value.strip().lower()
    if value in {"low", "medium", "high", "critical"}:
        return value
    return "medium"


def _make_finding(
    node: DecompositionNode,
    critic_type: str,
    provider: str,
    payload: dict[str, Any],
) -> CriticFinding:
    return CriticFinding(
        id=str(uuid4()),
        node_id=node.id,
        critic_type=critic_type,
        issue=str(payload.get("issue", "Insufficient detail in component")).strip(),
        severity=_valid_severity(str(payload.get("severity", "medium"))),
        confidence=float(payload.get("confidence", 0.5)),
        suggested_fix=str(
            payload.get(
                "suggested_fix",
                "Clarify assumptions, evidence, and execution dependencies.",
            )
        ).strip(),
        provider=provider,
    )


async def run_critics(
    router: ProviderRouter,
    nodes: list[DecompositionNode],
    settings: ModeSettings,
) -> list[CriticFinding]:
    critics = DEEP_CRITICS if settings.mode == "deep" else FAST_CRITICS
    max_parallel = int(os.getenv("RIM_MAX_PARALLEL_CRITICS", "6"))
    semaphore = asyncio.Semaphore(max_parallel)
    findings: list[CriticFinding] = []

    async def _job(node: DecompositionNode, stage: str, critic_type: str) -> None:
        prompt = CRITIC_PROMPT.format(
            critic_type=critic_type,
            evidence_requirement=settings.evidence_requirement,
            component=node.component_text,
        )
        async with semaphore:
            try:
                payload, provider = await router.invoke_json(
                    stage,
                    prompt,
                    json_schema=CRITIC_SCHEMA,
                )
                findings.append(_make_finding(node, critic_type, provider, payload))
            except Exception:  # noqa: BLE001
                findings.append(
                    CriticFinding(
                        id=str(uuid4()),
                        node_id=node.id,
                        critic_type=critic_type,
                        issue="Critic stage failed to parse a valid response.",
                        severity="high",
                        confidence=0.2,
                        suggested_fix="Retry this component with tighter JSON constraints.",
                        provider=None,
                    )
                )

    tasks = [
        _job(node, stage_name, critic_type)
        for node in nodes
        for stage_name, critic_type in critics[: settings.critics_per_node]
    ]
    await asyncio.gather(*tasks)
    return findings
