from __future__ import annotations

import asyncio
import os
import re
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

Domain context:
{domain}

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

_DOMAIN_RE = re.compile(r"[^a-z0-9]+")


def _domain_slug(domain: str | None) -> str:
    text = str(domain or "").strip().lower()
    if not text:
        return ""
    slug = _DOMAIN_RE.sub("_", text).strip("_")
    return slug[:40]


def _domain_specialist_stage(domain: str | None) -> tuple[str, str] | None:
    slug = _domain_slug(domain)
    if not slug:
        return None
    return (f"critic_domain_{slug}", f"domain_{slug}")


def _domain_critic_enabled() -> bool:
    raw = os.getenv("RIM_ENABLE_DOMAIN_CRITIC", "1")
    value = str(raw).strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return True


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
    domain: str | None = None,
    extra_critics: list[tuple[str, str]] | None = None,
) -> list[CriticFinding]:
    base_critics = list(DEEP_CRITICS if settings.mode == "deep" else FAST_CRITICS)
    cleaned_extra_critics: list[tuple[str, str]] = []
    for item in list(extra_critics or []):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        stage_name, critic_type = item
        stage_text = str(stage_name).strip()
        critic_text = str(critic_type).strip()
        if not stage_text or not critic_text:
            continue
        pair = (stage_text, critic_text)
        if pair not in cleaned_extra_critics and pair not in base_critics:
            cleaned_extra_critics.append(pair)
    domain_stage = _domain_specialist_stage(domain)
    selected_critics = list(base_critics[: settings.critics_per_node])
    for pair in cleaned_extra_critics:
        if pair not in selected_critics:
            selected_critics.append(pair)
    if domain_stage is not None and _domain_critic_enabled() and domain_stage not in selected_critics:
        selected_critics.append(domain_stage)
    max_parallel = int(os.getenv("RIM_MAX_PARALLEL_CRITICS", "6"))
    semaphore = asyncio.Semaphore(max_parallel)
    findings: list[CriticFinding] = []

    async def _job(node: DecompositionNode, stage: str, critic_type: str) -> None:
        prompt = CRITIC_PROMPT.format(
            critic_type=critic_type,
            evidence_requirement=settings.evidence_requirement,
            domain=domain or "general",
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
        for stage_name, critic_type in selected_critics
    ]
    await asyncio.gather(*tasks)
    return findings
