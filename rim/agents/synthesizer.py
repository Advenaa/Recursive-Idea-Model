from __future__ import annotations

import json
from typing import Any

from rim.core.modes import ModeSettings
from rim.core.schemas import CriticFinding, DecompositionNode
from rim.providers.router import ProviderRouter

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "synthesized_idea": {"type": "string"},
        "changes_summary": {
            "type": "array",
            "items": {"type": "string"},
        },
        "residual_risks": {
            "type": "array",
            "items": {"type": "string"},
        },
        "next_experiments": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "synthesized_idea",
        "changes_summary",
        "residual_risks",
        "next_experiments",
        "confidence_score",
    ],
}

SYNTHESIZE_PROMPT = """You are a synthesis engine for Recursive Idea Model.
Return STRICT JSON only:
{{
  "synthesized_idea": "string",
  "changes_summary": ["string"],
  "residual_risks": ["string"],
  "next_experiments": ["string"],
  "confidence_score": 0.0
}}

Original idea:
{idea}

Decomposition nodes:
{nodes}

Critic findings:
{findings}

Critique reconciliation:
{reconciliation}

Memory context from prior runs:
{memory_context}
"""

REFINE_PROMPT = """Refine the synthesis output for clarity and rigor.
Return STRICT JSON with the same schema as before.

Original idea:
{idea}

Current synthesis:
{synthesis}

Top findings:
{findings}

Memory context from prior runs:
{memory_context}
"""


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _memory_context_block(memory_context: list[str] | None) -> str:
    if not memory_context:
        return "none"
    return "\n".join(f"- {entry}" for entry in memory_context[:10])


def _normalize_synthesis(payload: dict[str, Any], idea: str) -> dict[str, Any]:
    synthesized_idea = str(payload.get("synthesized_idea", "")).strip() or idea
    changes = _clean_list(payload.get("changes_summary"))[:8]
    risks = _clean_list(payload.get("residual_risks"))[:8]
    experiments = _clean_list(payload.get("next_experiments"))[:8]
    confidence = float(payload.get("confidence_score", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "synthesized_idea": synthesized_idea,
        "changes_summary": changes,
        "residual_risks": risks,
        "next_experiments": experiments,
        "confidence_score": confidence,
    }


def _reconciliation_block(reconciliation: dict[str, Any] | None) -> str:
    if not reconciliation:
        return "none"
    summary = reconciliation.get("summary")
    consensus = list(reconciliation.get("consensus_flaws") or [])[:6]
    disagreements = list(reconciliation.get("disagreements") or [])[:6]
    payload = {
        "summary": summary if isinstance(summary, dict) else {},
        "consensus_flaws": consensus,
        "disagreements": disagreements,
    }
    return json.dumps(payload)


async def synthesize_idea(
    router: ProviderRouter,
    idea: str,
    nodes: list[DecompositionNode],
    findings: list[CriticFinding],
    settings: ModeSettings,
    memory_context: list[str] | None = None,
    reconciliation: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    node_view = [
        {
            "id": node.id,
            "depth": node.depth,
            "node_type": node.node_type,
            "component_text": node.component_text,
        }
        for node in nodes
    ]
    finding_view = [
        {
            "node_id": finding.node_id,
            "critic_type": finding.critic_type,
            "severity": finding.severity,
            "issue": finding.issue,
            "suggested_fix": finding.suggested_fix,
        }
        for finding in findings
    ]

    prompt = SYNTHESIZE_PROMPT.format(
        idea=idea,
        nodes=node_view,
        findings=finding_view,
        reconciliation=_reconciliation_block(reconciliation),
        memory_context=_memory_context_block(memory_context),
    )
    primary_payload, primary_provider = await router.invoke_json(
        "synthesize_primary",
        prompt,
        json_schema=SYNTHESIS_SCHEMA,
    )
    synthesis = _normalize_synthesis(primary_payload, idea)
    providers = [primary_provider]

    if settings.self_critique_pass and settings.synthesis_passes > 1:
        refine_prompt = REFINE_PROMPT.format(
            idea=idea,
            synthesis=synthesis,
            findings=finding_view[:12],
            memory_context=_memory_context_block(memory_context),
        )
        refined_payload, refined_provider = await router.invoke_json(
            "synthesize_refine",
            refine_prompt,
            json_schema=SYNTHESIS_SCHEMA,
        )
        synthesis = _normalize_synthesis(refined_payload, idea)
        providers.append(refined_provider)

    return synthesis, providers
