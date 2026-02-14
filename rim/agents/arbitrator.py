from __future__ import annotations

from typing import Any

from rim.core.schemas import CriticFinding

ARBITRATION_SCHEMA = {
    "type": "object",
    "properties": {
        "node_id": {"type": "string"},
        "resolved_issue": {"type": "string"},
        "rationale": {"type": "string"},
        "action": {"type": "string", "enum": ["escalate", "merge", "drop"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["node_id", "resolved_issue", "rationale", "action", "confidence"],
}

ARBITRATION_PROMPT = """You are an arbitration judge for conflicting critiques.
Return STRICT JSON only with:
{{
  "node_id": "string",
  "resolved_issue": "string",
  "rationale": "string",
  "action": "escalate|merge|drop",
  "confidence": 0.0
}}

Disagreement target:
{disagreement}

Relevant findings:
{findings}
"""

DEVILS_ADVOCATE_PROMPT = """You are a devil's-advocate arbitration judge.
Given an earlier arbitration decision, challenge it and return STRICT JSON only with:
{{
  "node_id": "string",
  "resolved_issue": "string",
  "rationale": "string",
  "action": "escalate|merge|drop",
  "confidence": 0.0
}}

Target disagreement:
{disagreement}

Prior arbitration:
{prior_decision}

Relevant findings:
{findings}
"""


def _normalize_arbitration(payload: dict[str, Any], default_node_id: str) -> dict[str, Any]:
    action = str(payload.get("action", "merge")).strip().lower()
    if action not in {"escalate", "merge", "drop"}:
        action = "merge"
    confidence = float(payload.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    node_id = str(payload.get("node_id") or "").strip() or default_node_id
    resolved_issue = str(payload.get("resolved_issue") or "").strip() or "Unresolved disagreement"
    rationale = str(payload.get("rationale") or "").strip() or "Insufficient detail from arbitration."
    return {
        "node_id": node_id,
        "resolved_issue": resolved_issue,
        "rationale": rationale,
        "action": action,
        "confidence": confidence,
    }


def _needs_devils_advocate_pass(decision: dict[str, Any], min_confidence: float) -> bool:
    confidence = float(decision.get("confidence", 0.0))
    action = str(decision.get("action", "merge")).strip().lower()
    if confidence < min_confidence:
        return True
    return action == "escalate"


async def run_arbitration(
    router: Any,
    *,
    reconciliation: dict[str, Any],
    findings: list[CriticFinding],
    max_jobs: int = 2,
    devils_advocate_rounds: int = 0,
    devils_advocate_min_confidence: float = 0.72,
) -> tuple[list[dict[str, Any]], list[str]]:
    disagreements = list(reconciliation.get("disagreements") or [])[: max(0, int(max_jobs))]
    if not disagreements:
        return [], []

    output: list[dict[str, Any]] = []
    providers: list[str] = []
    disagreement_by_node: dict[str, dict[str, Any]] = {}
    for disagreement in disagreements:
        if not isinstance(disagreement, dict):
            continue
        node_id = str(disagreement.get("node_id") or "").strip()
        if not node_id:
            continue
        disagreement_by_node[node_id] = disagreement
        scoped_findings = [
            {
                "critic_type": item.critic_type,
                "severity": item.severity,
                "issue": item.issue,
                "suggested_fix": item.suggested_fix,
            }
            for item in findings
            if item.node_id == node_id
        ][:8]
        prompt = ARBITRATION_PROMPT.format(
            disagreement=disagreement,
            findings=scoped_findings,
        )
        try:
            payload, provider = await router.invoke_json(
                "critic_arbitration",
                prompt,
                json_schema=ARBITRATION_SCHEMA,
            )
            normalized = _normalize_arbitration(payload, node_id)
            normalized["round"] = "primary"
            output.append(normalized)
            providers.append(provider)
        except Exception:  # noqa: BLE001
            output.append(
                {
                    "node_id": node_id,
                    "resolved_issue": "Escalate disagreement for additional review",
                    "rationale": "Arbitration stage failed; preserving disagreement as unresolved risk.",
                    "action": "escalate",
                    "confidence": 0.2,
                    "round": "primary",
                }
            )

    rounds = max(0, int(devils_advocate_rounds))
    min_confidence = max(0.0, min(1.0, float(devils_advocate_min_confidence)))
    latest_by_node: dict[str, dict[str, Any]] = {
        str(item["node_id"]): item for item in output if isinstance(item, dict) and item.get("node_id")
    }
    for round_index in range(1, rounds + 1):
        unresolved = [
            item
            for item in latest_by_node.values()
            if _needs_devils_advocate_pass(item, min_confidence)
        ][: max(0, int(max_jobs))]
        if not unresolved:
            break

        for prior in unresolved:
            node_id = str(prior.get("node_id") or "").strip()
            if not node_id:
                continue
            disagreement = disagreement_by_node.get(node_id, {"node_id": node_id})
            scoped_findings = [
                {
                    "critic_type": item.critic_type,
                    "severity": item.severity,
                    "issue": item.issue,
                    "suggested_fix": item.suggested_fix,
                }
                for item in findings
                if item.node_id == node_id
            ][:8]
            prompt = DEVILS_ADVOCATE_PROMPT.format(
                disagreement=disagreement,
                prior_decision=prior,
                findings=scoped_findings,
            )
            try:
                payload, provider = await router.invoke_json(
                    "critic_arbitration_devil",
                    prompt,
                    json_schema=ARBITRATION_SCHEMA,
                )
                normalized = _normalize_arbitration(payload, node_id)
                normalized["round"] = f"devil_{round_index}"
                output.append(normalized)
                latest_by_node[node_id] = normalized
                providers.append(provider)
            except Exception:  # noqa: BLE001
                fallback = {
                    "node_id": node_id,
                    "resolved_issue": str(prior.get("resolved_issue") or "Escalate disagreement"),
                    "rationale": "Devil's-advocate arbitration failed; preserving unresolved risk.",
                    "action": "escalate",
                    "confidence": min(0.3, float(prior.get("confidence") or 0.3)),
                    "round": f"devil_{round_index}",
                }
                output.append(fallback)
                latest_by_node[node_id] = fallback
    return output, providers
