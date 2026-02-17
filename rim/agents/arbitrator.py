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

SPECIALIST_ARBITRATION_PROMPT = """You are a specialist arbitration reviewer.
Return STRICT JSON only with:
{{
  "node_id": "string",
  "resolved_issue": "string",
  "rationale": "string",
  "action": "escalate|merge|drop",
  "confidence": 0.0
}}

Node diversity flag:
{flag}

Assigned specialist contract:
{specialist_contract}

Target disagreement:
{disagreement}

Latest arbitration decision:
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


def _needs_specialist_pass(decision: dict[str, Any], min_confidence: float) -> bool:
    confidence = float(decision.get("confidence", 0.0))
    action = str(decision.get("action", "merge")).strip().lower()
    if confidence < min_confidence:
        return True
    return action == "escalate"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_specialist_contracts(
    specialist_contracts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in list(specialist_contracts or []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        critic_type = str(item.get("critic_type") or "").strip()
        stage = str(item.get("stage") or "").strip() or "critic_arbitration_specialist"
        tool_contract = item.get("tool_contract")
        if not isinstance(tool_contract, dict):
            tool_contract = {}
        tools: list[str] = []
        seen_tools: set[str] = set()
        for raw_tool in list(tool_contract.get("tools") or []):
            tool = str(raw_tool or "").strip()
            if not tool or tool in seen_tools:
                continue
            seen_tools.add(tool)
            tools.append(tool)
        routing_policy = str(tool_contract.get("routing_policy") or "generic").strip() or "generic"
        matched_keywords: list[str] = []
        for raw_keyword in list(item.get("matched_keywords") or []):
            keyword = str(raw_keyword or "").strip().lower()
            if keyword and keyword not in matched_keywords:
                matched_keywords.append(keyword)
        if not role and not critic_type:
            continue
        role_name = role or critic_type
        critic_name = critic_type or role_name
        normalized.append(
            {
                "role": role_name,
                "critic_type": critic_name,
                "stage": stage,
                "tools": tools,
                "routing_policy": routing_policy,
                "score": round(_coerce_float(item.get("score"), 0.0), 3),
                "matched_keywords": matched_keywords[:8],
                "dynamic": bool(item.get("dynamic", False)),
            }
        )
    return normalized


def _specialist_match_score(
    *,
    contract: dict[str, Any],
    disagreement: dict[str, Any],
    findings: list[CriticFinding],
) -> float:
    role = str(contract.get("role") or "").strip().lower()
    critic_type = str(contract.get("critic_type") or "").strip().lower()
    base_score = round(_coerce_float(contract.get("score"), 0.0), 3)
    matched_keywords = {
        str(item or "").strip().lower()
        for item in list(contract.get("matched_keywords") or [])
        if str(item or "").strip()
    }
    disagreement_critic_types = {
        str(item or "").strip().lower()
        for item in list(disagreement.get("critic_types") or [])
        if str(item or "").strip()
    }
    disagreement_text = " ".join(str(item) for item in list(disagreement.get("issues") or [])).lower()

    score = 0.05 * base_score
    if critic_type and critic_type in disagreement_critic_types:
        score += 4.0
    if role and role in disagreement_critic_types:
        score += 2.0
    if role and role in disagreement_text:
        score += 0.9
    if critic_type and critic_type in disagreement_text:
        score += 0.9
    if matched_keywords:
        score += 0.2 * sum(1 for keyword in matched_keywords if keyword in disagreement_text)

    for finding in findings:
        finding_type = str(finding.critic_type or "").strip().lower()
        if critic_type and finding_type == critic_type:
            score += 2.5
        if role and finding_type == role:
            score += 1.5
        finding_text = f"{finding.issue} {finding.suggested_fix}".lower()
        if role and role in finding_text:
            score += 0.5
        if critic_type and critic_type in finding_text:
            score += 0.5
        if matched_keywords:
            score += 0.25 * sum(1 for keyword in matched_keywords if keyword in finding_text)
    return round(score, 3)


def _select_specialist_contract(
    *,
    node_id: str,
    disagreement: dict[str, Any],
    findings: list[CriticFinding],
    specialist_contracts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not specialist_contracts:
        return None
    node_findings = [item for item in findings if item.node_id == node_id]
    ranked: list[tuple[float, float, int, str, dict[str, Any]]] = []
    for contract in specialist_contracts:
        match_score = _specialist_match_score(
            contract=contract,
            disagreement=disagreement,
            findings=node_findings,
        )
        ranked.append(
            (
                match_score,
                _coerce_float(contract.get("score"), 0.0),
                len(list(contract.get("matched_keywords") or [])),
                str(contract.get("role") or ""),
                contract,
            )
        )
    if not ranked:
        return None
    ranked.sort(reverse=True)
    selected = dict(ranked[0][4])
    selected["match_score"] = round(float(ranked[0][0]), 3)
    return selected


def _default_specialist_contract() -> dict[str, Any]:
    return {
        "role": "specialist",
        "critic_type": "specialist",
        "stage": "critic_arbitration_specialist",
        "tools": ["consistency_check", "counterexample_search"],
        "routing_policy": "prioritize_high_risk_disagreements",
        "score": 0.0,
        "matched_keywords": [],
        "dynamic": False,
        "match_score": 0.0,
    }


def _attach_specialist_metadata(
    decision: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    tools: list[str] = []
    seen_tools: set[str] = set()
    for raw_tool in list(contract.get("tools") or []):
        tool = str(raw_tool or "").strip()
        if not tool or tool in seen_tools:
            continue
        seen_tools.add(tool)
        tools.append(tool)
    decision["specialist_role"] = str(contract.get("role") or "specialist")
    decision["specialist_critic_type"] = str(
        contract.get("critic_type") or decision["specialist_role"]
    )
    decision["specialist_stage"] = str(
        contract.get("stage") or "critic_arbitration_specialist"
    )
    decision["specialist_routing_policy"] = str(
        contract.get("routing_policy") or "generic"
    )
    decision["specialist_tools"] = tools
    decision["specialist_dynamic"] = bool(contract.get("dynamic", False))
    decision["specialist_match_score"] = round(
        _coerce_float(contract.get("match_score"), 0.0),
        3,
    )


async def run_arbitration(
    router: Any,
    *,
    reconciliation: dict[str, Any],
    findings: list[CriticFinding],
    max_jobs: int = 2,
    devils_advocate_rounds: int = 0,
    devils_advocate_min_confidence: float = 0.72,
    specialist_loop_enabled: bool = False,
    specialist_max_jobs: int = 2,
    specialist_min_confidence: float = 0.78,
    specialist_contracts: list[dict[str, Any]] | None = None,
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

    if specialist_loop_enabled:
        guardrails = reconciliation.get("diversity_guardrails", {})
        flagged_nodes = list(guardrails.get("flagged_nodes") or [])
        normalized_specialist_max_jobs = max(0, int(specialist_max_jobs))
        normalized_specialist_conf = max(0.0, min(1.0, float(specialist_min_confidence)))
        normalized_specialist_contracts = _normalize_specialist_contracts(specialist_contracts)
        jobs: list[tuple[str, dict[str, Any]]] = []
        for item in flagged_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id") or "").strip()
            if not node_id:
                continue
            latest = latest_by_node.get(node_id)
            if latest is None:
                continue
            if not _needs_specialist_pass(latest, normalized_specialist_conf):
                continue
            jobs.append((node_id, item))
            if len(jobs) >= normalized_specialist_max_jobs:
                break

        for node_id, flag in jobs:
            disagreement = disagreement_by_node.get(node_id, {"node_id": node_id})
            prior = latest_by_node.get(node_id, {"node_id": node_id})
            specialist_contract = _select_specialist_contract(
                node_id=node_id,
                disagreement=disagreement,
                findings=findings,
                specialist_contracts=normalized_specialist_contracts,
            ) or _default_specialist_contract()
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
            prompt = SPECIALIST_ARBITRATION_PROMPT.format(
                flag=flag,
                specialist_contract=specialist_contract,
                disagreement=disagreement,
                prior_decision=prior,
                findings=scoped_findings,
            )
            try:
                payload, provider = await router.invoke_json(
                    "critic_arbitration_specialist",
                    prompt,
                    json_schema=ARBITRATION_SCHEMA,
                )
                normalized = _normalize_arbitration(payload, node_id)
                normalized["round"] = "specialist"
                _attach_specialist_metadata(normalized, specialist_contract)
                output.append(normalized)
                latest_by_node[node_id] = normalized
                providers.append(provider)
            except Exception:  # noqa: BLE001
                fallback = {
                    "node_id": node_id,
                    "resolved_issue": str(prior.get("resolved_issue") or "Escalate disagreement"),
                    "rationale": "Specialist arbitration failed; preserving unresolved risk.",
                    "action": "escalate",
                    "confidence": min(0.3, float(prior.get("confidence") or 0.3)),
                    "round": "specialist",
                }
                _attach_specialist_metadata(fallback, specialist_contract)
                output.append(fallback)
                latest_by_node[node_id] = fallback
    return output, providers
