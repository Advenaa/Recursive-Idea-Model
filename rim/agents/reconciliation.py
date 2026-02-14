from __future__ import annotations

from collections import defaultdict

from rim.core.schemas import CriticFinding

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _normalized_issue(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _max_severity(values: list[str]) -> str:
    if not values:
        return "medium"
    return max(values, key=lambda value: SEVERITY_ORDER.get(value, 2))


def reconcile_findings(
    findings: list[CriticFinding],
    *,
    consensus_min_agents: int = 3,
    consensus_min_confidence: float = 0.7,
) -> dict:
    grouped: dict[tuple[str, str], list[CriticFinding]] = defaultdict(list)
    for finding in findings:
        key = (finding.node_id, _normalized_issue(finding.issue))
        grouped[key].append(finding)

    consensus_flaws: list[dict] = []
    per_node_issues: dict[str, list[dict]] = defaultdict(list)

    for (node_id, issue_key), items in grouped.items():
        if not issue_key:
            continue
        critic_types = sorted({item.critic_type for item in items})
        confidences = [float(item.confidence) for item in items]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        representative = max(items, key=lambda item: float(item.confidence))
        issue_meta = {
            "node_id": node_id,
            "issue": representative.issue,
            "critic_types": critic_types,
            "support_count": len(items),
            "avg_confidence": round(avg_confidence, 4),
            "max_confidence": round(max(confidences) if confidences else 0.0, 4),
            "severity": _max_severity([item.severity for item in items]),
        }
        per_node_issues[node_id].append(issue_meta)
        if len(critic_types) >= consensus_min_agents and avg_confidence >= consensus_min_confidence:
            consensus_flaws.append(issue_meta)

    disagreements: list[dict] = []
    for node_id, issues in per_node_issues.items():
        if len(issues) <= 1:
            continue
        disagreements.append(
            {
                "node_id": node_id,
                "issue_count": len(issues),
                "issues": [item["issue"] for item in issues[:4]],
                "critic_types": sorted(
                    {
                        critic
                        for item in issues
                        for critic in list(item.get("critic_types") or [])
                    }
                ),
            }
        )

    return {
        "consensus_flaws": consensus_flaws[:12],
        "disagreements": disagreements[:12],
        "summary": {
            "total_findings": len(findings),
            "consensus_count": len(consensus_flaws),
            "disagreement_count": len(disagreements),
        },
    }
