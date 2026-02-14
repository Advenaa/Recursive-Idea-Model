from __future__ import annotations

import re

from rim.core.schemas import CriticFinding

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "have",
    "must",
    "should",
    "will",
    "can",
}


def _keywords(text: str, *, min_len: int = 4) -> set[str]:
    tokens = _TOKEN_RE.findall(str(text).lower())
    return {
        token
        for token in tokens
        if len(token) >= min_len and token not in _STOPWORDS
    }


def _overlap_ratio(source: set[str], target: set[str]) -> float:
    if not source:
        return 1.0
    overlap = len(source & target)
    return overlap / float(len(source))


def _normalized_synthesis_text(synthesis: dict[str, object]) -> str:
    chunks: list[str] = []
    chunks.append(str(synthesis.get("synthesized_idea") or "").strip())
    for change in list(synthesis.get("changes_summary") or []):
        text = str(change).strip()
        if text:
            chunks.append(text)
    for experiment in list(synthesis.get("next_experiments") or []):
        text = str(experiment).strip()
        if text:
            chunks.append(text)
    return " ".join(item for item in chunks if item)


def verify_synthesis(
    *,
    synthesis: dict[str, object],
    findings: list[CriticFinding],
    constraints: list[str] | None = None,
    min_constraint_overlap: float = 0.6,
    min_finding_overlap: float = 0.35,
    max_high_findings: int = 8,
) -> dict[str, object]:
    output_text = _normalized_synthesis_text(synthesis)
    output_tokens = _keywords(output_text, min_len=3)
    checks: list[dict[str, object]] = []

    for constraint in list(constraints or []):
        text = str(constraint).strip()
        if not text:
            continue
        source = _keywords(text)
        ratio = _overlap_ratio(source, output_tokens)
        checks.append(
            {
                "check_type": "constraint_coverage",
                "description": text,
                "passed": ratio >= float(min_constraint_overlap),
                "ratio": round(ratio, 4),
                "threshold": float(min_constraint_overlap),
                "severity": "high",
            }
        )

    high_findings = [
        finding
        for finding in findings
        if finding.severity in {"high", "critical"}
    ][: max(1, int(max_high_findings))]
    for finding in high_findings:
        source = _keywords(finding.issue)
        ratio = _overlap_ratio(source, output_tokens)
        checks.append(
            {
                "check_type": "risk_coverage",
                "description": finding.issue,
                "passed": ratio >= float(min_finding_overlap),
                "ratio": round(ratio, 4),
                "threshold": float(min_finding_overlap),
                "severity": finding.severity,
                "critic_type": finding.critic_type,
            }
        )

    experiments = list(synthesis.get("next_experiments") or [])
    checks.append(
        {
            "check_type": "actionability",
            "description": "Next experiments should include at least one concrete step.",
            "passed": len([item for item in experiments if str(item).strip()]) >= 1,
            "ratio": float(len(experiments)),
            "threshold": 1.0,
            "severity": "medium",
        }
    )

    failed = [item for item in checks if not bool(item.get("passed"))]
    critical_failures = [
        item
        for item in failed
        if str(item.get("severity")) in {"high", "critical"}
    ]
    total = len(checks)
    score = 1.0 if total == 0 else max(0.0, (total - len(failed)) / float(total))

    return {
        "summary": {
            "total_checks": total,
            "failed_checks": len(failed),
            "critical_failures": len(critical_failures),
            "verification_score": round(score, 4),
        },
        "checks": checks[:30],
    }
