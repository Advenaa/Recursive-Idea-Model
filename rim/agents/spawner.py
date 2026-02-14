from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")

_ROLE_RULES: list[tuple[str, list[str], str, str]] = [
    ("security", ["security", "threat", "privacy", "auth", "compliance"], "critic_security", "security"),
    ("finance", ["budget", "cost", "revenue", "pricing", "margin"], "critic_finance", "finance"),
    (
        "scalability",
        ["latency", "throughput", "scale", "scalable", "performance", "capacity"],
        "critic_scalability",
        "scalability",
    ),
    ("ux", ["ux", "usability", "adoption", "onboarding", "retention"], "critic_ux", "ux"),
]

_ROLE_TOOL_CONTRACTS: dict[str, dict[str, Any]] = {
    "security": {
        "tools": ["threat_model", "policy_checklist", "abuse_case_review"],
        "routing_policy": "prioritize_high_severity_and_compliance_constraints",
    },
    "finance": {
        "tools": ["unit_economics_model", "sensitivity_analysis", "pricing_stress_test"],
        "routing_policy": "prioritize_margin_and_budget_constraints",
    },
    "scalability": {
        "tools": ["capacity_profile", "latency_budget", "failure_mode_envelope"],
        "routing_policy": "prioritize_runtime_and_reliability_constraints",
    },
    "ux": {
        "tools": ["journey_map", "onboarding_friction_scan", "retention_hypothesis_grid"],
        "routing_policy": "prioritize_adoption_and_clarity_constraints",
    },
}

_DOMAIN_ROLE_HINTS = {
    "finance": "finance",
    "fintech": "finance",
    "security": "security",
    "privacy": "security",
    "compliance": "security",
    "scalability": "scalability",
    "performance": "scalability",
    "ops": "scalability",
    "product": "ux",
    "consumer": "ux",
    "education": "ux",
}


def _token_counts(text: str) -> Counter[str]:
    return Counter(match.group(0) for match in _WORD_RE.finditer(str(text).lower()))


def _parse_int_env(name: str, default: int, *, lower: int, upper: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw)) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(lower, min(upper, value))


def _parse_float_env(name: str, default: float, *, lower: float, upper: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(str(raw)) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(lower, min(upper, value))


def _domain_hint_role(domain: str | None) -> str | None:
    normalized = str(domain or "").strip().lower()
    if not normalized:
        return None
    for key, role in _DOMAIN_ROLE_HINTS.items():
        if key in normalized:
            return role
    return None


def _role_score(
    *,
    role_name: str,
    keywords: list[str],
    domain_counts: Counter[str],
    constraint_counts: Counter[str],
    memory_counts: Counter[str],
    domain_hint_role: str | None,
) -> tuple[float, list[str], dict[str, int]]:
    domain_hits = sum(domain_counts.get(keyword, 0) for keyword in keywords)
    constraint_hits = sum(constraint_counts.get(keyword, 0) for keyword in keywords)
    memory_hits = sum(memory_counts.get(keyword, 0) for keyword in keywords)
    matched_keywords = [
        keyword
        for keyword in keywords
        if domain_counts.get(keyword, 0)
        or constraint_counts.get(keyword, 0)
        or memory_counts.get(keyword, 0)
    ]
    role_boost = 1.0 if domain_hint_role == role_name else 0.0
    score = (
        (2.0 * float(domain_hits))
        + (1.0 * float(constraint_hits))
        + (0.7 * float(memory_hits))
        + role_boost
    )
    return score, matched_keywords, {
        "domain_hits": domain_hits,
        "constraint_hits": constraint_hits,
        "memory_hits": memory_hits,
    }


def build_spawn_plan(
    *,
    mode: str,
    domain: str | None,
    constraints: list[str] | None,
    memory_context: list[str] | None = None,
) -> dict[str, Any]:
    domain_counts = _token_counts(str(domain or ""))
    constraint_counts = _token_counts(" ".join(str(item) for item in list(constraints or [])))
    memory_counts = _token_counts(" ".join(str(item) for item in list(memory_context or [])[:6]))
    hint_role = _domain_hint_role(domain)
    min_role_score = _parse_float_env(
        "RIM_SPAWN_MIN_ROLE_SCORE",
        1.0,
        lower=0.0,
        upper=20.0,
    )
    scored_candidates: list[dict[str, Any]] = []
    for role_name, keywords, stage, critic_type in _ROLE_RULES:
        score, matched_keywords, evidence = _role_score(
            role_name=role_name,
            keywords=keywords,
            domain_counts=domain_counts,
            constraint_counts=constraint_counts,
            memory_counts=memory_counts,
            domain_hint_role=hint_role,
        )
        if score < min_role_score:
            continue
        scored_candidates.append(
            {
                "role": role_name,
                "stage": stage,
                "critic_type": critic_type,
                "score": round(score, 3),
                "matched_keywords": matched_keywords[:5],
                "evidence": evidence,
                "tool_contract": _ROLE_TOOL_CONTRACTS.get(role_name, {"tools": [], "routing_policy": "generic"}),
            }
        )

    # Deep mode can carry more specialists, fast mode keeps one maximum.
    mode_normalized = str(mode).strip().lower()
    max_specialists = (
        _parse_int_env("RIM_SPAWN_MAX_SPECIALISTS_DEEP", 3, lower=1, upper=8)
        if mode_normalized == "deep"
        else _parse_int_env("RIM_SPAWN_MAX_SPECIALISTS_FAST", 1, lower=1, upper=4)
    )
    scored_candidates.sort(
        key=lambda item: (
            float(item["score"]),
            len(item["matched_keywords"]),
            item["role"],
        ),
        reverse=True,
    )
    selected = scored_candidates[:max_specialists]
    return {
        "domain": str(domain or "").strip() or None,
        "extra_critics": selected,
        "selected_count": len(selected),
        "candidate_count": len(scored_candidates),
        "min_role_score": min_role_score,
        "domain_hint_role": hint_role,
        "mode": str(mode),
    }
