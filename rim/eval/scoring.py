from __future__ import annotations

DEFAULT_RUBRIC_WEIGHTS = {
    "rigor": 0.4,
    "novelty": 0.3,
    "practicality": 0.3,
}

DOMAIN_RUBRIC_WEIGHTS = {
    "general": DEFAULT_RUBRIC_WEIGHTS,
    "finance": {"rigor": 0.52, "novelty": 0.16, "practicality": 0.32},
    "healthcare": {"rigor": 0.5, "novelty": 0.16, "practicality": 0.34},
    "legal": {"rigor": 0.56, "novelty": 0.12, "practicality": 0.32},
    "enterprise": {"rigor": 0.45, "novelty": 0.2, "practicality": 0.35},
    "developer_tools": {"rigor": 0.36, "novelty": 0.28, "practicality": 0.36},
    "consumer": {"rigor": 0.3, "novelty": 0.34, "practicality": 0.36},
    "education": {"rigor": 0.42, "novelty": 0.22, "practicality": 0.36},
}

DOMAIN_ALIASES = {
    "fintech": "finance",
    "banking": "finance",
    "medical": "healthcare",
    "health": "healthcare",
    "medtech": "healthcare",
    "policy": "legal",
    "compliance": "legal",
    "b2b": "enterprise",
    "saas": "enterprise",
    "devtools": "developer_tools",
    "developer": "developer_tools",
    "productivity": "consumer",
    "consumer_tech": "consumer",
    "edtech": "education",
}


def resolve_rubric_domain(domain: str | None) -> str:
    normalized = str(domain or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return "general"
    canonical = DOMAIN_ALIASES.get(normalized, normalized)
    if canonical in DOMAIN_RUBRIC_WEIGHTS:
        return canonical
    return "general"


def rubric_weights(domain: str | None = None) -> dict[str, float]:
    return DOMAIN_RUBRIC_WEIGHTS[resolve_rubric_domain(domain)]


def weighted_quality_score(
    rigor: float,
    novelty: float,
    practicality: float,
    domain: str | None = None,
) -> float:
    weights = rubric_weights(domain)
    score = (
        (weights["rigor"] * rigor)
        + (weights["novelty"] * novelty)
        + (weights["practicality"] * practicality)
    )
    return max(0.0, min(1.0, score))
