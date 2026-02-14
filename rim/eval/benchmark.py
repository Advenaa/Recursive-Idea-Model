from __future__ import annotations

from collections.abc import Callable

from rim.core.schemas import AnalyzeResult
from rim.eval.scoring import resolve_rubric_domain, rubric_weights, weighted_quality_score


def evaluate_run(
    result: AnalyzeResult,
    reviewer: Callable[[AnalyzeResult], tuple[float, float, float]],
    domain: str | None = None,
) -> dict[str, float | str | dict[str, float]]:
    rigor, novelty, practicality = reviewer(result)
    quality = weighted_quality_score(
        rigor,
        novelty,
        practicality,
        domain=domain,
    )
    resolved_domain = resolve_rubric_domain(domain)
    weights = rubric_weights(domain)
    return {
        "rigor": rigor,
        "novelty": novelty,
        "practicality": practicality,
        "quality_score": quality,
        "rubric_domain": resolved_domain,
        "weights": weights,
    }
