from __future__ import annotations

from collections.abc import Callable

from rim.core.schemas import AnalyzeResult
from rim.eval.scoring import weighted_quality_score


def evaluate_run(
    result: AnalyzeResult,
    reviewer: Callable[[AnalyzeResult], tuple[float, float, float]],
) -> dict[str, float]:
    rigor, novelty, practicality = reviewer(result)
    quality = weighted_quality_score(rigor, novelty, practicality)
    return {
        "rigor": rigor,
        "novelty": novelty,
        "practicality": practicality,
        "quality_score": quality,
    }
