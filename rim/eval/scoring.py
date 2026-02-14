def weighted_quality_score(rigor: float, novelty: float, practicality: float) -> float:
    """Simple baseline scorer to be replaced with a domain rubric."""
    score = (0.4 * rigor) + (0.3 * novelty) + (0.3 * practicality)
    return max(0.0, min(1.0, score))
