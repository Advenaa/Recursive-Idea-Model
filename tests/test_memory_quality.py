from __future__ import annotations

from rim.core.memory_quality import adapt_memory_fold_policy


def test_adapt_memory_fold_policy_no_change_with_insufficient_data() -> None:
    max_entries, novelty_floor, max_duplicate_ratio, meta = adapt_memory_fold_policy(
        base_max_entries=12,
        base_novelty_floor=0.35,
        base_max_duplicate_ratio=0.5,
        telemetry={"fold_count": 2, "degradation_rate": 0.4},
        min_folds=4,
    )
    assert max_entries == 12
    assert novelty_floor == 0.35
    assert max_duplicate_ratio == 0.5
    assert meta["applied"] is False
    assert meta["reason"] == "insufficient_fold_data"


def test_adapt_memory_fold_policy_no_change_when_healthy() -> None:
    max_entries, novelty_floor, max_duplicate_ratio, meta = adapt_memory_fold_policy(
        base_max_entries=12,
        base_novelty_floor=0.35,
        base_max_duplicate_ratio=0.5,
        telemetry={
            "fold_count": 8,
            "degradation_rate": 0.05,
            "avg_novelty_ratio": 0.62,
            "avg_duplicate_ratio": 0.22,
        },
        min_folds=4,
    )
    assert max_entries == 12
    assert novelty_floor == 0.35
    assert max_duplicate_ratio == 0.5
    assert meta["applied"] is False
    assert meta["reason"] == "telemetry_within_thresholds"


def test_adapt_memory_fold_policy_tightens_with_degradation_pressure() -> None:
    max_entries, novelty_floor, max_duplicate_ratio, meta = adapt_memory_fold_policy(
        base_max_entries=12,
        base_novelty_floor=0.35,
        base_max_duplicate_ratio=0.5,
        telemetry={
            "fold_count": 10,
            "degradation_rate": 0.7,
            "avg_novelty_ratio": 0.2,
            "avg_duplicate_ratio": 0.8,
        },
        min_folds=4,
    )
    assert max_entries < 12
    assert novelty_floor > 0.35
    assert max_duplicate_ratio < 0.5
    assert meta["applied"] is True
    assert meta["reason"] == "degradation_guardrail_adjustment"
    assert float(meta["quality_pressure"]) > 0.0
