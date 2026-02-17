from __future__ import annotations

from rim.core.specialist_contract_quality import derive_specialist_role_boost_adjustments


def test_derive_specialist_role_boost_adjustments_requires_min_rounds() -> None:
    adjustments, meta = derive_specialist_role_boost_adjustments(
        telemetry={
            "specialist_round_count": 1,
            "role_stats": {
                "security": {
                    "selected_count": 1,
                    "merge_rate": 1.0,
                    "escalate_rate": 0.0,
                    "avg_run_confidence": 0.9,
                    "avg_match_score": 1.8,
                }
            },
        },
        min_rounds=2,
        min_role_samples=1,
    )
    assert adjustments == {}
    assert meta["applied"] is False
    assert meta["reason"] == "insufficient_specialist_rounds"


def test_derive_specialist_role_boost_adjustments_no_change_when_balanced() -> None:
    adjustments, meta = derive_specialist_role_boost_adjustments(
        telemetry={
            "specialist_round_count": 6,
            "role_stats": {
                "security": {
                    "selected_count": 3,
                    "merge_rate": 0.5,
                    "escalate_rate": 0.5,
                    "avg_run_confidence": 0.7,
                    "avg_match_score": 1.0,
                }
            },
        },
        min_rounds=4,
        min_role_samples=2,
    )
    assert adjustments == {}
    assert meta["applied"] is False
    assert meta["reason"] == "telemetry_within_thresholds"


def test_derive_specialist_role_boost_adjustments_tunes_roles() -> None:
    adjustments, meta = derive_specialist_role_boost_adjustments(
        telemetry={
            "specialist_round_count": 8,
            "role_stats": {
                "security": {
                    "selected_count": 4,
                    "merge_rate": 1.0,
                    "escalate_rate": 0.0,
                    "avg_run_confidence": 0.9,
                    "avg_match_score": 2.0,
                },
                "finance": {
                    "selected_count": 4,
                    "merge_rate": 0.0,
                    "escalate_rate": 1.0,
                    "avg_run_confidence": 0.5,
                    "avg_match_score": 0.2,
                },
            },
        },
        min_rounds=4,
        min_role_samples=2,
    )
    assert meta["applied"] is True
    assert meta["reason"] == "specialist_role_boost_adjustment"
    assert float(meta["quality_pressure"]) > 0.0
    assert adjustments["security"] > 0.0
    assert adjustments["finance"] < 0.0
