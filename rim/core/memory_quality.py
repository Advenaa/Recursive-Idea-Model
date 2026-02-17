from __future__ import annotations

from typing import Any


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def adapt_memory_fold_policy(
    *,
    base_max_entries: int,
    base_novelty_floor: float,
    base_max_duplicate_ratio: float,
    telemetry: dict[str, Any] | None,
    min_folds: int = 4,
) -> tuple[int, float, float, dict[str, Any]]:
    metrics = dict(telemetry or {})
    fold_count = max(0, _to_int(metrics.get("fold_count"), 0))

    meta: dict[str, Any] = {
        "applied": False,
        "reason": "insufficient_fold_data",
        "fold_count": fold_count,
        "min_folds": max(1, int(min_folds)),
        "quality_pressure": 0.0,
        "components": {
            "degradation": 0.0,
            "low_novelty": 0.0,
            "high_duplicate": 0.0,
        },
    }
    if fold_count < max(1, int(min_folds)):
        return (
            int(base_max_entries),
            float(base_novelty_floor),
            float(base_max_duplicate_ratio),
            meta,
        )

    degradation_rate = _clamp_float(_to_float(metrics.get("degradation_rate"), 0.0), 0.0, 1.0)
    novelty_ratio = _clamp_float(_to_float(metrics.get("avg_novelty_ratio"), 0.0), 0.0, 1.0)
    duplicate_ratio = _clamp_float(_to_float(metrics.get("avg_duplicate_ratio"), 0.0), 0.0, 1.0)

    degradation_component = _clamp_float((degradation_rate - 0.2) / 0.8, 0.0, 1.0)
    low_novelty_component = _clamp_float((0.35 - novelty_ratio) / 0.35, 0.0, 1.0)
    high_duplicate_component = _clamp_float((duplicate_ratio - 0.5) / 0.5, 0.0, 1.0)

    quality_pressure = _clamp_float(
        degradation_component + (0.8 * low_novelty_component) + (0.8 * high_duplicate_component),
        0.0,
        1.5,
    )
    meta["quality_pressure"] = round(quality_pressure, 4)
    meta["components"] = {
        "degradation": round(degradation_component, 4),
        "low_novelty": round(low_novelty_component, 4),
        "high_duplicate": round(high_duplicate_component, 4),
    }
    if quality_pressure <= 0.0:
        meta["reason"] = "telemetry_within_thresholds"
        return (
            int(base_max_entries),
            float(base_novelty_floor),
            float(base_max_duplicate_ratio),
            meta,
        )

    adjusted_max_entries = _clamp_int(
        int(round(float(base_max_entries) - (4.0 * quality_pressure))),
        6,
        40,
    )
    adjusted_novelty_floor = round(
        _clamp_float(float(base_novelty_floor) + (0.10 * quality_pressure), 0.15, 0.85),
        3,
    )
    adjusted_max_duplicate_ratio = round(
        _clamp_float(float(base_max_duplicate_ratio) - (0.12 * quality_pressure), 0.15, 0.8),
        3,
    )
    meta["applied"] = True
    meta["reason"] = "degradation_guardrail_adjustment"
    return (
        adjusted_max_entries,
        adjusted_novelty_floor,
        adjusted_max_duplicate_ratio,
        meta,
    )
