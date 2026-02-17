from __future__ import annotations

from typing import Any


def _clamp_float(value: float, lower: float, upper: float) -> float:
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


def derive_specialist_role_boost_adjustments(
    *,
    telemetry: dict[str, Any] | None,
    min_rounds: int = 4,
    min_role_samples: int = 2,
) -> tuple[dict[str, float], dict[str, Any]]:
    metrics = dict(telemetry or {})
    role_stats = metrics.get("role_stats")
    if not isinstance(role_stats, dict):
        role_stats = {}

    specialist_round_count = max(0, _to_int(metrics.get("specialist_round_count"), 0))
    normalized_min_rounds = max(1, int(min_rounds))
    normalized_min_role_samples = max(1, int(min_role_samples))
    meta: dict[str, Any] = {
        "applied": False,
        "reason": "insufficient_specialist_rounds",
        "specialist_round_count": specialist_round_count,
        "min_rounds": normalized_min_rounds,
        "min_role_samples": normalized_min_role_samples,
        "quality_pressure": 0.0,
        "roles_considered": 0,
        "roles_adjusted": [],
    }
    if specialist_round_count < normalized_min_rounds:
        return {}, meta

    adjustments: dict[str, float] = {}
    pressure_total = 0.0
    considered = 0
    for raw_role, raw_stats in role_stats.items():
        role = str(raw_role or "").strip().lower()
        if not role or not isinstance(raw_stats, dict):
            continue
        selected_count = max(0, _to_int(raw_stats.get("selected_count"), 0))
        if selected_count < normalized_min_role_samples:
            continue
        considered += 1
        merge_rate = _clamp_float(_to_float(raw_stats.get("merge_rate"), 0.0), 0.0, 1.0)
        escalate_rate = _clamp_float(_to_float(raw_stats.get("escalate_rate"), 0.0), 0.0, 1.0)
        avg_run_confidence = _clamp_float(
            _to_float(raw_stats.get("avg_run_confidence"), 0.0),
            0.0,
            1.0,
        )
        avg_match_score = _clamp_float(
            _to_float(raw_stats.get("avg_match_score"), 0.0),
            0.0,
            8.0,
        )
        signal = (
            (merge_rate - escalate_rate)
            + (0.8 * (avg_run_confidence - 0.70))
            + (0.10 * (avg_match_score - 1.0))
        )
        delta = _clamp_float(signal, -1.2, 1.2)
        if abs(delta) < 0.08:
            continue
        rounded_delta = round(delta, 4)
        adjustments[role] = rounded_delta
        pressure_total += abs(rounded_delta)

    meta["roles_considered"] = considered
    if not adjustments:
        meta["reason"] = "telemetry_within_thresholds"
        return {}, meta

    meta["applied"] = True
    meta["reason"] = "specialist_role_boost_adjustment"
    meta["roles_adjusted"] = sorted(adjustments.keys())
    meta["quality_pressure"] = round(pressure_total / float(len(adjustments)), 4)
    return adjustments, meta
