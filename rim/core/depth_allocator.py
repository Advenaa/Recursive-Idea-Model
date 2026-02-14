from __future__ import annotations

from dataclasses import dataclass

from rim.core.schemas import CriticFinding


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class DepthAllocatorDecision:
    recurse: bool
    reason: str
    cycle: int
    next_cycle: int | None
    confidence_score: float
    high_severity_findings: int
    critical_findings: int
    residual_risk_count: int
    confidence_delta: float | None = None


def severity_counts(findings: list[CriticFinding]) -> tuple[int, int]:
    high = 0
    critical = 0
    for finding in findings:
        if finding.severity == "critical":
            critical += 1
            high += 1
        elif finding.severity == "high":
            high += 1
    return high, critical


def decide_next_cycle(
    *,
    cycle: int,
    max_cycles: int,
    confidence_score: float,
    residual_risk_count: int,
    high_severity_findings: int,
    critical_findings: int,
    previous_confidence: float | None,
    min_confidence_to_stop: float,
    max_residual_risks_to_stop: int,
    max_high_findings_to_stop: int,
) -> DepthAllocatorDecision:
    confidence = _clamp_float(confidence_score, 0.0, 1.0)
    delta = None
    if previous_confidence is not None:
        delta = round(confidence - _clamp_float(previous_confidence, 0.0, 1.0), 4)

    if cycle >= max_cycles:
        return DepthAllocatorDecision(
            recurse=False,
            reason="max_cycles_reached",
            cycle=cycle,
            next_cycle=None,
            confidence_score=confidence,
            high_severity_findings=high_severity_findings,
            critical_findings=critical_findings,
            residual_risk_count=residual_risk_count,
            confidence_delta=delta,
        )

    if critical_findings > 0:
        return DepthAllocatorDecision(
            recurse=True,
            reason="critical_findings_present",
            cycle=cycle,
            next_cycle=cycle + 1,
            confidence_score=confidence,
            high_severity_findings=high_severity_findings,
            critical_findings=critical_findings,
            residual_risk_count=residual_risk_count,
            confidence_delta=delta,
        )

    if confidence < min_confidence_to_stop and residual_risk_count > 0:
        return DepthAllocatorDecision(
            recurse=True,
            reason="low_confidence_with_residual_risk",
            cycle=cycle,
            next_cycle=cycle + 1,
            confidence_score=confidence,
            high_severity_findings=high_severity_findings,
            critical_findings=critical_findings,
            residual_risk_count=residual_risk_count,
            confidence_delta=delta,
        )

    if high_severity_findings > max_high_findings_to_stop:
        return DepthAllocatorDecision(
            recurse=True,
            reason="high_severity_pressure",
            cycle=cycle,
            next_cycle=cycle + 1,
            confidence_score=confidence,
            high_severity_findings=high_severity_findings,
            critical_findings=critical_findings,
            residual_risk_count=residual_risk_count,
            confidence_delta=delta,
        )

    if residual_risk_count > max_residual_risks_to_stop:
        return DepthAllocatorDecision(
            recurse=True,
            reason="residual_risk_pressure",
            cycle=cycle,
            next_cycle=cycle + 1,
            confidence_score=confidence,
            high_severity_findings=high_severity_findings,
            critical_findings=critical_findings,
            residual_risk_count=residual_risk_count,
            confidence_delta=delta,
        )

    return DepthAllocatorDecision(
        recurse=False,
        reason="stability_reached",
        cycle=cycle,
        next_cycle=None,
        confidence_score=confidence,
        high_severity_findings=high_severity_findings,
        critical_findings=critical_findings,
        residual_risk_count=residual_risk_count,
        confidence_delta=delta,
    )
