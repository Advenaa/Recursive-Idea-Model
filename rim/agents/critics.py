from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from uuid import uuid4

from rim.core.modes import ModeSettings
from rim.core.schemas import CriticFinding, DecompositionNode
from rim.providers.base import BudgetExceededError
from rim.providers.router import ProviderRouter

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "issue": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "suggested_fix": {"type": "string"},
    },
    "required": ["issue", "severity", "confidence", "suggested_fix"],
}

CRITIC_PROMPT = """You are the {critic_type} critic for one idea component.
Return STRICT JSON only with:
{{
  "issue": "string",
  "severity": "low|medium|high|critical",
  "confidence": 0.0,
  "suggested_fix": "string"
}}

Evidence requirement: {evidence_requirement}

Domain context:
{domain}

Component:
{component}
"""

DEEP_CRITICS: list[tuple[str, str]] = [
    ("critic_logic", "logic"),
    ("critic_evidence", "evidence"),
    ("critic_execution", "execution"),
    ("critic_adversarial", "adversarial"),
]
FAST_CRITICS: list[tuple[str, str]] = [
    ("critic_logic", "logic"),
    ("critic_execution", "execution"),
]

_DOMAIN_RE = re.compile(r"[^a-z0-9]+")
_JSON_ERROR_MARKERS = (
    "json",
    "parse",
    "schema",
    "structured_output",
    "no valid json object",
)


def _domain_slug(domain: str | None) -> str:
    text = str(domain or "").strip().lower()
    if not text:
        return ""
    slug = _DOMAIN_RE.sub("_", text).strip("_")
    return slug[:40]


def _domain_specialist_stage(domain: str | None) -> tuple[str, str] | None:
    slug = _domain_slug(domain)
    if not slug:
        return None
    return (f"critic_domain_{slug}", f"domain_{slug}")


def _domain_critic_enabled() -> bool:
    raw = os.getenv("RIM_ENABLE_DOMAIN_CRITIC", "1")
    value = str(raw).strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return True


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        parsed = int(str(raw if raw is not None else default))
    except (TypeError, ValueError):
        return max(1, default)
    return max(1, parsed)


def _looks_like_json_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    if not message:
        return False
    return any(marker in message for marker in _JSON_ERROR_MARKERS)


def _resolve_job_limit(
    *,
    router: ProviderRouter,
    total_jobs: int,
) -> int:
    configured_max = _parse_positive_int_env("RIM_MAX_CRITIC_JOBS_PER_CYCLE", 48)
    limit = min(total_jobs, configured_max)
    get_remaining_budget = getattr(router, "get_remaining_budget", None)
    if not callable(get_remaining_budget):
        return limit
    try:
        remaining = get_remaining_budget()
    except Exception:  # noqa: BLE001
        return limit
    if not isinstance(remaining, dict):
        return limit

    remaining_calls = int(remaining.get("calls", limit) or 0)
    limit = min(limit, max(0, remaining_calls))

    remaining_latency_ms = int(remaining.get("latency_ms", 0) or 0)
    reserve_latency_ms = _parse_positive_int_env(
        "RIM_CRITIC_RESERVE_LATENCY_MS",
        420_000,
    )
    est_latency_per_call_ms = _parse_positive_int_env(
        "RIM_ESTIMATED_CRITIC_LATENCY_MS",
        45_000,
    )
    if remaining_latency_ms > 0:
        budget_for_critics = max(0, remaining_latency_ms - reserve_latency_ms)
        latency_limited = budget_for_critics // est_latency_per_call_ms
        if budget_for_critics > 0 and latency_limited == 0:
            latency_limited = 1
        limit = min(limit, max(0, latency_limited))
    return max(0, limit)


def _select_jobs_round_robin(
    *,
    nodes: list[DecompositionNode],
    selected_critics: list[tuple[str, str]],
    max_jobs: int,
) -> list[tuple[DecompositionNode, str, str]]:
    if max_jobs <= 0:
        return []
    buckets: list[list[tuple[DecompositionNode, str, str]]] = [
        [(node, stage_name, critic_type) for stage_name, critic_type in selected_critics]
        for node in nodes
    ]
    jobs: list[tuple[DecompositionNode, str, str]] = []
    while len(jobs) < max_jobs:
        progressed = False
        for bucket in buckets:
            if not bucket:
                continue
            jobs.append(bucket.pop(0))
            progressed = True
            if len(jobs) >= max_jobs:
                break
        if not progressed:
            break
    return jobs


def _valid_severity(value: str) -> str:
    value = value.strip().lower()
    if value in {"low", "medium", "high", "critical"}:
        return value
    return "medium"


def _make_finding(
    node: DecompositionNode,
    critic_type: str,
    provider: str,
    payload: dict[str, Any],
) -> CriticFinding:
    return CriticFinding(
        id=str(uuid4()),
        node_id=node.id,
        critic_type=critic_type,
        issue=str(payload.get("issue", "Insufficient detail in component")).strip(),
        severity=_valid_severity(str(payload.get("severity", "medium"))),
        confidence=float(payload.get("confidence", 0.5)),
        suggested_fix=str(
            payload.get(
                "suggested_fix",
                "Clarify assumptions, evidence, and execution dependencies.",
            )
        ).strip(),
        provider=provider,
    )


async def run_critics(
    router: ProviderRouter,
    nodes: list[DecompositionNode],
    settings: ModeSettings,
    domain: str | None = None,
    extra_critics: list[tuple[str, str]] | None = None,
) -> list[CriticFinding]:
    base_critics = list(DEEP_CRITICS if settings.mode == "deep" else FAST_CRITICS)
    cleaned_extra_critics: list[tuple[str, str]] = []
    for item in list(extra_critics or []):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        stage_name, critic_type = item
        stage_text = str(stage_name).strip()
        critic_text = str(critic_type).strip()
        if not stage_text or not critic_text:
            continue
        pair = (stage_text, critic_text)
        if pair not in cleaned_extra_critics and pair not in base_critics:
            cleaned_extra_critics.append(pair)
    domain_stage = _domain_specialist_stage(domain)
    selected_critics = list(base_critics[: settings.critics_per_node])
    for pair in cleaned_extra_critics:
        if pair not in selected_critics:
            selected_critics.append(pair)
    if domain_stage is not None and _domain_critic_enabled() and domain_stage not in selected_critics:
        selected_critics.append(domain_stage)
    total_jobs = len(nodes) * len(selected_critics)
    if total_jobs == 0:
        return []
    job_limit = _resolve_job_limit(router=router, total_jobs=total_jobs)
    critic_jobs = _select_jobs_round_robin(
        nodes=nodes,
        selected_critics=selected_critics,
        max_jobs=job_limit,
    )
    if not critic_jobs:
        return []
    max_parallel = _parse_positive_int_env("RIM_MAX_PARALLEL_CRITICS", 6)
    max_parallel = min(max_parallel, len(critic_jobs))
    semaphore = asyncio.Semaphore(max_parallel)
    findings: list[CriticFinding] = []

    async def _job(node: DecompositionNode, stage: str, critic_type: str) -> None:
        prompt = CRITIC_PROMPT.format(
            critic_type=critic_type,
            evidence_requirement=settings.evidence_requirement,
            domain=domain or "general",
            component=node.component_text,
        )
        async with semaphore:
            try:
                payload, provider = await router.invoke_json(
                    stage,
                    prompt,
                    json_schema=CRITIC_SCHEMA,
                )
                if not isinstance(payload, dict):
                    raise ValueError("Critic stage returned non-object JSON payload.")
                findings.append(_make_finding(node, critic_type, provider, payload))
            except BudgetExceededError:
                raise
            except Exception as exc:  # noqa: BLE001
                parse_like_error = _looks_like_json_error(exc)
                findings.append(
                    CriticFinding(
                        id=str(uuid4()),
                        node_id=node.id,
                        critic_type=critic_type,
                        issue=(
                            "Critic stage failed to parse a valid response."
                            if parse_like_error
                            else "Critic stage execution failed."
                        ),
                        severity="high" if parse_like_error else "medium",
                        confidence=0.2 if parse_like_error else 0.3,
                        suggested_fix=(
                            "Retry this component with tighter JSON constraints."
                            if parse_like_error
                            else "Retry this component and inspect provider CLI logs."
                        ),
                        provider=None,
                    )
                )

    tasks = [_job(node, stage_name, critic_type) for node, stage_name, critic_type in critic_jobs]
    await asyncio.gather(*tasks)
    return findings
