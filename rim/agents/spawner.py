from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")
_DYNAMIC_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{3,32}$")
_DYNAMIC_STOPWORDS = {
    "need",
    "must",
    "should",
    "with",
    "from",
    "into",
    "that",
    "this",
    "have",
    "will",
    "could",
    "would",
    "across",
    "general",
}

_ROLE_RULES: list[tuple[str, list[str], str, str]] = [
    ("security", ["security", "threat", "privacy", "auth", "compliance"], "critic_security", "security"),
    ("finance", ["budget", "cost", "revenue", "pricing", "margin"], "critic_finance", "finance"),
    (
        "scalability",
        ["latency", "throughput", "scale", "scalable", "performance", "capacity"],
        "critic_scalability",
        "scalability",
    ),
    ("ux", ["ux", "usability", "adoption", "onboarding", "retention"], "critic_ux", "ux"),
]

_ROLE_TOOL_CONTRACTS: dict[str, dict[str, Any]] = {
    "security": {
        "tools": ["threat_model", "policy_checklist", "abuse_case_review"],
        "routing_policy": "prioritize_high_severity_and_compliance_constraints",
    },
    "finance": {
        "tools": ["unit_economics_model", "sensitivity_analysis", "pricing_stress_test"],
        "routing_policy": "prioritize_margin_and_budget_constraints",
    },
    "scalability": {
        "tools": ["capacity_profile", "latency_budget", "failure_mode_envelope"],
        "routing_policy": "prioritize_runtime_and_reliability_constraints",
    },
    "ux": {
        "tools": ["journey_map", "onboarding_friction_scan", "retention_hypothesis_grid"],
        "routing_policy": "prioritize_adoption_and_clarity_constraints",
    },
}

_DOMAIN_ROLE_HINTS = {
    "finance": "finance",
    "fintech": "finance",
    "security": "security",
    "privacy": "security",
    "compliance": "security",
    "scalability": "scalability",
    "performance": "scalability",
    "ops": "scalability",
    "product": "ux",
    "consumer": "ux",
    "education": "ux",
}


def _token_counts(text: str) -> Counter[str]:
    return Counter(match.group(0) for match in _WORD_RE.finditer(str(text).lower()))


def _parse_int_env(name: str, default: int, *, lower: int, upper: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw)) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(lower, min(upper, value))


def _parse_float_env(name: str, default: float, *, lower: float, upper: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(str(raw)) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(lower, min(upper, value))


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _coerce_int(value: object, default: int, *, lower: int, upper: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(lower, min(upper, parsed))


def _coerce_float(value: object, default: float, *, lower: float, upper: float) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        parsed = float(default)
    return max(lower, min(upper, parsed))


def _coerce_string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, str] = {}
    for key, item in value.items():
        role = str(key or "").strip().lower()
        text = str(item or "").strip()
        if role and text:
            parsed[role] = text
    return parsed


def _coerce_float_map(value: object, *, lower: float, upper: float) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, item in value.items():
        role = str(key or "").strip().lower()
        if not role:
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        parsed[role] = round(max(lower, min(upper, number)), 4)
    return parsed


def _coerce_tools(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    parsed: list[str] = []
    for item in value:
        tool = str(item or "").strip()
        if not tool or tool in seen:
            continue
        seen.add(tool)
        parsed.append(tool)
    return parsed


def _coerce_tool_map(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, list[str]] = {}
    for key, item in value.items():
        role = str(key or "").strip().lower()
        tools = _coerce_tools(item)
        if role and tools:
            parsed[role] = tools
    return parsed


def _parse_json_env(name: str) -> object | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


def _parse_float_map_env(
    name: str,
    default: dict[str, float],
    *,
    lower: float,
    upper: float,
) -> dict[str, float]:
    payload = _parse_json_env(name)
    if payload is None:
        return dict(default)
    return _coerce_float_map(payload, lower=lower, upper=upper)


def _parse_string_map_env(name: str, default: dict[str, str]) -> dict[str, str]:
    payload = _parse_json_env(name)
    if payload is None:
        return dict(default)
    return _coerce_string_map(payload)


def _parse_tool_map_env(name: str, default: dict[str, list[str]]) -> dict[str, list[str]]:
    payload = _parse_json_env(name)
    if payload is None:
        return {key: list(value) for key, value in default.items()}
    return _coerce_tool_map(payload)


def _extract_policy_env(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    direct = {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and key.startswith("RIM_")
    }
    if direct:
        return direct
    for key in ("policy_env", "recommended_env", "policy", "calibration"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            extracted = _extract_policy_env(nested)
            if extracted:
                return extracted
    return {}


def _load_spawn_policy_env(path_value: str) -> tuple[dict[str, object], str | None]:
    path = str(path_value or "").strip()
    if not path:
        return {}, None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.loads(handle.read())
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)
    env = _extract_policy_env(payload)
    allowed = {
        "RIM_SPAWN_MIN_ROLE_SCORE",
        "RIM_SPAWN_MAX_SPECIALISTS_DEEP",
        "RIM_SPAWN_MAX_SPECIALISTS_FAST",
        "RIM_ENABLE_DYNAMIC_SPECIALISTS",
        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS",
        "RIM_SPAWN_ROLE_BOOSTS",
        "RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS",
        "RIM_SPAWN_ROLE_ROUTING_OVERRIDES",
        "RIM_SPAWN_ROLE_TOOL_OVERRIDES",
    }
    filtered = {key: value for key, value in env.items() if key in allowed}
    if not filtered:
        return {}, "No spawn-policy keys found in policy file."
    return filtered, None


def _domain_hint_role(domain: str | None) -> str | None:
    normalized = str(domain or "").strip().lower()
    if not normalized:
        return None
    for key, role in _DOMAIN_ROLE_HINTS.items():
        if key in normalized:
            return role
    return None


def _role_score(
    *,
    role_name: str,
    keywords: list[str],
    domain_counts: Counter[str],
    constraint_counts: Counter[str],
    memory_counts: Counter[str],
    domain_hint_role: str | None,
    role_score_boost: float = 0.0,
) -> tuple[float, list[str], dict[str, int]]:
    domain_hits = sum(domain_counts.get(keyword, 0) for keyword in keywords)
    constraint_hits = sum(constraint_counts.get(keyword, 0) for keyword in keywords)
    memory_hits = sum(memory_counts.get(keyword, 0) for keyword in keywords)
    matched_keywords = [
        keyword
        for keyword in keywords
        if domain_counts.get(keyword, 0)
        or constraint_counts.get(keyword, 0)
        or memory_counts.get(keyword, 0)
    ]
    role_boost = 1.0 if domain_hint_role == role_name else 0.0
    score = (
        (2.0 * float(domain_hits))
        + (1.0 * float(constraint_hits))
        + (0.7 * float(memory_hits))
        + role_boost
        + float(role_score_boost)
    )
    return score, matched_keywords, {
        "domain_hits": domain_hits,
        "constraint_hits": constraint_hits,
        "memory_hits": memory_hits,
    }


def _extract_dynamic_tokens(
    *,
    domain_counts: Counter[str],
    constraint_counts: Counter[str],
    memory_counts: Counter[str],
    known_keywords: set[str],
    max_tokens: int,
    token_boosts: dict[str, float] | None = None,
) -> list[tuple[str, float, dict[str, int]]]:
    boosts = token_boosts or {}
    combined: Counter[str] = Counter()
    combined.update(domain_counts)
    combined.update(constraint_counts)
    combined.update(memory_counts)
    candidates: list[tuple[str, float, dict[str, int]]] = []
    for token, count in combined.items():
        if token in known_keywords:
            continue
        if token in _DYNAMIC_STOPWORDS:
            continue
        if not _DYNAMIC_TOKEN_RE.match(token):
            continue
        domain_hits = int(domain_counts.get(token, 0))
        constraint_hits = int(constraint_counts.get(token, 0))
        memory_hits = int(memory_counts.get(token, 0))
        boost = float(boosts.get(token, 0.0))
        score = (
            (2.0 * domain_hits)
            + (1.0 * constraint_hits)
            + (0.7 * memory_hits)
            + (0.1 * count)
            + boost
        )
        if score <= 0.0:
            continue
        candidates.append(
            (
                token,
                round(float(score), 3),
                {
                    "domain_hits": domain_hits,
                    "constraint_hits": constraint_hits,
                    "memory_hits": memory_hits,
                },
            )
        )
    candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return candidates[: max(0, int(max_tokens))]


def build_spawn_plan(
    *,
    mode: str,
    domain: str | None,
    constraints: list[str] | None,
    memory_context: list[str] | None = None,
) -> dict[str, Any]:
    domain_counts = _token_counts(str(domain or ""))
    constraint_counts = _token_counts(" ".join(str(item) for item in list(constraints or [])))
    memory_counts = _token_counts(" ".join(str(item) for item in list(memory_context or [])[:6]))
    hint_role = _domain_hint_role(domain)
    spawn_policy_path = str(os.getenv("RIM_SPAWN_POLICY_PATH", "")).strip()
    spawn_policy_env: dict[str, object] = {}
    spawn_policy_error: str | None = None
    if spawn_policy_path:
        spawn_policy_env, spawn_policy_error = _load_spawn_policy_env(spawn_policy_path)
    min_role_score_default = 1.0
    dynamic_enabled_default = True
    max_dynamic_specialists_default = 2
    max_specialists_deep_default = 3
    max_specialists_fast_default = 1
    role_boosts_default: dict[str, float] = {}
    dynamic_token_boosts_default: dict[str, float] = {}
    role_routing_overrides_default: dict[str, str] = {}
    role_tool_overrides_default: dict[str, list[str]] = {}
    if spawn_policy_env:
        min_role_score_default = _coerce_float(
            spawn_policy_env.get("RIM_SPAWN_MIN_ROLE_SCORE"),
            min_role_score_default,
            lower=0.0,
            upper=20.0,
        )
        dynamic_enabled_default = _coerce_bool(
            spawn_policy_env.get("RIM_ENABLE_DYNAMIC_SPECIALISTS"),
            dynamic_enabled_default,
        )
        max_dynamic_specialists_default = _coerce_int(
            spawn_policy_env.get("RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS"),
            max_dynamic_specialists_default,
            lower=0,
            upper=6,
        )
        max_specialists_deep_default = _coerce_int(
            spawn_policy_env.get("RIM_SPAWN_MAX_SPECIALISTS_DEEP"),
            max_specialists_deep_default,
            lower=1,
            upper=8,
        )
        max_specialists_fast_default = _coerce_int(
            spawn_policy_env.get("RIM_SPAWN_MAX_SPECIALISTS_FAST"),
            max_specialists_fast_default,
            lower=1,
            upper=4,
        )
        role_boosts_default = _coerce_float_map(
            spawn_policy_env.get("RIM_SPAWN_ROLE_BOOSTS"),
            lower=-4.0,
            upper=4.0,
        )
        dynamic_token_boosts_default = _coerce_float_map(
            spawn_policy_env.get("RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS"),
            lower=-4.0,
            upper=4.0,
        )
        role_routing_overrides_default = _coerce_string_map(
            spawn_policy_env.get("RIM_SPAWN_ROLE_ROUTING_OVERRIDES")
        )
        role_tool_overrides_default = _coerce_tool_map(
            spawn_policy_env.get("RIM_SPAWN_ROLE_TOOL_OVERRIDES")
        )
    min_role_score = _parse_float_env(
        "RIM_SPAWN_MIN_ROLE_SCORE",
        min_role_score_default,
        lower=0.0,
        upper=20.0,
    )
    enable_dynamic_specialists = _parse_bool_env(
        "RIM_ENABLE_DYNAMIC_SPECIALISTS",
        dynamic_enabled_default,
    )
    max_dynamic_specialists = _parse_int_env(
        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS",
        max_dynamic_specialists_default,
        lower=0,
        upper=6,
    )
    role_boosts = _parse_float_map_env(
        "RIM_SPAWN_ROLE_BOOSTS",
        role_boosts_default,
        lower=-4.0,
        upper=4.0,
    )
    dynamic_token_boosts = _parse_float_map_env(
        "RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS",
        dynamic_token_boosts_default,
        lower=-4.0,
        upper=4.0,
    )
    role_routing_overrides = _parse_string_map_env(
        "RIM_SPAWN_ROLE_ROUTING_OVERRIDES",
        role_routing_overrides_default,
    )
    role_tool_overrides = _parse_tool_map_env(
        "RIM_SPAWN_ROLE_TOOL_OVERRIDES",
        role_tool_overrides_default,
    )
    scored_candidates: list[dict[str, Any]] = []
    known_keywords = {
        keyword
        for _, keywords, _, _ in _ROLE_RULES
        for keyword in keywords
    }
    for role_name, keywords, stage, critic_type in _ROLE_RULES:
        score_boost = float(role_boosts.get(role_name, 0.0))
        score, matched_keywords, evidence = _role_score(
            role_name=role_name,
            keywords=keywords,
            domain_counts=domain_counts,
            constraint_counts=constraint_counts,
            memory_counts=memory_counts,
            domain_hint_role=hint_role,
            role_score_boost=score_boost,
        )
        if score < min_role_score:
            continue
        base_contract = _ROLE_TOOL_CONTRACTS.get(role_name, {"tools": [], "routing_policy": "generic"})
        tools = _coerce_tools(base_contract.get("tools"))
        routing_policy = str(base_contract.get("routing_policy") or "generic")
        routing_override = role_routing_overrides.get(role_name)
        tools_override = role_tool_overrides.get(role_name)
        if routing_override:
            routing_policy = routing_override
        if tools_override:
            tools = list(tools_override)
        scored_candidates.append(
            {
                "role": role_name,
                "stage": stage,
                "critic_type": critic_type,
                "score": round(score, 3),
                "matched_keywords": matched_keywords[:5],
                "evidence": evidence,
                "tool_contract": {
                    "tools": tools,
                    "routing_policy": routing_policy,
                },
                "policy_score_boost": round(score_boost, 4),
            }
        )

    if enable_dynamic_specialists:
        dynamic_tokens = _extract_dynamic_tokens(
            domain_counts=domain_counts,
            constraint_counts=constraint_counts,
            memory_counts=memory_counts,
            known_keywords=known_keywords,
            max_tokens=max_dynamic_specialists,
            token_boosts=dynamic_token_boosts,
        )
        for token, score, evidence in dynamic_tokens:
            if score < min_role_score:
                continue
            dynamic_role = f"dynamic_{token}"
            routing_policy = (
                role_routing_overrides.get(dynamic_role)
                or role_routing_overrides.get(token)
                or "prioritize_domain_specific_signals"
            )
            tools = role_tool_overrides.get(dynamic_role) or role_tool_overrides.get(token) or [
                f"context_probe:{token}",
                "evidence_scan",
                "counterexample_search",
            ]
            scored_candidates.append(
                {
                    "role": dynamic_role,
                    "stage": f"critic_dynamic_{token}",
                    "critic_type": f"dynamic_{token}",
                    "score": score,
                    "matched_keywords": [token],
                    "evidence": evidence,
                    "tool_contract": {
                        "tools": list(tools),
                        "routing_policy": routing_policy,
                    },
                    "policy_score_boost": round(float(dynamic_token_boosts.get(token, 0.0)), 4),
                    "dynamic": True,
                }
            )

    # Deep mode can carry more specialists, fast mode keeps one maximum.
    mode_normalized = str(mode).strip().lower()
    max_specialists = (
        _parse_int_env(
            "RIM_SPAWN_MAX_SPECIALISTS_DEEP",
            max_specialists_deep_default,
            lower=1,
            upper=8,
        )
        if mode_normalized == "deep"
        else _parse_int_env(
            "RIM_SPAWN_MAX_SPECIALISTS_FAST",
            max_specialists_fast_default,
            lower=1,
            upper=4,
        )
    )
    scored_candidates.sort(
        key=lambda item: (
            float(item["score"]),
            len(item["matched_keywords"]),
            item["role"],
        ),
        reverse=True,
    )
    selected = scored_candidates[:max_specialists]
    return {
        "domain": str(domain or "").strip() or None,
        "extra_critics": selected,
        "selected_count": len(selected),
        "candidate_count": len(scored_candidates),
        "min_role_score": min_role_score,
        "max_specialists": max_specialists,
        "max_dynamic_specialists": max_dynamic_specialists,
        "domain_hint_role": hint_role,
        "dynamic_enabled": enable_dynamic_specialists,
        "role_boosts": role_boosts,
        "dynamic_token_boosts": dynamic_token_boosts,
        "role_routing_overrides": role_routing_overrides,
        "role_tool_overrides": role_tool_overrides,
        "policy_applied": bool(spawn_policy_env),
        "policy_path": spawn_policy_path or None,
        "policy_error": spawn_policy_error,
        "mode": str(mode),
    }
