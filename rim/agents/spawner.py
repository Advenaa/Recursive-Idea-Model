from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")

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


def _tokens(text: str) -> set[str]:
    return {match.group(0) for match in _WORD_RE.finditer(str(text).lower())}


def build_spawn_plan(
    *,
    mode: str,
    domain: str | None,
    constraints: list[str] | None,
    memory_context: list[str] | None = None,
) -> dict[str, Any]:
    corpus = " ".join(
        [
            str(domain or ""),
            " ".join(str(item) for item in list(constraints or [])),
            " ".join(str(item) for item in list(memory_context or [])[:6]),
        ]
    ).strip()
    token_set = _tokens(corpus)
    extra_critics: list[dict[str, str]] = []

    for role_name, keywords, stage, critic_type in _ROLE_RULES:
        if any(keyword in token_set for keyword in keywords):
            extra_critics.append(
                {
                    "role": role_name,
                    "stage": stage,
                    "critic_type": critic_type,
                }
            )

    # Deep mode can carry more specialists, fast mode keeps one maximum.
    max_specialists = 3 if str(mode).strip().lower() == "deep" else 1
    selected = extra_critics[:max_specialists]
    return {
        "domain": str(domain or "").strip() or None,
        "extra_critics": selected,
        "selected_count": len(selected),
        "candidate_count": len(extra_critics),
        "mode": str(mode),
    }
