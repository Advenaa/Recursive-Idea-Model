from __future__ import annotations

from rim.core.schemas import CriticFinding


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text).strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip() + "..."


def fold_cycle_memory(
    *,
    cycle: int,
    prior_context: list[str],
    synthesis: dict[str, object],
    findings: list[CriticFinding],
    max_entries: int = 12,
) -> dict[str, object]:
    synthesized_idea = _truncate_words(str(synthesis.get("synthesized_idea") or ""), 24)
    changes = [
        _truncate_words(str(item), 12)
        for item in list(synthesis.get("changes_summary") or [])[:3]
        if str(item).strip()
    ]
    risks = [
        _truncate_words(str(item), 14)
        for item in list(synthesis.get("residual_risks") or [])[:3]
        if str(item).strip()
    ]

    episodic = [
        f"Cycle {cycle} summary: {synthesized_idea}" if synthesized_idea else f"Cycle {cycle} summary",
    ]
    for item in changes:
        episodic.append(f"Cycle {cycle} change: {item}")

    working = [f"Cycle {cycle} open risk: {item}" for item in risks]
    if not working:
        working.append(f"Cycle {cycle} open risk: none")

    tool_critic_counts: dict[str, int] = {}
    for finding in findings:
        key = str(finding.critic_type).strip().lower() or "unknown"
        tool_critic_counts[key] = tool_critic_counts.get(key, 0) + 1
    tool = [
        f"Cycle {cycle} critic signal: {critic_type} x{count}"
        for critic_type, count in sorted(tool_critic_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    ]
    if not tool:
        tool.append(f"Cycle {cycle} critic signal: none")

    folded_context: list[str] = []
    seen: set[str] = set()
    for entry in [
        *episodic,
        *working,
        *tool,
        *list(prior_context or []),
    ]:
        text = str(entry).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        folded_context.append(text)
        if len(folded_context) >= max(4, int(max_entries)):
            break

    return {
        "episodic": episodic,
        "working": working,
        "tool": tool,
        "folded_context": folded_context,
    }


def fold_to_memory_entries(
    fold_payload: dict[str, object],
    *,
    domain: str | None = None,
) -> list[dict]:
    entries: list[dict] = []

    for item in list(fold_payload.get("episodic") or [])[:3]:
        text = str(item).strip()
        if text:
            entries.append(
                {
                    "entry_type": "episodic",
                    "entry_text": text,
                    "domain": domain,
                    "severity": "medium",
                    "score": 0.65,
                }
            )

    for item in list(fold_payload.get("working") or [])[:3]:
        text = str(item).strip()
        if text:
            severity = "high" if "risk:" in text and "none" not in text.lower() else "medium"
            entries.append(
                {
                    "entry_type": "working",
                    "entry_text": text,
                    "domain": domain,
                    "severity": severity,
                    "score": 0.7 if severity == "high" else 0.6,
                }
            )

    for item in list(fold_payload.get("tool") or [])[:3]:
        text = str(item).strip()
        if text:
            entries.append(
                {
                    "entry_type": "tool",
                    "entry_text": text,
                    "domain": domain,
                    "severity": "low",
                    "score": 0.55,
                }
            )
    return entries[:9]
