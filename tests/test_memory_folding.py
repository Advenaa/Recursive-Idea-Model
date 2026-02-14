from rim.core.memory_folding import fold_cycle_memory, fold_to_memory_entries
from rim.core.schemas import CriticFinding


def _finding(critic_type: str, issue: str, severity: str = "medium") -> CriticFinding:
    return CriticFinding(
        node_id="n1",
        critic_type=critic_type,
        issue=issue,
        severity=severity,
        confidence=0.7,
        suggested_fix="fix",
        provider="codex",
    )


def test_fold_cycle_memory_produces_tripartite_sections() -> None:
    synthesis = {
        "synthesized_idea": "Ship a phased rollout with explicit safeguards and controls.",
        "changes_summary": ["Added phased rollout", "Added safeguards"],
        "residual_risks": ["Vendor dependency risk"],
    }
    payload = fold_cycle_memory(
        cycle=2,
        prior_context=["past memory A", "past memory B"],
        synthesis=synthesis,
        findings=[
            _finding("logic", "Issue A"),
            _finding("evidence", "Issue B"),
            _finding("logic", "Issue C"),
        ],
        max_entries=10,
    )
    assert payload["episodic"]
    assert payload["working"]
    assert payload["tool"]
    assert len(payload["folded_context"]) <= 10
    assert any("Cycle 2 summary" in entry for entry in payload["folded_context"])
    assert payload["fold_version"] == "v2"
    assert "quality" in payload
    assert "novelty_ratio" in payload["quality"]


def test_fold_to_memory_entries_maps_sections_to_entry_types() -> None:
    fold_payload = {
        "episodic": ["Cycle 1 summary: x"],
        "working": ["Cycle 1 open risk: y"],
        "tool": ["Cycle 1 critic signal: logic x2"],
    }
    entries = fold_to_memory_entries(fold_payload, domain="finance")
    types = [item["entry_type"] for item in entries]
    assert "episodic" in types
    assert "working" in types
    assert "tool" in types
    assert all(item["domain"] == "finance" for item in entries)


def test_fold_cycle_memory_flags_degradation_when_context_is_stale() -> None:
    synthesis = {
        "synthesized_idea": "Same idea",
        "changes_summary": [],
        "residual_risks": [],
    }
    prior_context = [
        "Cycle 2 summary: Same idea",
        "Cycle 2 open risk: none",
        "Cycle 2 critic signal: logic x1",
        "Cycle 2 summary: Same idea",
    ]
    payload = fold_cycle_memory(
        cycle=2,
        prior_context=prior_context,
        synthesis=synthesis,
        findings=[_finding("logic", "Issue A")],
        max_entries=6,
        novelty_floor=0.8,
        max_duplicate_ratio=0.1,
    )
    quality = payload["quality"]
    assert quality["degradation_detected"] is True
    assert quality["duplicate_ratio"] >= 0.1
