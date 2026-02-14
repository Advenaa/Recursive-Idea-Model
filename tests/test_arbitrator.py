from __future__ import annotations

import asyncio

from rim.agents.arbitrator import run_arbitration
from rim.core.schemas import CriticFinding


class FakeRouter:
    async def invoke_json(self, stage: str, prompt: str, json_schema=None):  # noqa: ANN001, ANN201
        if stage != "critic_arbitration":
            raise AssertionError("unexpected stage")
        return (
            {
                "node_id": "n1",
                "resolved_issue": "Merged related issues into one prioritized risk",
                "rationale": "Both critiques target rollout feasibility.",
                "action": "merge",
                "confidence": 0.82,
            },
            "claude",
        )


def test_run_arbitration_resolves_disagreement() -> None:
    findings = [
        CriticFinding(
            node_id="n1",
            critic_type="logic",
            issue="No rollout criteria",
            severity="high",
            confidence=0.8,
            suggested_fix="Define criteria",
            provider="codex",
        ),
        CriticFinding(
            node_id="n1",
            critic_type="execution",
            issue="Unclear deployment gating",
            severity="high",
            confidence=0.7,
            suggested_fix="Add gates",
            provider="claude",
        ),
    ]
    reconciliation = {
        "disagreements": [
            {
                "node_id": "n1",
                "issue_count": 2,
                "issues": ["No rollout criteria", "Unclear deployment gating"],
                "critic_types": ["logic", "execution"],
            }
        ]
    }

    arbitrations, providers = asyncio.run(
        run_arbitration(
            FakeRouter(),
            reconciliation=reconciliation,
            findings=findings,
            max_jobs=2,
        )
    )
    assert len(arbitrations) == 1
    assert arbitrations[0]["action"] == "merge"
    assert providers == ["claude"]
