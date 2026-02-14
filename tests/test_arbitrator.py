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


class MultiRoundRouter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke_json(self, stage: str, prompt: str, json_schema=None):  # noqa: ANN001, ANN201
        self.calls.append(stage)
        if stage == "critic_arbitration":
            return (
                {
                    "node_id": "n1",
                    "resolved_issue": "Escalate for deeper review",
                    "rationale": "Primary arbitration confidence is low.",
                    "action": "escalate",
                    "confidence": 0.41,
                },
                "codex",
            )
        if stage == "critic_arbitration_devil":
            return (
                {
                    "node_id": "n1",
                    "resolved_issue": "Merge into one concrete rollout blocker",
                    "rationale": "Devil pass shows concerns are overlapping.",
                    "action": "merge",
                    "confidence": 0.86,
                },
                "claude",
            )
        raise AssertionError("unexpected stage")


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


def test_run_arbitration_runs_devils_advocate_round() -> None:
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
    router = MultiRoundRouter()
    arbitrations, providers = asyncio.run(
        run_arbitration(
            router,
            reconciliation=reconciliation,
            findings=findings,
            max_jobs=2,
            devils_advocate_rounds=1,
            devils_advocate_min_confidence=0.72,
        )
    )
    assert router.calls == ["critic_arbitration", "critic_arbitration_devil"]
    assert len(arbitrations) == 2
    assert arbitrations[0]["round"] == "primary"
    assert arbitrations[1]["round"] == "devil_1"
    assert arbitrations[1]["action"] == "merge"
    assert providers == ["codex", "claude"]
