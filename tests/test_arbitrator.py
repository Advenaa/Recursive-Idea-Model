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
        self.prompts: dict[str, str] = {}

    async def invoke_json(self, stage: str, prompt: str, json_schema=None):  # noqa: ANN001, ANN201
        self.calls.append(stage)
        self.prompts[stage] = prompt
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
        if stage == "critic_arbitration_specialist":
            return (
                {
                    "node_id": "n1",
                    "resolved_issue": "Specialist review confirms concrete rollout blocker",
                    "rationale": "Specialist loop adds stronger role diversity judgment.",
                    "action": "merge",
                    "confidence": 0.9,
                },
                "codex",
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


def test_run_arbitration_runs_specialist_round_for_flagged_nodes() -> None:
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
        CriticFinding(
            node_id="n1",
            critic_type="security",
            issue="Rollback path lacks abuse-case controls",
            severity="high",
            confidence=0.74,
            suggested_fix="Threat model rollback abuse vectors",
            provider="codex",
        ),
    ]
    reconciliation = {
        "disagreements": [
            {
                "node_id": "n1",
                "issue_count": 3,
                "issues": [
                    "No rollout criteria",
                    "Unclear deployment gating",
                    "Rollback path lacks abuse-case controls",
                ],
                "critic_types": ["logic", "execution", "security"],
            }
        ],
        "diversity_guardrails": {
            "flagged_nodes": [
                {
                    "node_id": "n1",
                    "unique_critics": 2,
                    "total_findings": 2,
                    "dominant_critic": "logic",
                    "dominant_share": 0.5,
                }
            ]
        },
    }
    specialist_contracts = [
        {
            "role": "finance",
            "stage": "critic_finance",
            "critic_type": "finance",
            "score": 2.0,
            "matched_keywords": ["budget", "margin"],
            "tool_contract": {
                "tools": ["unit_economics_model", "pricing_stress_test"],
                "routing_policy": "prioritize_margin_and_budget_constraints",
            },
        },
        {
            "role": "security",
            "stage": "critic_security",
            "critic_type": "security",
            "score": 3.3,
            "matched_keywords": ["abuse", "rollback", "controls"],
            "tool_contract": {
                "tools": ["threat_model", "policy_checklist"],
                "routing_policy": "prioritize_high_severity_and_compliance_constraints",
            },
        },
    ]
    router = MultiRoundRouter()
    arbitrations, providers = asyncio.run(
        run_arbitration(
            router,
            reconciliation=reconciliation,
            findings=findings,
            max_jobs=2,
            specialist_loop_enabled=True,
            specialist_max_jobs=2,
            specialist_min_confidence=0.95,
            specialist_contracts=specialist_contracts,
        )
    )
    assert "critic_arbitration_specialist" in router.calls
    specialist_round = next(item for item in arbitrations if item.get("round") == "specialist")
    assert specialist_round["specialist_role"] == "security"
    assert specialist_round["specialist_critic_type"] == "security"
    assert specialist_round["specialist_stage"] == "critic_security"
    assert specialist_round["specialist_routing_policy"] == (
        "prioritize_high_severity_and_compliance_constraints"
    )
    assert "threat_model" in specialist_round["specialist_tools"]
    assert float(specialist_round["specialist_match_score"]) > 0.0
    assert "Assigned specialist contract" in router.prompts["critic_arbitration_specialist"]
    assert "critic_security" in router.prompts["critic_arbitration_specialist"]
    assert providers[-1] in {"codex", "claude"}
