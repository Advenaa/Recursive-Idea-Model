from __future__ import annotations

import asyncio

from rim.agents.critics import run_critics
from rim.core.modes import get_mode_settings
from rim.core.schemas import DecompositionNode


class CaptureRouter:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.prompts: list[str] = []

    async def invoke_json(self, stage: str, prompt: str, json_schema=None):  # noqa: ANN001, ANN201
        self.stages.append(stage)
        self.prompts.append(prompt)
        return (
            {
                "issue": f"issue for {stage}",
                "severity": "medium",
                "confidence": 0.6,
                "suggested_fix": "fix",
            },
            "codex",
        )


def test_domain_specialist_critic_added_in_deep_mode(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("RIM_ENABLE_DOMAIN_CRITIC", "1")
    router = CaptureRouter()
    settings = get_mode_settings("deep")
    node = DecompositionNode(
        depth=0,
        component_text="Build compliance workflow",
        node_type="claim",
        confidence=0.5,
    )

    findings = asyncio.run(
        run_critics(
            router=router,  # type: ignore[arg-type]
            nodes=[node],
            settings=settings,
            domain="FinTech Payments",
        )
    )
    assert len(findings) == settings.critics_per_node + 1
    assert any(stage == "critic_domain_fintech_payments" for stage in router.stages)
    assert any(finding.critic_type == "domain_fintech_payments" for finding in findings)
    assert any("Domain context:\nFinTech Payments" in prompt for prompt in router.prompts)


def test_domain_specialist_critic_disabled_by_env(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("RIM_ENABLE_DOMAIN_CRITIC", "0")
    router = CaptureRouter()
    settings = get_mode_settings("deep")
    node = DecompositionNode(
        depth=0,
        component_text="Build compliance workflow",
        node_type="claim",
        confidence=0.5,
    )

    findings = asyncio.run(
        run_critics(
            router=router,  # type: ignore[arg-type]
            nodes=[node],
            settings=settings,
            domain="FinTech Payments",
        )
    )
    assert len(findings) == settings.critics_per_node
    assert all(stage != "critic_domain_fintech_payments" for stage in router.stages)


def test_extra_critics_are_included() -> None:
    router = CaptureRouter()
    settings = get_mode_settings("fast")
    node = DecompositionNode(
        depth=0,
        component_text="Improve onboarding",
        node_type="claim",
        confidence=0.5,
    )
    findings = asyncio.run(
        run_critics(
            router=router,  # type: ignore[arg-type]
            nodes=[node],
            settings=settings,
            domain=None,
            extra_critics=[("critic_ux", "ux")],
        )
    )
    assert any(stage == "critic_ux" for stage in router.stages)
    assert any(finding.critic_type == "ux" for finding in findings)
