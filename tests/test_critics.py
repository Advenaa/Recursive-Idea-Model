from __future__ import annotations

import asyncio

import pytest

from rim.agents.critics import run_critics
from rim.core.modes import get_mode_settings
from rim.core.schemas import DecompositionNode
from rim.providers.base import BudgetExceededError


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


class RemainingBudgetRouter(CaptureRouter):
    def __init__(self, *, calls: int, latency_ms: int) -> None:
        super().__init__()
        self._remaining_calls = calls
        self._remaining_latency_ms = latency_ms

    def get_remaining_budget(self) -> dict[str, int]:
        return {
            "calls": self._remaining_calls,
            "latency_ms": self._remaining_latency_ms,
            "tokens": 1000,
        }


class BudgetExceededRouter:
    async def invoke_json(self, stage: str, prompt: str, json_schema=None):  # noqa: ANN001, ANN201
        raise BudgetExceededError("budget exceeded")


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


def test_critic_jobs_are_capped_by_env(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("RIM_MAX_CRITIC_JOBS_PER_CYCLE", "3")
    router = CaptureRouter()
    settings = get_mode_settings("deep")
    nodes = [
        DecompositionNode(
            depth=0,
            component_text=f"component {idx}",
            node_type="claim",
            confidence=0.6,
        )
        for idx in range(4)
    ]
    findings = asyncio.run(
        run_critics(
            router=router,  # type: ignore[arg-type]
            nodes=nodes,
            settings=settings,
            domain=None,
        )
    )
    assert len(findings) == 3
    assert len(router.stages) == 3


def test_critic_jobs_use_remaining_budget_cap() -> None:
    router = RemainingBudgetRouter(calls=2, latency_ms=999_999)
    settings = get_mode_settings("deep")
    node = DecompositionNode(
        depth=0,
        component_text="component",
        node_type="claim",
        confidence=0.6,
    )
    findings = asyncio.run(
        run_critics(
            router=router,  # type: ignore[arg-type]
            nodes=[node],
            settings=settings,
            domain=None,
            extra_critics=[("critic_security", "security"), ("critic_finance", "finance")],
        )
    )
    assert len(findings) == 2
    assert len(router.stages) == 2


def test_critic_budget_exceeded_error_propagates() -> None:
    settings = get_mode_settings("fast")
    node = DecompositionNode(
        depth=0,
        component_text="component",
        node_type="claim",
        confidence=0.5,
    )
    with pytest.raises(BudgetExceededError):
        asyncio.run(
            run_critics(
                router=BudgetExceededRouter(),  # type: ignore[arg-type]
                nodes=[node],
                settings=settings,
                domain=None,
            )
        )
