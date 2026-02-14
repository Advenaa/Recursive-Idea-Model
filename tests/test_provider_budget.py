from __future__ import annotations

import asyncio

import pytest

from rim.providers.base import BudgetExceededError, ProviderResult
from rim.providers.router import ProviderRunSession, RunBudget


class FakeAdapter:
    async def invoke(self, prompt, config):  # noqa: ANN001, ANN201
        return ProviderResult(
            text="ok",
            raw_output="ok",
            latency_ms=10,
            estimated_tokens_in=5,
            estimated_tokens_out=5,
            provider="codex",
            exit_code=0,
        )

    async def invoke_json_with_result(self, prompt, config, json_schema=None):  # noqa: ANN001, ANN201
        return (
            {"ok": True},
            ProviderResult(
                text='{"ok": true}',
                raw_output='{"ok": true}',
                latency_ms=10,
                estimated_tokens_in=5,
                estimated_tokens_out=5,
                provider="codex",
                exit_code=0,
            ),
        )


def test_run_budget_enforced() -> None:
    session = ProviderRunSession(
        run_id="run-1",
        providers={"codex": FakeAdapter()},
        stage_policy={"decompose": ["codex"]},
        timeout_sec=60,
        budget=RunBudget(
            max_calls=1,
            max_latency_ms=100,
            max_tokens=100,
            max_estimated_cost_usd=10.0,
        ),
    )
    payload, provider = asyncio.run(session.invoke_json("decompose", "x"))
    assert payload["ok"] is True
    assert provider == "codex"
    with pytest.raises(BudgetExceededError):
        asyncio.run(session.invoke_json("decompose", "y"))
