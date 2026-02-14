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


class FlakyJsonAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    async def invoke_json_with_result(self, prompt, config, json_schema=None):  # noqa: ANN001, ANN201
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            raise ValueError("malformed json")
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


class AlwaysFailJsonAdapter:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke_json_with_result(self, prompt, config, json_schema=None):  # noqa: ANN001, ANN201
        self.calls += 1
        raise ValueError("still malformed json")


class SuccessJsonAdapter:
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.calls = 0

    async def invoke_json_with_result(self, prompt, config, json_schema=None):  # noqa: ANN001, ANN201
        self.calls += 1
        return (
            {"ok": True},
            ProviderResult(
                text='{"ok": true}',
                raw_output='{"ok": true}',
                latency_ms=5,
                estimated_tokens_in=2,
                estimated_tokens_out=2,
                provider=self.provider_name,
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


def test_json_repair_retry_before_fallback() -> None:
    flaky = FlakyJsonAdapter()
    backup = SuccessJsonAdapter(provider_name="claude")
    session = ProviderRunSession(
        run_id="run-2",
        providers={"codex": flaky, "claude": backup},
        stage_policy={"decompose": ["codex", "claude"]},
        timeout_sec=60,
        budget=RunBudget(
            max_calls=10,
            max_latency_ms=1000,
            max_tokens=1000,
            max_estimated_cost_usd=10.0,
        ),
        json_repair_attempts=1,
    )

    payload, provider = asyncio.run(session.invoke_json("decompose", "root prompt"))
    assert payload["ok"] is True
    assert provider == "codex"
    assert flaky.calls == 2
    assert backup.calls == 0
    assert "Return one strict JSON object only." in flaky.prompts[1]


def test_json_fallback_after_repair_exhausted() -> None:
    failing = AlwaysFailJsonAdapter()
    backup = SuccessJsonAdapter(provider_name="claude")
    session = ProviderRunSession(
        run_id="run-3",
        providers={"codex": failing, "claude": backup},
        stage_policy={"decompose": ["codex", "claude"]},
        timeout_sec=60,
        budget=RunBudget(
            max_calls=10,
            max_latency_ms=1000,
            max_tokens=1000,
            max_estimated_cost_usd=10.0,
        ),
        json_repair_attempts=1,
    )

    payload, provider = asyncio.run(session.invoke_json("decompose", "root prompt"))
    assert payload["ok"] is True
    assert provider == "claude"
    assert failing.calls == 2
    assert backup.calls == 1
