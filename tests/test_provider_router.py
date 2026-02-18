from __future__ import annotations

import asyncio

from rim.providers.base import ProviderResult
from rim.providers.router import ProviderRouter, ProviderRunSession, RunBudget


class PiOnlyJsonAdapter:
    async def invoke_json_with_result(self, prompt, config, json_schema=None):  # noqa: ANN001, ANN201
        return (
            {"ok": True},
            ProviderResult(
                text='{"ok": true}',
                raw_output='{"ok": true}',
                latency_ms=5,
                estimated_tokens_in=2,
                estimated_tokens_out=2,
                provider="pi",
                exit_code=0,
            ),
        )


def test_provider_router_defaults_pi_first(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RIM_PROVIDER_ORDER", "pi,claude,codex")
    router = ProviderRouter()
    assert router.default_provider_order[0] == "pi"
    assert router.stage_policy["decompose"][0] == "pi"
    assert "pi" in router.providers


def test_provider_router_pi_only_mode(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RIM_PI_ONLY", "1")
    router = ProviderRouter()
    assert router.pi_only is True
    assert list(router.providers.keys()) == ["pi"]
    assert router.default_provider_order == ["pi"]
    assert router.stage_policy["decompose"][0] == "pi"


def test_provider_run_session_ignores_unknown_stage_providers() -> None:
    session = ProviderRunSession(
        run_id="run-pi",
        providers={"pi": PiOnlyJsonAdapter()},
        stage_policy={"decompose": ["unknown", "pi"]},
        timeout_sec=30,
        budget=RunBudget(
            max_calls=3,
            max_latency_ms=500,
            max_tokens=1000,
            max_estimated_cost_usd=1.0,
        ),
        default_provider_order=["pi"],
    )
    payload, provider = asyncio.run(session.invoke_json("decompose", "root prompt"))
    assert payload["ok"] is True
    assert provider == "pi"
