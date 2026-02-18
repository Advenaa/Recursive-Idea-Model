from __future__ import annotations

import asyncio

from rim.providers.base import ProviderConfig
from rim.providers.pi_cli import PiCLIAdapter


def test_pi_adapter_builds_cmd_with_optional_provider_model_and_thinking(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RIM_PI_PROVIDER", "openai")
    monkeypatch.setenv("RIM_PI_MODEL", "gpt-5")
    monkeypatch.setenv("RIM_PI_THINKING", "high")
    adapter = PiCLIAdapter(command="pi")
    cmd = adapter._build_base_cmd()
    assert cmd[:5] == ["pi", "--print", "--no-session", "--mode", "text"]
    assert "--provider" in cmd
    assert "--model" in cmd
    assert "--thinking" in cmd


def test_pi_adapter_invoke_json_with_result_parses_json(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    adapter = PiCLIAdapter(command="pi")

    async def _fake_run_cmd(args, timeout):  # noqa: ANN001, ANN201
        assert args[-1].startswith("Return JSON")
        return ('{"ok": true}', "", 0, 9)

    monkeypatch.setattr(adapter, "_run_cmd", _fake_run_cmd)
    payload, result = asyncio.run(
        adapter.invoke_json_with_result(
            prompt="Return JSON",
            config=ProviderConfig(timeout_sec=10),
            json_schema={"type": "object"},
        )
    )
    assert payload["ok"] is True
    assert result.provider == "pi"
