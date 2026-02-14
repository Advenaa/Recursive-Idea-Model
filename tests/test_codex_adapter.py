from __future__ import annotations

from rim.providers.codex_cli import CodexCLIAdapter


def test_codex_adapter_enables_experimental_feature_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("RIM_CODEX_ENABLE_FEATURES", raising=False)
    monkeypatch.delenv("RIM_CODEX_EXPERIMENTAL_FEATURES", raising=False)
    adapter = CodexCLIAdapter(command="codex")
    cmd = adapter._build_base_cmd()
    assert "--enable" in cmd
    idx = cmd.index("--enable")
    assert cmd[idx + 1] == "collab"


def test_codex_adapter_env_feature_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RIM_CODEX_ENABLE_FEATURES", "collab apps runtime_metrics")
    monkeypatch.setenv("RIM_CODEX_DISABLE_FEATURES", "web_search_cached")
    adapter = CodexCLIAdapter(command="codex")
    cmd = adapter._build_base_cmd()
    assert cmd.count("--enable") == 3
    assert "apps" in cmd
    assert "runtime_metrics" in cmd
    assert "--disable" in cmd
    d_idx = cmd.index("--disable")
    assert cmd[d_idx + 1] == "web_search_cached"
