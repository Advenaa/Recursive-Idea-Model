import os
from typing import Any

from rim.providers.base import ProviderConfig
from rim.providers.claude_cli import ClaudeCLIAdapter
from rim.providers.codex_cli import CodexCLIAdapter

DEFAULT_STAGE_POLICY: dict[str, list[str]] = {
    "decompose": ["codex", "claude"],
    "critic_logic": ["codex", "claude"],
    "critic_evidence": ["claude", "codex"],
    "critic_execution": ["codex", "claude"],
    "critic_adversarial": ["claude", "codex"],
    "synthesize_primary": ["claude", "codex"],
    "synthesize_refine": ["codex", "claude"],
}


class ProviderRouter:
    def __init__(self, stage_policy: dict[str, list[str]] | None = None) -> None:
        self.stage_policy = stage_policy or DEFAULT_STAGE_POLICY
        self.providers = {
            "codex": CodexCLIAdapter(),
            "claude": ClaudeCLIAdapter(),
        }
        self.default_timeout_sec = int(os.getenv("RIM_PROVIDER_TIMEOUT_SEC", "180"))

    async def healthcheck(self) -> dict[str, bool]:
        return {
            name: await adapter.healthcheck()
            for name, adapter in self.providers.items()
        }

    def _providers_for_stage(self, stage: str) -> list[str]:
        return self.stage_policy.get(stage, ["codex", "claude"])

    async def invoke_text(self, stage: str, prompt: str) -> tuple[str, str]:
        errors: list[str] = []
        config = ProviderConfig(timeout_sec=self.default_timeout_sec)
        for provider_name in self._providers_for_stage(stage):
            adapter = self.providers[provider_name]
            try:
                result = await adapter.invoke(prompt, config)
                return result.text, provider_name
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider_name}: {exc}")
        raise RuntimeError(
            f"All providers failed for stage '{stage}'. " + "; ".join(errors)
        )

    async def invoke_json(self, stage: str, prompt: str) -> tuple[dict[str, Any], str]:
        errors: list[str] = []
        config = ProviderConfig(timeout_sec=self.default_timeout_sec)
        for provider_name in self._providers_for_stage(stage):
            adapter = self.providers[provider_name]
            try:
                result = await adapter.invoke_json(prompt, config)
                return result, provider_name
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider_name}: {exc}")
        raise RuntimeError(
            f"All providers failed for stage '{stage}'. " + "; ".join(errors)
        )
