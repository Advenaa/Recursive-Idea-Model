from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from rim.providers.base import BudgetExceededError, ProviderConfig, ProviderResult
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


def _json_repair_prompt(
    *,
    stage: str,
    original_prompt: str,
    error: str,
) -> str:
    return (
        f"{original_prompt}\n\n"
        "Previous response could not be parsed as valid JSON.\n"
        f"Stage: {stage}\n"
        f"Error: {error}\n\n"
        "Retry now. Return one strict JSON object only.\n"
        "Do not wrap in markdown fences.\n"
        "Do not include explanation text.\n"
    )


@dataclass
class RunBudget:
    max_calls: int
    max_latency_ms: int
    max_tokens: int
    max_estimated_cost_usd: float


@dataclass
class RunUsage:
    calls: int = 0
    latency_ms: int = 0
    tokens: int = 0
    estimated_cost_usd: float = 0.0


def _estimate_call_cost_usd(provider: str, tokens_in: int, tokens_out: int) -> float:
    total_tokens = max(0, tokens_in) + max(0, tokens_out)
    if provider == "claude":
        # Conservative blended estimate for CLI usage.
        rate_per_1k = float(os.getenv("RIM_CLAUDE_COST_PER_1K_TOKENS", "0.015"))
    else:
        rate_per_1k = float(os.getenv("RIM_CODEX_COST_PER_1K_TOKENS", "0.01"))
    return (total_tokens / 1000.0) * rate_per_1k


class ProviderRunSession:
    def __init__(
        self,
        *,
        run_id: str,
        providers: dict[str, Any],
        stage_policy: dict[str, list[str]],
        timeout_sec: int,
        budget: RunBudget,
        json_repair_attempts: int = 1,
    ) -> None:
        self.run_id = run_id
        self.providers = providers
        self.stage_policy = stage_policy
        self.timeout_sec = timeout_sec
        self.budget = budget
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.usage = RunUsage()
        self.provider_results: list[ProviderResult] = []

    def _providers_for_stage(self, stage: str) -> list[str]:
        return self.stage_policy.get(stage, ["codex", "claude"])

    def _check_budget_before_call(self) -> None:
        if self.usage.calls >= self.budget.max_calls:
            raise BudgetExceededError(
                f"Run {self.run_id} exceeded provider call budget ({self.budget.max_calls})"
            )
        if self.usage.latency_ms >= self.budget.max_latency_ms:
            raise BudgetExceededError(
                f"Run {self.run_id} exceeded provider latency budget ({self.budget.max_latency_ms} ms)"
            )
        if self.usage.tokens >= self.budget.max_tokens:
            raise BudgetExceededError(
                f"Run {self.run_id} exceeded token budget ({self.budget.max_tokens})"
            )
        if self.usage.estimated_cost_usd >= self.budget.max_estimated_cost_usd:
            raise BudgetExceededError(
                f"Run {self.run_id} exceeded estimated cost budget (${self.budget.max_estimated_cost_usd:.2f})"
            )

    def _check_budget_after_usage(self) -> None:
        if self.usage.calls > self.budget.max_calls:
            raise BudgetExceededError(
                f"Run {self.run_id} exceeded provider call budget ({self.budget.max_calls})"
            )
        if self.usage.latency_ms > self.budget.max_latency_ms:
            raise BudgetExceededError(
                f"Run {self.run_id} exceeded provider latency budget ({self.budget.max_latency_ms} ms)"
            )
        if self.usage.tokens > self.budget.max_tokens:
            raise BudgetExceededError(
                f"Run {self.run_id} exceeded token budget ({self.budget.max_tokens})"
            )
        if self.usage.estimated_cost_usd > self.budget.max_estimated_cost_usd:
            raise BudgetExceededError(
                f"Run {self.run_id} exceeded estimated cost budget (${self.budget.max_estimated_cost_usd:.2f})"
            )

    def _apply_result_to_usage(self, result: ProviderResult) -> None:
        self.usage.calls += 1
        self.usage.latency_ms += int(result.latency_ms or 0)
        self.usage.tokens += int(result.estimated_tokens_in or 0) + int(
            result.estimated_tokens_out or 0
        )
        self.usage.estimated_cost_usd += _estimate_call_cost_usd(
            result.provider,
            result.estimated_tokens_in,
            result.estimated_tokens_out,
        )
        self.provider_results.append(result)
        # Hard fail if this call pushed usage past budget.
        self._check_budget_after_usage()

    def get_usage_meta(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "usage": {
                "calls": self.usage.calls,
                "latency_ms": self.usage.latency_ms,
                "tokens": self.usage.tokens,
                "estimated_cost_usd": round(self.usage.estimated_cost_usd, 6),
            },
            "budget": {
                "max_calls": self.budget.max_calls,
                "max_latency_ms": self.budget.max_latency_ms,
                "max_tokens": self.budget.max_tokens,
                "max_estimated_cost_usd": self.budget.max_estimated_cost_usd,
            },
        }

    async def invoke_text(self, stage: str, prompt: str) -> tuple[str, str]:
        errors: list[str] = []
        config = ProviderConfig(timeout_sec=self.timeout_sec)
        for provider_name in self._providers_for_stage(stage):
            self._check_budget_before_call()
            adapter = self.providers[provider_name]
            try:
                result = await adapter.invoke(prompt, config)
                self._apply_result_to_usage(result)
                return result.text, provider_name
            except BudgetExceededError:
                raise
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider_name}: {exc}")
        raise RuntimeError(
            f"All providers failed for stage '{stage}'. " + "; ".join(errors)
        )

    async def invoke_json(
        self,
        stage: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        errors: list[str] = []
        config = ProviderConfig(timeout_sec=self.timeout_sec)
        for provider_name in self._providers_for_stage(stage):
            adapter = self.providers[provider_name]
            attempt_prompt = prompt
            for attempt in range(self.json_repair_attempts + 1):
                self._check_budget_before_call()
                try:
                    payload, result = await adapter.invoke_json_with_result(
                        prompt=attempt_prompt,
                        config=config,
                        json_schema=json_schema,
                    )
                    self._apply_result_to_usage(result)
                    return payload, provider_name
                except BudgetExceededError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    suffix = (
                        " (repair retry)"
                        if attempt < self.json_repair_attempts
                        else ""
                    )
                    errors.append(f"{provider_name}: {exc}{suffix}")
                    if attempt < self.json_repair_attempts:
                        attempt_prompt = _json_repair_prompt(
                            stage=stage,
                            original_prompt=prompt,
                            error=str(exc),
                        )
                        continue
                    break
        raise RuntimeError(
            f"All providers failed for stage '{stage}'. " + "; ".join(errors)
        )


class ProviderRouter:
    def __init__(self, stage_policy: dict[str, list[str]] | None = None) -> None:
        self.stage_policy = stage_policy or DEFAULT_STAGE_POLICY
        self.providers = {
            "codex": CodexCLIAdapter(),
            "claude": ClaudeCLIAdapter(),
        }
        self.default_timeout_sec = int(os.getenv("RIM_PROVIDER_TIMEOUT_SEC", "180"))
        self.default_budget = RunBudget(
            max_calls=int(os.getenv("RIM_RUN_MAX_PROVIDER_CALLS", "120")),
            max_latency_ms=int(os.getenv("RIM_RUN_MAX_PROVIDER_LATENCY_MS", "900000")),
            max_tokens=int(os.getenv("RIM_RUN_MAX_ESTIMATED_TOKENS", "500000")),
            max_estimated_cost_usd=float(os.getenv("RIM_RUN_MAX_ESTIMATED_COST_USD", "10.0")),
        )
        self.default_json_repair_attempts = int(os.getenv("RIM_JSON_REPAIR_RETRIES", "1"))

    async def healthcheck(self) -> dict[str, bool]:
        return {
            name: await adapter.healthcheck()
            for name, adapter in self.providers.items()
        }

    def create_session(self, run_id: str) -> ProviderRunSession:
        return ProviderRunSession(
            run_id=run_id,
            providers=self.providers,
            stage_policy=self.stage_policy,
            timeout_sec=self.default_timeout_sec,
            budget=self.default_budget,
            json_repair_attempts=self.default_json_repair_attempts,
        )

    async def invoke_text(self, stage: str, prompt: str) -> tuple[str, str]:
        # Compatibility path for existing call sites.
        session = self.create_session("adhoc")
        return await session.invoke_text(stage, prompt)

    async def invoke_json(
        self,
        stage: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        # Compatibility path for existing call sites.
        session = self.create_session("adhoc")
        return await session.invoke_json(stage, prompt, json_schema=json_schema)
