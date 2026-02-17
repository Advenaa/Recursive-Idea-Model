from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from rim.providers.base import (
    BudgetExceededError,
    ProviderConfig,
    ProviderResult,
    StageExecutionError,
)
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


def _normalize_determinism_mode(value: str | None) -> str:
    parsed = str(value or "strict").strip().lower()
    if parsed in {"off", "strict", "balanced"}:
        return parsed
    return "strict"


def _parse_int_env(value: str | None, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _with_determinism_hints(
    *,
    stage: str,
    prompt: str,
    mode: str,
    seed: int,
) -> str:
    if mode == "off":
        return prompt
    if mode == "balanced":
        rules = [
            "Favor consistent structure and stable ordering across retries.",
            "Avoid unnecessary randomness in phrasing and field ordering.",
        ]
    else:
        rules = [
            "Use deterministic wording and stable ordering.",
            "If uncertain, choose the most conservative interpretation.",
            "Avoid introducing random alternatives between retries.",
        ]
    header = (
        f"Determinism policy: {mode}\n"
        f"Determinism seed: {seed}\n"
        f"Stage: {stage}\n"
        "Rules:\n"
        + "\n".join(f"- {rule}" for rule in rules)
    )
    return f"{header}\n\n{prompt}"


def _is_transient_provider_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, OSError):
        return True
    if isinstance(exc, StageExecutionError):
        return bool(exc.retryable)
    text = str(exc).strip().lower()
    if not text:
        return False
    transient_markers = (
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "503",
        "service unavailable",
        "temporar",
        "connection reset",
        "connection aborted",
        "connection refused",
        "network",
        "try again",
    )
    return any(marker in text for marker in transient_markers)


def _is_json_shape_error(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    if not text:
        return False
    markers = (
        "json",
        "no valid json object",
        "missing json",
        "invalid json",
        "parse",
        "schema",
    )
    return any(marker in text for marker in markers)


def _backoff_seconds(base_ms: int, attempt: int) -> float:
    if base_ms <= 0:
        return 0.0
    return (base_ms * (2**attempt)) / 1000.0


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
        determinism_mode: str = "strict",
        determinism_seed: int = 42,
        provider_retry_attempts: int = 2,
        retry_base_ms: int = 250,
    ) -> None:
        self.run_id = run_id
        self.providers = providers
        self.stage_policy = stage_policy
        self.timeout_sec = timeout_sec
        self.budget = budget
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.determinism_mode = _normalize_determinism_mode(determinism_mode)
        self.determinism_seed = int(determinism_seed)
        self.provider_retry_attempts = max(0, int(provider_retry_attempts))
        self.retry_base_ms = max(0, int(retry_base_ms))
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
            "retry": {
                "provider_retry_attempts": self.provider_retry_attempts,
                "retry_base_ms": self.retry_base_ms,
                "json_repair_attempts": self.json_repair_attempts,
            },
        }

    def get_remaining_budget(self) -> dict[str, Any]:
        return {
            "calls": max(0, self.budget.max_calls - self.usage.calls),
            "latency_ms": max(0, self.budget.max_latency_ms - self.usage.latency_ms),
            "tokens": max(0, self.budget.max_tokens - self.usage.tokens),
            "estimated_cost_usd": max(
                0.0,
                self.budget.max_estimated_cost_usd - self.usage.estimated_cost_usd,
            ),
        }

    async def _invoke_with_backoff(
        self,
        *,
        stage: str,
        provider_name: str,
        invoke_once: Any,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.provider_retry_attempts + 1):
            self._check_budget_before_call()
            try:
                return await invoke_once()
            except BudgetExceededError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                retryable = _is_transient_provider_error(exc)
                if retryable and attempt < self.provider_retry_attempts:
                    await asyncio.sleep(_backoff_seconds(self.retry_base_ms, attempt))
                    continue
                raise StageExecutionError(
                    stage=stage,
                    provider=provider_name,
                    message=f"{exc} (attempts={attempt + 1})",
                    retryable=retryable,
                ) from exc
        if last_exc is None:
            raise StageExecutionError(
                stage=stage,
                provider=provider_name,
                message="Provider call failed without exception details.",
                retryable=False,
            )
        raise StageExecutionError(
            stage=stage,
            provider=provider_name,
            message=str(last_exc),
            retryable=False,
        ) from last_exc

    async def invoke_text(self, stage: str, prompt: str) -> tuple[str, str]:
        errors: list[StageExecutionError] = []
        config = ProviderConfig(timeout_sec=self.timeout_sec)
        stage_prompt = _with_determinism_hints(
            stage=stage,
            prompt=prompt,
            mode=self.determinism_mode,
            seed=self.determinism_seed,
        )
        for provider_name in self._providers_for_stage(stage):
            adapter = self.providers[provider_name]
            try:
                result = await self._invoke_with_backoff(
                    stage=stage,
                    provider_name=provider_name,
                    invoke_once=lambda: adapter.invoke(stage_prompt, config),
                )
                self._apply_result_to_usage(result)
                return result.text, provider_name
            except BudgetExceededError:
                raise
            except StageExecutionError as exc:
                errors.append(exc)
        joined = "; ".join(
            f"{err.provider}: {err.message}"
            for err in errors
        )
        raise StageExecutionError(
            stage=stage,
            provider=None,
            message=f"All providers failed for stage '{stage}'. {joined}",
            retryable=any(err.retryable for err in errors),
        )

    async def invoke_json(
        self,
        stage: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        errors: list[StageExecutionError] = []
        config = ProviderConfig(timeout_sec=self.timeout_sec)
        stage_prompt = _with_determinism_hints(
            stage=stage,
            prompt=prompt,
            mode=self.determinism_mode,
            seed=self.determinism_seed,
        )
        for provider_name in self._providers_for_stage(stage):
            adapter = self.providers[provider_name]
            attempt_prompt = stage_prompt
            for attempt in range(self.json_repair_attempts + 1):
                try:
                    payload, result = await self._invoke_with_backoff(
                        stage=stage,
                        provider_name=provider_name,
                        invoke_once=lambda: adapter.invoke_json_with_result(
                            prompt=attempt_prompt,
                            config=config,
                            json_schema=json_schema,
                        ),
                    )
                    self._apply_result_to_usage(result)
                    return payload, provider_name
                except BudgetExceededError:
                    raise
                except StageExecutionError as exc:
                    if attempt < self.json_repair_attempts and _is_json_shape_error(exc):
                        errors.append(
                            StageExecutionError(
                                stage=exc.stage,
                                provider=exc.provider,
                                message=f"{exc.message} (repair retry)",
                                retryable=exc.retryable,
                            )
                        )
                        attempt_prompt = _json_repair_prompt(
                            stage=stage,
                            original_prompt=stage_prompt,
                            error=exc.message,
                        )
                        continue
                    errors.append(exc)
                    break
        joined = "; ".join(
            f"{err.provider}: {err.message}"
            for err in errors
        )
        raise StageExecutionError(
            stage=stage,
            provider=None,
            message=f"All providers failed for stage '{stage}'. {joined}",
            retryable=any(err.retryable for err in errors),
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
        self.default_provider_retry_attempts = max(
            0,
            _parse_int_env(os.getenv("RIM_PROVIDER_MAX_RETRIES"), 2),
        )
        self.default_retry_base_ms = max(
            0,
            _parse_int_env(os.getenv("RIM_PROVIDER_RETRY_BASE_MS"), 250),
        )
        self.default_determinism_mode = _normalize_determinism_mode(
            os.getenv("RIM_DETERMINISM_MODE", "strict")
        )
        self.default_determinism_seed = _parse_int_env(
            os.getenv("RIM_DETERMINISM_SEED"),
            42,
        )

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
            determinism_mode=self.default_determinism_mode,
            determinism_seed=self.default_determinism_seed,
            provider_retry_attempts=self.default_provider_retry_attempts,
            retry_base_ms=self.default_retry_base_ms,
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
