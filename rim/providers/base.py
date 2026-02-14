from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    timeout_sec: int = 180
    max_output_chars: int = 120_000


@dataclass(frozen=True)
class ProviderResult:
    text: str
    raw_output: str
    latency_ms: int
    estimated_tokens_in: int
    estimated_tokens_out: int
    provider: str
    exit_code: int


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def extract_json_blob(text: str) -> str:
    cleaned = _strip_markdown_fences(text)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        json.loads(candidate)
        return candidate
    raise ValueError("No valid JSON object found in provider output")


class ProviderAdapter(ABC):
    name: str

    @abstractmethod
    async def invoke(self, prompt: str, config: ProviderConfig) -> ProviderResult:
        raise NotImplementedError

    async def invoke_json(
        self,
        prompt: str,
        config: ProviderConfig,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = await self.invoke(prompt, config)
        blob = extract_json_blob(result.text)
        return json.loads(blob)

    @abstractmethod
    async def healthcheck(self) -> bool:
        raise NotImplementedError
