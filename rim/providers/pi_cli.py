from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import time
from typing import Any

from rim.providers.base import ProviderAdapter, ProviderConfig, ProviderResult, extract_json_blob


class PiCLIAdapter(ProviderAdapter):
    name = "pi"

    def __init__(self, command: str | None = None) -> None:
        self.command = command or os.getenv("RIM_PI_CMD", "pi")
        self.default_timeout_sec = int(os.getenv("RIM_PROVIDER_TIMEOUT_SEC", "180"))
        self.default_args = shlex.split(
            os.getenv("RIM_PI_ARGS", "--print --no-session --mode text")
        )
        self.provider = os.getenv("RIM_PI_PROVIDER")
        self.model = os.getenv("RIM_PI_MODEL")
        self.thinking = os.getenv("RIM_PI_THINKING")

    def _build_base_cmd(self) -> list[str]:
        cmd = [self.command, *self.default_args]
        if self.provider:
            cmd.extend(["--provider", self.provider])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.thinking:
            cmd.extend(["--thinking", self.thinking])
        return cmd

    async def _run_cmd(
        self,
        args: list[str],
        timeout: int,
    ) -> tuple[str, str, int, int]:
        start = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise TimeoutError(f"{self.name} command timed out after {timeout}s") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            process.returncode or 0,
            latency_ms,
        )

    async def invoke(self, prompt: str, config: ProviderConfig) -> ProviderResult:
        timeout = config.timeout_sec or self.default_timeout_sec
        cmd = self._build_base_cmd()
        cmd.append(prompt)
        stdout, stderr, exit_code, latency_ms = await self._run_cmd(cmd, timeout)
        text = stdout.strip() or stderr.strip()
        return ProviderResult(
            text=text[: config.max_output_chars],
            raw_output=(stdout + "\n" + stderr).strip(),
            latency_ms=latency_ms,
            estimated_tokens_in=max(1, len(prompt) // 4),
            estimated_tokens_out=max(1, len(text) // 4),
            provider=self.name,
            exit_code=exit_code,
        )

    def _json_schema_hint(self, json_schema: dict[str, Any]) -> str:
        schema_text = json.dumps(json_schema, separators=(",", ":"), ensure_ascii=True)
        return (
            "\n\nJSON schema requirement:\n"
            f"{schema_text}\n"
            "Return one strict JSON object only.\n"
            "Do not wrap in markdown fences.\n"
        )

    async def invoke_json_with_result(
        self,
        prompt: str,
        config: ProviderConfig,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ProviderResult]:
        effective_prompt = prompt
        if json_schema is not None:
            effective_prompt = prompt + self._json_schema_hint(json_schema)
        result = await self.invoke(effective_prompt, config)
        blob = extract_json_blob(result.text)
        return json.loads(blob), result

    async def invoke_json(
        self,
        prompt: str,
        config: ProviderConfig,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload, _result = await self.invoke_json_with_result(
            prompt=prompt,
            config=config,
            json_schema=json_schema,
        )
        return payload

    async def healthcheck(self) -> bool:
        binary = shlex.split(self.command)[0]
        return shutil.which(binary) is not None
