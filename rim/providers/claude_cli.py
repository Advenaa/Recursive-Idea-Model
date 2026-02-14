from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import time
from typing import Any

from rim.providers.base import ProviderAdapter, ProviderConfig, ProviderResult, extract_json_blob


class ClaudeCLIAdapter(ProviderAdapter):
    name = "claude"

    def __init__(self, command: str | None = None) -> None:
        self.command = command or os.getenv("RIM_CLAUDE_CMD", "claude")
        self.default_timeout_sec = int(os.getenv("RIM_PROVIDER_TIMEOUT_SEC", "180"))
        self.default_args = shlex.split(os.getenv("RIM_CLAUDE_ARGS", "-p --output-format json"))
        self.model = os.getenv("RIM_CLAUDE_MODEL")

    def _build_base_cmd(self) -> list[str]:
        cmd = [self.command, *self.default_args]
        if self.model:
            cmd.extend(["--model", self.model])
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

    def _extract_text(self, raw_stdout: str) -> str:
        stripped = raw_stdout.strip()
        if not stripped:
            return ""
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(payload, dict):
            if isinstance(payload.get("result"), str) and payload["result"].strip():
                return payload["result"].strip()
            structured = payload.get("structured_output")
            if structured is not None:
                return json.dumps(structured)
        return stripped

    async def invoke(self, prompt: str, config: ProviderConfig) -> ProviderResult:
        timeout = config.timeout_sec or self.default_timeout_sec
        cmd = self._build_base_cmd()
        cmd.append(prompt)
        stdout, stderr, exit_code, latency_ms = await self._run_cmd(cmd, timeout)
        text = self._extract_text(stdout)
        if not text:
            text = stderr.strip()
        return ProviderResult(
            text=text[: config.max_output_chars],
            raw_output=(stdout + "\n" + stderr).strip(),
            latency_ms=latency_ms,
            estimated_tokens_in=max(1, len(prompt) // 4),
            estimated_tokens_out=max(1, len(text) // 4),
            provider=self.name,
            exit_code=exit_code,
        )

    async def invoke_json_with_result(
        self,
        prompt: str,
        config: ProviderConfig,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ProviderResult]:
        timeout = config.timeout_sec or self.default_timeout_sec
        cmd = self._build_base_cmd()
        if json_schema is not None:
            cmd.extend(["--json-schema", json.dumps(json_schema)])
        cmd.append(prompt)

        stdout, stderr, exit_code, latency_ms = await self._run_cmd(cmd, timeout)
        stripped = stdout.strip()
        if not stripped:
            raise ValueError(f"{self.name} returned empty output: {stderr.strip()}")

        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{self.name} returned invalid JSON envelope")

        structured = payload.get("structured_output")
        if isinstance(structured, dict):
            result = ProviderResult(
                text=json.dumps(structured)[: config.max_output_chars],
                raw_output=(stdout + "\n" + stderr).strip(),
                latency_ms=latency_ms,
                estimated_tokens_in=max(1, len(prompt) // 4),
                estimated_tokens_out=max(1, len(json.dumps(structured)) // 4),
                provider=self.name,
                exit_code=exit_code,
            )
            return structured, result

        result_text = payload.get("result")
        if isinstance(result_text, str) and result_text.strip():
            blob = extract_json_blob(result_text)
            parsed = json.loads(blob)
            result = ProviderResult(
                text=result_text[: config.max_output_chars],
                raw_output=(stdout + "\n" + stderr).strip(),
                latency_ms=latency_ms,
                estimated_tokens_in=max(1, len(prompt) // 4),
                estimated_tokens_out=max(1, len(result_text) // 4),
                provider=self.name,
                exit_code=exit_code,
            )
            return parsed, result
        raise ValueError(f"{self.name} output missing JSON result payload")

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
