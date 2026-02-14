from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from rim.providers.base import ProviderAdapter, ProviderConfig, ProviderResult, extract_json_blob


def _parse_feature_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    chunks = raw.replace(",", " ").split()
    # Preserve ordering but deduplicate.
    seen: set[str] = set()
    features: list[str] = []
    for chunk in chunks:
        feature = chunk.strip()
        if not feature or feature in seen:
            continue
        seen.add(feature)
        features.append(feature)
    return features


class CodexCLIAdapter(ProviderAdapter):
    name = "codex"

    def __init__(self, command: str | None = None) -> None:
        self.command = command or os.getenv("RIM_CODEX_CMD", "codex")
        self.default_timeout_sec = int(os.getenv("RIM_PROVIDER_TIMEOUT_SEC", "180"))
        self.default_args = shlex.split(
            os.getenv(
                "RIM_CODEX_ARGS",
                "exec --skip-git-repo-check --sandbox read-only",
            )
        )
        self.model = os.getenv("RIM_CODEX_MODEL")
        default_features = os.getenv("RIM_CODEX_EXPERIMENTAL_FEATURES", "collab")
        self.enable_features = _parse_feature_list(
            os.getenv("RIM_CODEX_ENABLE_FEATURES", default_features)
        )
        self.disable_features = _parse_feature_list(
            os.getenv("RIM_CODEX_DISABLE_FEATURES")
        )

    def _build_base_cmd(self) -> list[str]:
        cmd = [self.command, *self.default_args]
        if self.model:
            cmd.extend(["--model", self.model])
        for feature in self.enable_features:
            cmd.extend(["--enable", feature])
        for feature in self.disable_features:
            cmd.extend(["--disable", feature])
        return cmd

    async def _run_cmd(
        self,
        args: list[str],
        prompt: str,
        timeout: int,
    ) -> tuple[str, str, int, int]:
        start = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=timeout,
            )
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
        with tempfile.NamedTemporaryFile(delete=False) as out_file:
            out_path = Path(out_file.name)

        cmd = self._build_base_cmd()
        cmd.extend(["-o", str(out_path), "-"])
        try:
            stdout, stderr, exit_code, latency_ms = await self._run_cmd(cmd, prompt, timeout)
            text = out_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
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
        finally:
            out_path.unlink(missing_ok=True)

    async def invoke_json_with_result(
        self,
        prompt: str,
        config: ProviderConfig,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ProviderResult]:
        timeout = config.timeout_sec or self.default_timeout_sec
        with tempfile.NamedTemporaryFile(delete=False) as out_file:
            out_path = Path(out_file.name)
        schema_path: Path | None = None
        if json_schema is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as schema_file:
                schema_path = Path(schema_file.name)
                schema_path.write_text(json.dumps(json_schema), encoding="utf-8")

        cmd = self._build_base_cmd()
        if schema_path is not None:
            cmd.extend(["--output-schema", str(schema_path)])
        cmd.extend(["-o", str(out_path), "-"])

        try:
            stdout, stderr, exit_code, latency_ms = await self._run_cmd(cmd, prompt, timeout)
            text = out_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                text = stdout.strip() or stderr.strip()
            blob = extract_json_blob(text)
            payload = json.loads(blob)
            result = ProviderResult(
                text=text[: config.max_output_chars],
                raw_output=(stdout + "\n" + stderr).strip(),
                latency_ms=latency_ms,
                estimated_tokens_in=max(1, len(prompt) // 4),
                estimated_tokens_out=max(1, len(text) // 4),
                provider=self.name,
                exit_code=exit_code,
            )
            return payload, result
        finally:
            out_path.unlink(missing_ok=True)
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)

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
