import asyncio
import os
import shlex
import shutil
import time

from rim.providers.base import ProviderAdapter, ProviderConfig, ProviderResult


class ClaudeCLIAdapter(ProviderAdapter):
    name = "claude"

    def __init__(self, command: str | None = None) -> None:
        self.command = command or os.getenv("RIM_CLAUDE_CMD", "claude")
        self.default_timeout_sec = int(os.getenv("RIM_PROVIDER_TIMEOUT_SEC", "180"))

    async def invoke(self, prompt: str, config: ProviderConfig) -> ProviderResult:
        timeout = config.timeout_sec or self.default_timeout_sec
        start = time.perf_counter()
        process = await asyncio.create_subprocess_shell(
            self.command,
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
        text = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if not text and err:
            text = err
        return ProviderResult(
            text=text[: config.max_output_chars],
            raw_output=text,
            latency_ms=latency_ms,
            estimated_tokens_in=max(1, len(prompt) // 4),
            estimated_tokens_out=max(1, len(text) // 4),
            provider=self.name,
            exit_code=process.returncode or 0,
        )

    async def healthcheck(self) -> bool:
        binary = shlex.split(self.command)[0]
        return shutil.which(binary) is not None
