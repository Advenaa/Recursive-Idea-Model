from __future__ import annotations

import asyncio
import json
import os

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, AnalyzeRunResponse
from rim.storage.repo import RunRepository

TERMINAL_STATUSES = {"completed", "failed", "partial"}


class RunJobQueue:
    def __init__(self, orchestrator: RimOrchestrator, repository: RunRepository) -> None:
        self.orchestrator = orchestrator
        self.repository = repository
        self.worker_count = max(1, int(os.getenv("RIM_QUEUE_WORKERS", "1")))
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.in_flight: dict[str, asyncio.Task] = {}
        self.workers: list[asyncio.Task] = []
        self.started = False

    async def start(self) -> None:
        if self.started:
            return
        self.started = True
        for run_id in self.repository.get_pending_runs():
            await self.queue.put(run_id)
        for _ in range(self.worker_count):
            self.workers.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        if not self.started:
            return
        for _ in self.workers:
            await self.queue.put(None)
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        self.started = False

    async def submit(self, request: AnalyzeRequest) -> str:
        run_id = self.orchestrator.create_run(request, status="queued")
        await self.queue.put(run_id)
        return run_id

    async def wait_for(self, run_id: str, poll_sec: float = 0.25) -> AnalyzeRunResponse | None:
        while True:
            task = self.in_flight.get(run_id)
            if task is not None:
                try:
                    await task
                except Exception:  # noqa: BLE001
                    pass
            run = self.orchestrator.get_run(run_id)
            if run is None:
                return None
            if run.status in TERMINAL_STATUSES:
                return run
            await asyncio.sleep(poll_sec)

    async def _worker_loop(self) -> None:
        while True:
            run_id = await self.queue.get()
            if run_id is None:
                self.queue.task_done()
                return

            request_payload = self.repository.get_run_request(run_id)
            if request_payload is None:
                error = {
                    "stage": "queue",
                    "provider": None,
                    "message": "Missing or invalid request payload for queued run.",
                    "retryable": False,
                }
                self.repository.mark_run_status(
                    run_id,
                    status="failed",
                    error_summary=json.dumps(error),
                )
                self.repository.log_stage(
                    run_id=run_id,
                    stage="queue",
                    status="failed",
                    meta={"error": error},
                )
                self.queue.task_done()
                continue

            try:
                request = AnalyzeRequest.model_validate(request_payload)
            except Exception as exc:  # noqa: BLE001
                error = {
                    "stage": "queue",
                    "provider": None,
                    "message": f"Failed to decode request payload: {exc}",
                    "retryable": False,
                }
                self.repository.mark_run_status(
                    run_id,
                    status="failed",
                    error_summary=json.dumps(error),
                )
                self.repository.log_stage(
                    run_id=run_id,
                    stage="queue",
                    status="failed",
                    meta={"error": error},
                )
                self.queue.task_done()
                continue

            task = asyncio.create_task(self.orchestrator.execute_run(run_id, request))
            self.in_flight[run_id] = task
            try:
                await task
            except Exception:  # noqa: BLE001
                pass
            finally:
                self.in_flight.pop(run_id, None)
                self.queue.task_done()
