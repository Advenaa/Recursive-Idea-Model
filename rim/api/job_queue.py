from __future__ import annotations

import asyncio
import json
import os
import sqlite3

from rim.core.orchestrator import RimOrchestrator
from rim.core.schemas import AnalyzeRequest, AnalyzeRunResponse
from rim.storage.repo import RETRYABLE_RUN_STATUSES, TERMINAL_RUN_STATUSES, RunRepository

TERMINAL_STATUSES = set(TERMINAL_RUN_STATUSES)
CANCELABLE_STATUSES = {"queued", "running"}
QUEUE_TERMINAL_ERROR = {
    "stage": "queue",
    "provider": None,
    "message": "Run canceled by user request.",
    "retryable": True,
}


class RunJobQueue:
    def __init__(self, orchestrator: RimOrchestrator, repository: RunRepository) -> None:
        self.orchestrator = orchestrator
        self.repository = repository
        self.worker_count = max(1, int(os.getenv("RIM_QUEUE_WORKERS", "1")))
        try:
            self.reconcile_interval_sec = max(
                0.25,
                float(os.getenv("RIM_QUEUE_RECONCILE_INTERVAL_SEC", "1.0")),
            )
        except ValueError:
            self.reconcile_interval_sec = 1.0
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.pending_ids: set[str] = set()
        self.in_flight: dict[str, asyncio.Task] = {}
        self.workers: list[asyncio.Task] = []
        self.started = False

    async def _enqueue_run(self, run_id: str) -> bool:
        if run_id in self.pending_ids or run_id in self.in_flight:
            return False
        await self.queue.put(run_id)
        self.pending_ids.add(run_id)
        return True

    async def _enqueue_pending_runs(self) -> None:
        for run_id in self.repository.get_pending_runs():
            await self._enqueue_run(run_id)

    async def start(self) -> None:
        if self.started:
            return
        self.started = True
        await self._enqueue_pending_runs()
        for _ in range(self.worker_count):
            self.workers.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        if not self.started:
            return
        for _ in self.workers:
            await self.queue.put(None)
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        self.pending_ids.clear()
        self.started = False

    async def submit(
        self,
        request: AnalyzeRequest,
        run_id: str | None = None,
    ) -> tuple[str, bool]:
        normalized_request = request.model_dump()
        if run_id:
            existing = self.orchestrator.get_run(run_id)
            if existing is not None:
                existing_request = self.orchestrator.get_run_request(run_id)
                if existing_request is not None:
                    if existing_request.model_dump() != normalized_request:
                        raise ValueError("run_id already exists with a different request payload.")
                if existing.status == "queued":
                    await self._enqueue_run(run_id)
                return run_id, False

        try:
            resolved_run_id = self.orchestrator.create_run(
                request,
                status="queued",
                run_id=run_id,
            )
            await self._enqueue_run(resolved_run_id)
            return resolved_run_id, True
        except sqlite3.IntegrityError:
            if not run_id:
                raise
            existing = self.orchestrator.get_run(run_id)
            if existing is None:
                raise
            existing_request = self.orchestrator.get_run_request(run_id)
            if existing_request is not None:
                if existing_request.model_dump() != normalized_request:
                    raise ValueError("run_id already exists with a different request payload.")
            if existing.status == "queued":
                await self._enqueue_run(run_id)
            return run_id, False

    async def wait_for(self, run_id: str, poll_sec: float = 0.25) -> AnalyzeRunResponse | None:
        while True:
            task = self.in_flight.get(run_id)
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    pass
            run = self.orchestrator.get_run(run_id)
            if run is None:
                return None
            if run.status in TERMINAL_STATUSES:
                return run
            await asyncio.sleep(poll_sec)

    async def cancel(self, run_id: str) -> AnalyzeRunResponse | None:
        run = self.orchestrator.get_run(run_id)
        if run is None:
            return None
        if run.status in TERMINAL_STATUSES:
            return run
        if run.status not in CANCELABLE_STATUSES:
            return run

        self.pending_ids.discard(run_id)
        task = self.in_flight.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass

        latest = self.orchestrator.get_run(run_id)
        if latest is None:
            return None
        if latest.status in TERMINAL_STATUSES and latest.status != "running":
            return latest

        self.repository.mark_run_status(
            run_id=run_id,
            status="canceled",
            error_summary=json.dumps(QUEUE_TERMINAL_ERROR),
        )
        self.repository.log_stage(
            run_id=run_id,
            stage="queue",
            status="canceled",
            meta={"error": QUEUE_TERMINAL_ERROR},
        )
        return self.orchestrator.get_run(run_id)

    async def retry(self, run_id: str) -> AnalyzeRunResponse | None:
        run = self.orchestrator.get_run(run_id)
        if run is None:
            return None
        if run.status not in RETRYABLE_RUN_STATUSES:
            raise ValueError("Run is not retryable unless status is failed, partial, or canceled.")
        if not self.repository.reset_run_for_retry(run_id):
            return None
        await self._enqueue_run(run_id)
        return self.orchestrator.get_run(run_id)

    async def _worker_loop(self) -> None:
        while True:
            try:
                run_id = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.reconcile_interval_sec,
                )
            except asyncio.TimeoutError:
                await self._enqueue_pending_runs()
                continue
            if run_id is None:
                self.queue.task_done()
                return
            self.pending_ids.discard(run_id)

            run = self.orchestrator.get_run(run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                self.queue.task_done()
                continue

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

            run = self.orchestrator.get_run(run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                self.queue.task_done()
                continue

            task = asyncio.create_task(self.orchestrator.execute_run(run_id, request))
            self.in_flight[run_id] = task
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass
            finally:
                self.in_flight.pop(run_id, None)
                self.queue.task_done()
