"""
dispatcher.py — worker process pool management and request routing.

Responsibility boundary:
  - Dispatcher owns worker lifecycle and routing.
  - QueueManager owns the asyncio waiting queue and overflow policy.

Locking model
-------------
A single asyncio.Condition (_cond) protects active_requests on every
WorkerHandle.  Using the Condition's built-in lock for *all* mutations and
reads of active_requests ensures consistency: dispatch(), release(), and
wait_for_free_worker() all hold the same underlying lock, so the notify from
release() is always visible to a waiting wait_for_free_worker().

multiprocessing context
-----------------------
All Queue objects are created via the same "spawn" context used for worker
Process objects.  Mixing contexts (e.g. bare multiprocessing.Queue() on Linux
which defaults to "fork") can produce incompatible pipe handles.  Use
Dispatcher.make_queue() everywhere a result queue is needed.
"""

import asyncio
import contextlib
from dataclasses import dataclass
import multiprocessing

from logging_config import get_logger
from worker import (
    ComputeLatentsRequest,
    LatentsResult,
    StreamSynthesisRequest,
    SynthesisRequest,
    worker_main,
)

logger = get_logger(__name__)

# Shared spawn context — used for both Process and Queue creation so that
# the communication pipes are always compatible.
_MP_CTX = multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# Per-worker state
# ---------------------------------------------------------------------------


@dataclass
class WorkerHandle:
    worker_id: str
    gpu_index: int
    process: multiprocessing.Process
    request_queue: multiprocessing.Queue
    active_requests: int = 0
    total_requests: int = 0
    total_synthesis_ms: float = 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class Dispatcher:
    def __init__(self, model_path: str, workers_per_gpu_list: list[int]) -> None:
        self._model_path = model_path
        self._workers_per_gpu = workers_per_gpu_list
        self._workers: list[WorkerHandle] = []
        # Single Condition for all active_requests mutations.  Its internal
        # lock is held by dispatch(), release(), and wait_for_free_worker().
        # Initialized lazily on first use so it binds to the running event loop.
        self._cond: asyncio.Condition | None = None
        self._status_task: asyncio.Task | None = None

    def _get_cond(self) -> asyncio.Condition:
        """Return the Condition, creating it lazily inside a running loop."""
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn all worker processes. Call before the event loop starts."""
        total = sum(self._workers_per_gpu)
        logger.info(
            "Dispatcher — spawning %d worker(s) across %d GPU slot(s)",
            total,
            len(self._workers_per_gpu),
        )

        # Import torch only for the device-name lookup — keep CUDA init out of
        # the main process as much as possible to avoid re-init errors in workers.
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except ImportError:
            cuda_ok = False

        for gpu_index, count in enumerate(self._workers_per_gpu):
            device_name = "CPU"
            if cuda_ok:
                with contextlib.suppress(Exception):
                    device_name = torch.cuda.get_device_name(gpu_index)

            for slot in range(count):
                worker_id = f"gpu{gpu_index}-w{slot}"
                q = _MP_CTX.Queue()
                p = _MP_CTX.Process(
                    target=worker_main,
                    args=(worker_id, gpu_index, self._model_path, q),
                    name=f"xtts-{worker_id}",
                    daemon=True,
                )
                p.start()
                self._workers.append(
                    WorkerHandle(
                        worker_id=worker_id,
                        gpu_index=gpu_index,
                        process=p,
                        request_queue=q,
                    )
                )
                logger.info(
                    "Spawned worker %s on GPU %d (%s) — PID %d",
                    worker_id,
                    gpu_index,
                    device_name,
                    p.pid,
                )

        logger.info("Dispatcher — all workers spawned")

    async def start_background_tasks(self) -> None:
        """Start periodic status logging. Call after the event loop is running."""
        self._status_task = asyncio.create_task(self._periodic_status())

    async def shutdown(self) -> None:
        """Send shutdown sentinel to every worker and wait for them to exit."""
        if self._status_task:
            self._status_task.cancel()

        loop = asyncio.get_running_loop()
        logger.info("Dispatcher — shutting down %d worker(s)", len(self._workers))

        for w in self._workers:
            await loop.run_in_executor(None, w.request_queue.put, None)

        for w in self._workers:
            await loop.run_in_executor(None, w.process.join, 15)
            if w.process.is_alive():
                logger.warning("Worker %s did not exit — terminating", w.worker_id)
                w.process.terminate()

        logger.info("Dispatcher — shutdown complete")

    # ------------------------------------------------------------------
    # Queue factory (M-10)
    # ------------------------------------------------------------------

    @staticmethod
    def make_queue() -> multiprocessing.Queue:
        """Create a result queue using the same spawn context as worker processes."""
        return _MP_CTX.Queue()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        request: SynthesisRequest | StreamSynthesisRequest | ComputeLatentsRequest,
    ) -> "WorkerHandle":
        """
        Route request to the least-loaded worker and increment its counter.
        Holds the Condition lock during the counter mutation so release()
        and wait_for_free_worker() always see a consistent view.
        """
        if not self._workers:
            raise RuntimeError("No workers available — did you call dispatcher.start()?")

        cond = self._get_cond()
        async with cond:
            worker = self._pick_worker()
            worker.active_requests += 1

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, worker.request_queue.put, request)

        logger.info(
            "Dispatched job=%s → worker=%s (gpu=%d, active=%d)",
            request.job_id,
            worker.worker_id,
            worker.gpu_index,
            worker.active_requests,
        )
        return worker

    async def release(self, worker_id: str, elapsed_ms: float = 0.0) -> None:
        """
        Decrement active_requests for the given worker and notify all waiters.
        Must be called every time a dispatched request completes (or errors).
        """
        if not worker_id:
            # Empty worker_id is a no-op (defensive guard for edge cases).
            return

        cond = self._get_cond()
        async with cond:
            for w in self._workers:
                if w.worker_id == worker_id:
                    w.active_requests = max(0, w.active_requests - 1)
                    w.total_synthesis_ms += elapsed_ms
                    break
            cond.notify_all()

    # ------------------------------------------------------------------
    # Latent computation helper (speaker registration path)
    # ------------------------------------------------------------------

    async def compute_latents(self, job_id: str, wav_path: str) -> LatentsResult:
        """
        Send a ComputeLatentsRequest to any worker and await the result.
        This bypasses the QueueManager waiting queue — latent computation
        is fast and called only from the clone endpoint (not high-frequency).
        """
        loop = asyncio.get_running_loop()
        result_queue = self.make_queue()

        request = ComputeLatentsRequest(
            job_id=job_id,
            wav_path=wav_path,
            result_queue=result_queue,
        )
        worker = await self.dispatch(request)

        try:
            result: LatentsResult = await loop.run_in_executor(None, result_queue.get)
        finally:
            # Always release, even if result_queue.get() raises.
            await self.release(worker.worker_id, elapsed_ms=0.0)

        return result

    # ------------------------------------------------------------------
    # Capacity queries (called by QueueManager)
    # ------------------------------------------------------------------

    def min_active_requests(self) -> int:
        if not self._workers:
            return 0
        return min(w.active_requests for w in self._workers)

    def all_busy(self) -> bool:
        """True when every worker has at least one active request."""
        return bool(self._workers) and all(w.active_requests >= 1 for w in self._workers)

    async def wait_for_free_worker(self) -> None:
        """
        Suspend until at least one worker drops to active_requests == 0.
        Uses the same Condition lock as dispatch() and release() so the
        notify from release() is guaranteed to wake this coroutine.
        """
        cond = self._get_cond()
        async with cond:
            while self.all_busy():
                await cond.wait()

    def worker_stats(self) -> list[dict]:
        """Snapshot of per-worker stats. Used for /v1/system/info and logging."""
        return [
            {
                "worker_id": w.worker_id,
                "gpu_index": w.gpu_index,
                "active_requests": w.active_requests,
                "total_requests": w.total_requests,
                "avg_synthesis_ms": (
                    w.total_synthesis_ms / w.total_requests if w.total_requests else 0.0
                ),
                "alive": w.process.is_alive(),
            }
            for w in self._workers
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_worker(self) -> WorkerHandle:
        """Return worker with the lowest active_requests. Call while holding _cond."""
        return min(self._workers, key=lambda w: w.active_requests)

    async def _periodic_status(self) -> None:
        while True:
            await asyncio.sleep(60)
            self._log_status()

    def _log_status(self) -> None:
        logger.info("--- Periodic worker status ---")
        for w in self._workers:
            avg_ms = w.total_synthesis_ms / w.total_requests if w.total_requests else 0.0
            logger.info(
                "  worker=%s gpu=%d active=%d total=%d avg_ms=%.1f alive=%s",
                w.worker_id,
                w.gpu_index,
                w.active_requests,
                w.total_requests,
                avg_ms,
                w.process.is_alive(),
            )

        # Lazy torch import — avoid pulling CUDA into the main process at module load.
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    used_mb = torch.cuda.memory_allocated(i) / (1024**2)
                    total_mb = torch.cuda.get_device_properties(i).total_memory / (1024**2)
                    logger.info(
                        "  GPU %d VRAM: %.1f / %.1f MB (%.1f%% used)",
                        i,
                        used_mb,
                        total_mb,
                        100 * used_mb / total_mb,
                    )
        except ImportError:
            pass

        logger.info("--- End status ---")
