"""OS-thread watchdog for the benchmark worker.

Why a *thread* and not an asyncio task
--------------------------------------
On 2026-07-21 the worker deadlocked for 15 days with **zero** CPU usage and no log
output. The asyncio event loop was alive but had 0 scheduled timers and 0 ready
callbacks, so it sat in ``epoll_wait(timeout=-1)`` forever. The chain was:

1. ``asyncio.wait_for(self.launch_runs(), timeout=30)`` fired.
2. CPython's ``wait_for`` then calls ``_cancel_and_wait(fut)``, which awaits the
   cancelled task's completion **with no timeout**.
3. The cancellation could not land, because unwinding the failed pool checkout hit
   ``sqlalchemy/connectors/asyncio.py::terminate()``, which does
   ``self.await_(asyncio.shield(self._terminate_graceful_close()))`` -- and
   ``asyncio.shield`` is explicitly designed to swallow cancellation.
4. asyncpg's ``connection.close()`` inside that shield never completed because the
   peer was gone.

Because the loop itself was wedged, an in-loop watchdog *task* is not trustworthy:
it only runs if the loop still schedules timers. A daemon OS thread runs regardless
of what the event loop is doing, which is exactly what this failure mode needs.

The watchdog observes a monotonic "tick" that the main loop bumps on every
iteration. If the tick goes stale it hard-exits via ``os._exit`` (bypassing atexit
handlers and any wedged shutdown path) so Docker's ``restart: unless-stopped``
policy brings the worker back.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from typing import Optional

# Exit code used when the watchdog kills a wedged worker. Distinct from ordinary
# crashes so it is greppable in `docker inspect` / logs.
WATCHDOG_EXIT_CODE = 70


class LoopWatchdog:
    """Hard-exits the process when the worker main loop stops ticking."""

    def __init__(
        self,
        timeout_seconds: float,
        check_interval_seconds: float = 15.0,
        heartbeat_file: Optional[str] = None,
        logger=None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.check_interval_seconds = max(1.0, float(check_interval_seconds))
        self.heartbeat_file = heartbeat_file
        self._logger = logger
        self._last_tick = time.monotonic()
        self._last_tick_wall = time.time()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._stage = "startup"

    # ------------------------------------------------------------------ ticks

    def tick(self, stage: str = "loop") -> None:
        """Called from the main loop on every iteration. Must never block."""
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            self._last_tick = now_mono
            self._last_tick_wall = now_wall
            self._stage = stage
        if self.heartbeat_file:
            # Best effort; a full disk must never take the worker down.
            try:
                with open(self.heartbeat_file, "w") as fh:
                    fh.write(f"{now_wall:.0f} {stage}\n")
            except OSError:
                pass

    def age(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_tick

    # ------------------------------------------------------------------ thread

    def start(self) -> None:
        if self._thread is not None:
            return
        self.tick("startup")
        self._thread = threading.Thread(
            target=self._run,
            name="worker-watchdog",
            daemon=True,
        )
        self._thread.start()
        self._log(
            "info",
            "Watchdog started",
            timeout_seconds=self.timeout_seconds,
            check_interval_seconds=self.check_interval_seconds,
        )

    def stop(self) -> None:
        self._stopped.set()

    def _run(self) -> None:
        while not self._stopped.wait(self.check_interval_seconds):
            age = self.age()
            if age <= self.timeout_seconds:
                continue
            with self._lock:
                stage = self._stage
                last_wall = self._last_tick_wall
            self._fire(age, stage, last_wall)
            return

    def _fire(self, age: float, stage: str, last_wall: float) -> None:
        """Dump every thread's stack, then hard-exit."""
        try:
            frames = sys._current_frames()
            dump_lines = []
            for thread_id, frame in frames.items():
                dump_lines.append(f"--- thread {thread_id} ---")
                dump_lines.extend(
                    line.rstrip() for line in traceback.format_stack(frame)
                )
            dump = "\n".join(dump_lines)
        except Exception:  # pragma: no cover - diagnostics must never mask the exit
            dump = "<stack dump unavailable>"

        message = (
            f"WATCHDOG: worker main loop has not ticked in {age:.0f}s "
            f"(limit {self.timeout_seconds:.0f}s, last stage={stage!r}, "
            f"last tick at epoch {last_wall:.0f}). Hard-exiting so the container "
            f"restart policy can recover the worker.\n{dump}\n"
        )
        self._log("error", "Watchdog tripped", age_seconds=round(age, 1), stage=stage)
        try:
            sys.stderr.write(message)
            sys.stderr.flush()
        except Exception:  # pragma: no cover
            pass
        # os._exit: the event loop is wedged, so any graceful shutdown path that
        # needs the loop (or an atexit handler that touches it) would hang too.
        os._exit(WATCHDOG_EXIT_CODE)

    # ------------------------------------------------------------------ logging

    def _log(self, level: str, event: str, **kw) -> None:
        if self._logger is None:
            return
        try:
            getattr(self._logger, level)(event, **kw)
        except Exception:  # pragma: no cover
            pass
