# chutes-bench-runner worker deadlock — forensic report

**Incident window:** 2026-07-21 15:17:03 UTC → 2026-08-05 23:30 UTC (15 days, 8 h)
**Host:** `own_postgres` (94.130.222.43)
**Container:** `chutes-bench-runner-worker-1` (`b623384cc1da`), image `chutes-bench-runner-worker`
**Impact:** data plane fully dead. Every submitted run would have sat in `queued` forever.
Control plane (Render API + UI + Postgres) was unaffected and healthy throughout.
**Detection:** none. Zero alerts fired in 15 days.

---

## 1. Observed state (before any intervention)

| Probe | Result |
|---|---|
| `docker ps` | `Up 2 weeks`, `RestartCount=0`, `StartedAt=2026-07-21T15:17:00.616Z`, `OOMKilled=false`, restart policy `unless-stopped` |
| `docker stats` | **CPU 0.00 %**, mem 60.95 MiB / 8 GiB, **PIDS 1**, NET I/O 13.9 kB / 19 kB total |
| CPU time | **1 second consumed in 15 days** |
| `/proc/559266/status` | `State: S (sleeping)`, single task |
| `/proc/559266/wchan` | `ep_poll` |
| `/proc/559266/stack` | `ep_poll → do_epoll_wait → __x64_sys_epoll_wait` |
| `/proc/559266/syscall` | `232 0x3 0x7d32c4f25350 0x1 **0xffffffffffffffff** …` — `epoll_wait(epfd=3, …, timeout=**-1**)` → **infinite block, no timer pending** |
| `/proc/559266/fd` | only `0→/dev/null`, `1,2→pipes`, `3→eventpoll`, `4,5→socketpair` (the asyncio self-pipe). **Zero network sockets.** |
| `ss` in netns | only the two `u_str` socketpair endpoints. No TCP to 94.130.222.43:5432. |
| `docker logs` tail | last line `{"event": "Worker started", "timestamp": "2026-07-21T15:17:03.603077Z"}` — **nothing after**, for 15 days |
| `worker_heartbeats` | newest row `b623384cc1da`, `last_seen = 2026-07-21 15:17:03` |
| `GET /api/ops/overview` | `"workers": []` |

## 2. py-spy dump (native stack)

```
Process 559266: python -m app.worker.runner
Python v3.11.15
Thread 1 (idle): "MainThread"
    select (selectors.py:468)
    _run_once (asyncio/base_events.py:1898)
    run_forever (asyncio/base_events.py:608)
    run_until_complete (asyncio/base_events.py:641)
    run (asyncio/runners.py:118)
    run (asyncio/runners.py:190)
    <module> (app/worker/runner.py:1769)
```

The event loop is alive but has nothing to do. py-spy cannot see coroutine state, so the
asyncio task graph was extracted by injecting a dump script through gdb
(`PyGILState_Ensure` → `PyRun_SimpleString` → write to fd 2).

## 3. asyncio task graph (the smoking gun)

```
LOOP=<_UnixSelectorEventLoop running=True closed=False> running=True
NUM_TASKS=3

=== TASK <Task pending name='Task-16'
     coro=<AsyncAdapt_asyncpg_connection._terminate_graceful_close() running at
           sqlalchemy/dialects/postgresql/asyncpg.py:912>
     wait_for=<Future pending cb=[Task.task_wakeup()]>
     cb=[shield.<locals>._inner_done_callback() at asyncio/tasks.py:891]> cancelling=0 ===
  sqlalchemy/dialects/postgresql/asyncpg.py:912 in _terminate_graceful_close

=== TASK <Task pending name='Task-1'
     coro=<run_worker() running at /app/app/worker/runner.py:1763>
     cb=[_run_until_complete_cb()]> cancelling=0 ===
  /app/app/worker/runner.py:1763 in run_worker

=== TASK <Task cancelling name='Task-14'
     coro=<BenchmarkWorker.launch_runs() running at /app/app/worker/runner.py:583>
     wait_for=<Future pending cb=[shield.<locals>._outer_done_callback() at asyncio/tasks.py:908,
                                  Task.task_wakeup()]>
     cb=[_release_waiter(...)() at asyncio/tasks.py:431]> cancelling=1 ===
  /app/app/worker/runner.py:583 in launch_runs

TIMER_HANDLES=0  READY=0
```

`TIMER_HANDLES=0 READY=0` is the proof that this is terminal: the loop has **no scheduled
timers and no ready callbacks**, which is exactly why `epoll_wait` was called with
`timeout=-1`. Nothing will ever wake this process again.

## 4. Root cause — a three-layer cancellation deadlock

Versions: Python **3.11.15**, SQLAlchemy **2.0.49**, asyncpg **0.31.0**.

### The trigger
`systemctl show docker -p ActiveEnterTimestamp` → **`Tue 2026-07-21 17:17:01 CEST`** =
`15:17:01 UTC`. The Docker daemon was restarted, which recreated the bridge network and
restarted the container at `15:17:00`. The worker logged `Worker started` at `15:17:03` and
then, on its **very first** `launch_runs()` iteration, tried to open its first pooled DB
connection while the container's networking / NAT rules were still being rebuilt. That
connection attempt never completed.

(The same class of DB unavailability had already hammered this worker on 2026-06-28
20:05–20:07 UTC, where 80 `Worker error` entries logged
`asyncpg.exceptions.CannotConnectNowError: the database system is in recovery mode`
from `requeue_stale_runs` → `wait_for(…, 30)`. Those recovered. This one did not.)

### The deadlock chain

1. `runner.py:550` — `await asyncio.wait_for(self.launch_runs(), timeout=30)`.
   `launch_runs` → `claim_next_run` → `async_session_maker()` → pool checkout → asyncpg
   connect, which hangs because the network is gone.

2. At t+30 s `wait_for` fires. In **CPython 3.11** `wait_for` does *not* simply raise — it runs:
   ```python
   fut.remove_done_callback(cb)
   await _cancel_and_wait(fut, loop=loop)   # asyncio/tasks.py
   raise exceptions.TimeoutError()
   ```
   and `_cancel_and_wait` is:
   ```python
   waiter = loop.create_future()
   fut.add_done_callback(functools.partial(_release_waiter, waiter))
   fut.cancel()
   await waiter        # <-- NO TIMEOUT. Waits forever for the cancel to land.
   ```
   This is Task-1 blocking on Task-14 (`cancelling=1`, callback `_release_waiter` at
   `tasks.py:431` — the `_cancel_and_wait` waiter, visible in the dump).

3. The `CancelledError` is thrown into the SQLAlchemy greenlet. Unwinding the failed
   checkout invalidates the connection, which calls
   **`sqlalchemy/connectors/asyncio.py:402`**:
   ```python
   def terminate(self) -> None:
       if in_greenlet():
           try:
               self.await_(asyncio.shield(self._terminate_graceful_close()))   # line 402
           except self._terminate_handled_exceptions() as e:
               self._terminate_force_close()
   ```
   `asyncio.shield()` **deliberately makes this uncancellable** — that is its entire purpose.
   This spawns Task-16.

4. Task-16 runs `asyncpg.py:912`:
   ```python
   async def _terminate_graceful_close(self) -> None:
       await self._connection.close(timeout=2)
   ```
   Despite the nominal `timeout=2`, the dump shows Task-16 pending on a bare
   `Future` with **no timer armed** (`TIMER_HANDLES=0`). asyncpg's close path had already
   burned its timeout and fallen through to a cancel-request/close waiter that no longer has a
   deadline, on a transport whose peer is gone and whose `connection_lost` will never fire.

5. Net effect:
   - Task-16 never finishes (no timer, no socket, shielded).
   - Task-14 cannot finish because it is awaiting the shielded future; the shield swallows
     the cancellation.
   - Task-1 cannot finish because `_cancel_and_wait` waits on Task-14 with no timeout.
   - The loop has 0 timers and 0 callbacks → `epoll_wait(-1)` → **permanent, silent,
     zero-CPU death.**

### Why nothing detected it

- The process stayed `running` and `S (sleeping)`, so Docker's `unless-stopped` policy never
  triggered — nothing exited.
- No `HEALTHCHECK` was defined on the image.
- `scripts/worker_autoscaler.py::get_worker_counts` counts containers via
  `docker ps --format '{{.Names}}'`. A deadlocked-but-running container counts as healthy;
  the autoscaler logged `current_workers=1` every 33 s for 15 days.
- `scripts/queue_health_monitor.py` only alerts on queue **depth** and queue **age**. The queue
  was empty, so it logged `queued=0 running=0 stale=0` forever. Its `consecutive_no_progress`
  "STUCK WORKERS" branch also requires a non-empty queue.
- The `asyncio.wait_for` guards that were supposed to be the safety net *were themselves the
  deadlock*. Commits `0de7c0b` ("prevent worker main loop freeze from DB pool exhaustion"),
  `dc4c034` ("add db pool headroom for worker ops") and `afbdb49` ("harden worker reliability")
  all added or tuned `wait_for` wrappers — they made the failure *more* likely, not less,
  because every added `wait_for` is another place `_cancel_and_wait` can wedge the loop.

## 5. Verdict

Not a DB pool-size problem and not a Postgres problem. It is a **cancellation-semantics bug in
the worker's use of `asyncio.wait_for` around SQLAlchemy async DB calls**, made unrecoverable by
`asyncio.shield` inside SQLAlchemy's connection-invalidation path. Any transient loss of DB
connectivity during a DB operation can reproduce it.

## 6. Fixes applied (see BENCH_RUNNER_REPAIR.md)

1. `run_guarded()` replaces every `asyncio.wait_for` in the main loop. It uses
   `asyncio.wait({task}, timeout=T)` — which **returns without awaiting cancellation** — then
   fires `task.cancel()` and *abandons* the task. The main loop can never be held hostage
   by an uncancellable child again. Because abandoned tasks still hold DB pool slots (and the
   loop would keep ticking past the watchdog while doing no work), the loop hard-exits once
   `WORKER_MAX_ABANDONED_OPS` (default 5) accumulate.

   Verified with a standalone reproduction run on the container's own Python 3.11.15:
   `asyncio.wait_for` deadlocks on a shielded, never-resolving cleanup; `run_guarded`
   returns cleanly and the event loop stays usable.
2. An OS-level `threading.Thread` watchdog (daemon) that `os._exit(70)`s the process if the
   main loop has not ticked in `WORKER_WATCHDOG_TIMEOUT_SECONDS` (default 600 s). It is a real
   thread, not an asyncio task, so it survives a fully wedged event loop.
3. `connect_args={"timeout": 15, "command_timeout": 120}` on the asyncpg engine, so a
   connect can no longer hang indefinitely in the first place.
4. `HEALTHCHECK` in the worker Dockerfile reading the watchdog's heartbeat file.
5. Heartbeat-staleness alerting (urgent Telegram) added to `scripts/queue_health_monitor.py`,
   plus stale-worker auto-restart in `scripts/worker_autoscaler.py`.
6. Host: new systemd drop-in
   `/etc/systemd/system/chutes-bench-runner-autoscaler.service.d/wedged-worker-watchdog.conf`
   giving the autoscaler `ADMIN_SECRET` (without it the stale-worker check is a no-op) and
   raising `COMPOSE_TIMEOUT_SECONDS` 120 → 900. The 120 s default was too short to build the
   worker image, so the autoscaler's auto-rebuild-on-new-commit kept logging
   *"Worker image build failed; keeping existing workers"* — silently pinning the fleet to
   stale code.
