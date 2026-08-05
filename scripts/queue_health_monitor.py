#!/usr/bin/env python3
"""
Bench Runner Queue Health Monitor with Telegram Alerts.

Monitors the bench-runner queue and sends URGENT Telegram notifications when:
- More than 50 runs are queued
- Any run has been queued for longer than 4 hours
- Workers appear stuck

Sends to the URGENT Telegram bot (distinct notification sound).
Designed to run as a systemd service that auto-restarts on boot.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Configuration (env vars with sensible defaults)
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv(
    "BENCH_BACKEND_URL",
    "https://chutes-bench-runner-api-v2.onrender.com",
).rstrip("/")

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "bench-admin-20260105-8a5f3d2e8b7c4a1b")

# URGENT Telegram bot — distinct notification sound from general cursor_noti_bot
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "8335161337:AAHH3LWuvUAPPKnEg_iRN1oWOT4AQ0ITrcg")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "7367548582")

QUEUE_DEPTH_THRESHOLD = int(os.getenv("QUEUE_DEPTH_THRESHOLD", "50"))
QUEUE_AGE_THRESHOLD_HOURS = float(os.getenv("QUEUE_AGE_THRESHOLD_HOURS", "12"))
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))  # 5 min
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "1800"))  # 30 min
API_TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "15"))

# Auto-cancel runs queued longer than this (hours) — prevents permanently stuck runs
AUTO_CANCEL_AGE_HOURS = float(os.getenv("AUTO_CANCEL_AGE_HOURS", "48"))

# Alert if the newest worker heartbeat is older than this. The worker writes one
# every WORKER_HEARTBEAT_SECONDS (60s in prod), so 10 min is ~10 missed beats.
WORKER_HEARTBEAT_STALE_SECONDS = float(os.getenv("WORKER_HEARTBEAT_STALE_SECONDS", "600"))

LOG_PATH = os.getenv("LOG_PATH", "/var/log/bench-queue-monitor.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("queue_health_monitor")


def configure_logging() -> None:
    logger.setLevel(LOG_LEVEL.upper())
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if LOG_PATH:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        fh = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=2)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _api_get(path: str) -> dict | list | None:
    url = f"{BACKEND_URL}{path}"
    try:
        req = Request(url, headers={"User-Agent": "bench-queue-monitor/1.0"})
        with urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        logger.warning("API request failed %s: %s", url, exc)
        return None


def _api_get_admin(path: str) -> dict | list | None:
    """GET an endpoint that requires the admin secret."""
    url = f"{BACKEND_URL}{path}"
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "bench-queue-monitor/1.0",
                "X-Admin-Secret": ADMIN_SECRET,
            },
        )
        with urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        logger.warning("Admin API request failed %s: %s", url, exc)
        return None


def _api_post(path: str, body: dict | None = None) -> dict | None:
    url = f"{BACKEND_URL}{path}"
    headers = {
        "User-Agent": "bench-queue-monitor/1.0",
        "X-Admin-Secret": ADMIN_SECRET,
        "Content-Type": "application/json",
    }
    try:
        data = json.dumps(body or {}).encode("utf-8")
        req = Request(url, data=data, headers=headers, method="POST")
        with urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        logger.warning("API POST failed %s: %s", url, exc)
        return None


def _parse_iso(s: str) -> datetime | None:
    """Parse ISO timestamp string to aware datetime."""
    if not s or not isinstance(s, str):
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def get_queue_stats() -> tuple[int, int, float | None, list, list]:
    """Return (queued_count, running_count, oldest_queued_age_hours, stale_run_ids, queued_runs)."""
    queued_data = _api_get("/api/runs?status=queued&limit=200")
    running_data = _api_get("/api/runs?status=running&limit=200")

    if queued_data is None or running_data is None:
        return -1, -1, None, [], []

    if isinstance(queued_data, dict):
        queued_count = queued_data.get("total", len(queued_data.get("runs", [])))
        queued_runs = queued_data.get("runs", [])
    else:
        queued_count = 0
        queued_runs = []

    if isinstance(running_data, dict):
        running_count = running_data.get("total", len(running_data.get("runs", [])))
    else:
        running_count = 0

    # Find oldest queued run age and collect stale runs
    oldest_age_hours = None
    stale_run_ids = []
    now = datetime.now(timezone.utc)

    for run in queued_runs:
        dt = _parse_iso(run.get("created_at") or run.get("createdAt"))
        if dt is None:
            continue
        age_hours = (now - dt).total_seconds() / 3600
        if oldest_age_hours is None or age_hours > oldest_age_hours:
            oldest_age_hours = age_hours
        if age_hours > AUTO_CANCEL_AGE_HOURS:
            stale_run_ids.append((run.get("id"), age_hours))

    return queued_count, running_count, oldest_age_hours, stale_run_ids, queued_runs


def get_worker_health() -> tuple[int, float | None, list]:
    """Return (live_worker_count, newest_heartbeat_age_seconds, workers).

    ``/api/ops/overview`` already filters ``workers[]`` by heartbeat recency
    (``get_active_workers`` uses ``last_seen >= cutoff``), so an EMPTY list means
    no worker has checked in recently -- i.e. the data plane is dead even though
    the container may still show as `Up` in ``docker ps``.

    Returns (-1, None, []) if the API could not be reached (don't alert on that;
    the API being down is a different, separately visible failure).
    """
    data = _api_get_admin("/api/ops/overview")
    if not isinstance(data, dict):
        return -1, None, []

    workers = data.get("workers") or []
    if not workers:
        return 0, None, []

    now = datetime.now(timezone.utc)
    newest_age = None
    for w in workers:
        dt = _parse_iso(w.get("last_seen") or w.get("lastSeen"))
        if dt is None:
            continue
        age = (now - dt).total_seconds()
        if newest_age is None or age < newest_age:
            newest_age = age

    return len(workers), newest_age, workers


def auto_cancel_stale_runs(stale_runs: list) -> int:
    """Cancel runs that have been queued too long. Returns count cancelled."""
    cancelled = 0
    for run_id, age_hours in stale_runs:
        result = _api_post(f"/api/runs/{run_id}/cancel")
        if result is not None:
            logger.info("Auto-cancelled stale run %s (queued %.1fh)", run_id, age_hours)
            cancelled += 1
        else:
            logger.warning("Failed to cancel stale run %s", run_id)
    return cancelled


# ---------------------------------------------------------------------------
# Telegram (URGENT bot)
# ---------------------------------------------------------------------------
def send_telegram(message: str) -> bool:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        logger.warning("Telegram credentials not configured, skipping alert")
        return False

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = urlencode({
        "chat_id": TG_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    try:
        req = Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                logger.info("Telegram alert sent successfully (urgent bot)")
                return True
            logger.warning("Telegram API returned ok=false: %s", result)
            return False
    except Exception as exc:
        logger.error("Failed to send Telegram alert: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    configure_logging()
    logger.info(
        "Queue health monitor started (URGENT bot). backend=%s queue_threshold=%d "
        "age_threshold=%.1fh auto_cancel_after=%.0fh heartbeat_stale=%.0fs interval=%ds",
        BACKEND_URL, QUEUE_DEPTH_THRESHOLD, QUEUE_AGE_THRESHOLD_HOURS,
        AUTO_CANCEL_AGE_HOURS, WORKER_HEARTBEAT_STALE_SECONDS, CHECK_INTERVAL_SECONDS,
    )

    last_depth_alert_at = 0.0
    last_age_alert_at = 0.0
    last_stuck_workers_alert_at = 0.0
    last_worker_alert_at = 0.0
    consecutive_no_progress = 0
    last_running_count = -1

    while True:
        try:
            queued, running, oldest_age, stale_runs, queued_runs_cache = get_queue_stats()

            if queued < 0:
                logger.warning("Failed to fetch queue stats, will retry")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # Log stale runs but don't auto-cancel (DB can be under heavy
            # load from workers, causing cancel requests to timeout)
            if stale_runs:
                logger.info("Stale runs (>%.0fh): %d", AUTO_CANCEL_AGE_HOURS, len(stale_runs))

            # For age alerting, exclude stale runs (>48h are permanently stuck,
            # not a queue processing issue). Focus on genuinely waiting runs.
            stale_ids = {r[0] for r in stale_runs}
            fresh_oldest_age = None
            now_dt = datetime.now(timezone.utc)
            for run in queued_runs_cache:
                dt = _parse_iso(run.get("created_at") or run.get("createdAt"))
                if dt is None or run.get("id") in stale_ids:
                    continue
                age_h = (now_dt - dt).total_seconds() / 3600
                if fresh_oldest_age is None or age_h > fresh_oldest_age:
                    fresh_oldest_age = age_h
            # Use fresh age for alerting, full oldest for logging
            alert_age = fresh_oldest_age if fresh_oldest_age is not None else oldest_age

            live_workers, hb_age, _worker_rows = get_worker_health()

            logger.info(
                "Queue check: queued=%d running=%d oldest_age=%.1fh alert_age=%.1fh "
                "stale=%d live_workers=%s heartbeat_age=%s",
                queued, running, oldest_age or 0, alert_age or 0, len(stale_runs),
                live_workers if live_workers >= 0 else "?",
                f"{hb_age:.0f}s" if hb_age is not None else "n/a",
            )

            now = time.time()
            alerts = []

            # Check 0: worker liveness. This is the check that was missing when the
            # worker silently deadlocked for 15 days (2026-07-21 -> 2026-08-05) with
            # an empty queue -- every other check here requires a NON-EMPTY queue and
            # therefore stayed quiet. See docs/bench_runner_incident_2026_08_06.md.
            worker_alert = None
            if live_workers == 0:
                worker_alert = (
                    f"*NO LIVE WORKERS*\n"
                    f"`/api/ops/overview` reports *zero* workers with a recent heartbeat.\n"
                    f"The data plane is down — every submitted run will sit in `queued` "
                    f"forever.\n"
                    f"Queued: {queued} | Running: {running}\n\n"
                    f"Fix: `docker restart chutes-bench-runner-worker-1` on own_postgres, "
                    f"then check `docker logs`."
                )
            elif hb_age is not None and hb_age > WORKER_HEARTBEAT_STALE_SECONDS:
                worker_alert = (
                    f"*STALE WORKER HEARTBEAT*\n"
                    f"Newest worker heartbeat is *{hb_age / 60:.0f} min* old "
                    f"(threshold: {WORKER_HEARTBEAT_STALE_SECONDS / 60:.0f} min).\n"
                    f"Live workers: {live_workers} | Queued: {queued} | Running: {running}\n\n"
                    f"A worker container can be `Up` in `docker ps` and still be wedged."
                )

            if worker_alert and (now - last_worker_alert_at) > ALERT_COOLDOWN_SECONDS:
                alerts.append(worker_alert)
                last_worker_alert_at = now

            # Check 1: Queue depth > threshold
            if queued > QUEUE_DEPTH_THRESHOLD and (now - last_depth_alert_at) > ALERT_COOLDOWN_SECONDS:
                alerts.append(
                    f"*QUEUE DEPTH ALERT*\n"
                    f"Queued runs: *{queued}* (threshold: {QUEUE_DEPTH_THRESHOLD})\n"
                    f"Running runs: {running}"
                )
                last_depth_alert_at = now

            # Check 2: Oldest non-stale queued run > age threshold
            if (
                alert_age is not None
                and alert_age > QUEUE_AGE_THRESHOLD_HOURS
                and (now - last_age_alert_at) > ALERT_COOLDOWN_SECONDS
            ):
                stale_note = f" ({len(stale_runs)} stale runs excluded)" if stale_runs else ""
                alerts.append(
                    f"*QUEUE AGE ALERT*\n"
                    f"Oldest queued run: *{alert_age:.1f}h* (threshold: {QUEUE_AGE_THRESHOLD_HOURS}h){stale_note}\n"
                    f"Queued: {queued} | Running: {running}"
                )
                last_age_alert_at = now

            # Check 3: Detect stuck workers (running count unchanged with large queue)
            if queued > 20 and running == last_running_count and running <= 4:
                consecutive_no_progress += 1
            else:
                consecutive_no_progress = 0
            last_running_count = running

            if consecutive_no_progress >= 6 and (now - last_stuck_workers_alert_at) > ALERT_COOLDOWN_SECONDS * 2:
                alerts.append(
                    f"*STUCK WORKERS ALERT*\n"
                    f"Only {running} runs active for 30+ min with {queued} queued.\n"
                    f"Workers may be hung or failing silently."
                )
                last_stuck_workers_alert_at = now
                consecutive_no_progress = 0

            if alerts:
                hostname = os.uname().nodename
                header = f"🚨 *Bench Runner Monitor* ({hostname})\n{'=' * 30}\n\n"
                message = header + "\n\n".join(alerts)
                send_telegram(message)

        except Exception as exc:
            logger.error("Monitor loop error: %s", exc, exc_info=True)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
