#!/usr/bin/env python3
"""
Bench Runner Queue Health Monitor with Telegram Alerts.

Monitors the bench-runner queue and sends urgent Telegram notifications when:
- More than 50 runs are queued
- Any run has been queued for longer than 4 hours

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

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "8264160091:AAHJVlv2MYbaU4plbpmBLnKn6Wi-vG52nGM")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "7367548582")

QUEUE_DEPTH_THRESHOLD = int(os.getenv("QUEUE_DEPTH_THRESHOLD", "50"))
QUEUE_AGE_THRESHOLD_HOURS = float(os.getenv("QUEUE_AGE_THRESHOLD_HOURS", "4"))
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))  # 5 min
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "1800"))  # 30 min
API_TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "15"))

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


def get_queue_stats() -> tuple[int, int, float | None]:
    """Return (queued_count, running_count, oldest_queued_age_hours)."""
    # Get queued runs
    queued_data = _api_get("/api/runs?status=queued&limit=200")
    running_data = _api_get("/api/runs?status=running&limit=200")

    if queued_data is None or running_data is None:
        return -1, -1, None

    # Extract counts
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

    # Find oldest queued run
    oldest_age_hours = None
    now = datetime.now(timezone.utc)
    for run in queued_runs:
        created_at = run.get("created_at") or run.get("createdAt")
        if not created_at:
            continue
        try:
            if isinstance(created_at, str):
                # Handle ISO format with or without Z
                created_at = created_at.replace("Z", "+00:00")
                if "+" not in created_at and created_at.endswith("00:00"):
                    pass  # already has timezone
                dt = datetime.fromisoformat(created_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            else:
                continue
            age_hours = (now - dt).total_seconds() / 3600
            if oldest_age_hours is None or age_hours > oldest_age_hours:
                oldest_age_hours = age_hours
        except (ValueError, TypeError):
            continue

    return queued_count, running_count, oldest_age_hours


# ---------------------------------------------------------------------------
# Telegram
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
                logger.info("Telegram alert sent successfully")
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
        "Queue health monitor started. backend=%s queue_threshold=%d age_threshold=%.1fh interval=%ds",
        BACKEND_URL, QUEUE_DEPTH_THRESHOLD, QUEUE_AGE_THRESHOLD_HOURS, CHECK_INTERVAL_SECONDS,
    )

    last_depth_alert_at = 0.0
    last_age_alert_at = 0.0
    last_stuck_workers_alert_at = 0.0
    consecutive_no_progress = 0
    last_running_count = -1

    while True:
        try:
            queued, running, oldest_age = get_queue_stats()

            if queued < 0:
                logger.warning("Failed to fetch queue stats, will retry")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            logger.info(
                "Queue check: queued=%d running=%d oldest_age=%.1fh",
                queued, running, oldest_age or 0,
            )

            now = time.time()
            alerts = []

            # Check 1: Queue depth > threshold
            if queued > QUEUE_DEPTH_THRESHOLD and (now - last_depth_alert_at) > ALERT_COOLDOWN_SECONDS:
                alerts.append(
                    f"*QUEUE DEPTH ALERT*\n"
                    f"Queued runs: *{queued}* (threshold: {QUEUE_DEPTH_THRESHOLD})\n"
                    f"Running runs: {running}"
                )
                last_depth_alert_at = now

            # Check 2: Oldest queued run > age threshold
            if (
                oldest_age is not None
                and oldest_age > QUEUE_AGE_THRESHOLD_HOURS
                and (now - last_age_alert_at) > ALERT_COOLDOWN_SECONDS
            ):
                alerts.append(
                    f"*QUEUE AGE ALERT*\n"
                    f"Oldest queued run: *{oldest_age:.1f}h* (threshold: {QUEUE_AGE_THRESHOLD_HOURS}h)\n"
                    f"Queued: {queued} | Running: {running}"
                )
                last_age_alert_at = now

            # Check 3: Detect stuck workers (running count unchanged with large queue)
            if queued > 20 and running == last_running_count and running <= 4:
                consecutive_no_progress += 1
            else:
                consecutive_no_progress = 0
            last_running_count = running

            # Alert after 6 consecutive checks (~30 min) with no progress
            if consecutive_no_progress >= 6 and (now - last_stuck_workers_alert_at) > ALERT_COOLDOWN_SECONDS * 2:
                alerts.append(
                    f"*STUCK WORKERS ALERT*\n"
                    f"Only {running} runs active for 30+ min with {queued} queued.\n"
                    f"Workers may be hung or failing silently."
                )
                last_stuck_workers_alert_at = now
                consecutive_no_progress = 0

            # Send combined alert
            if alerts:
                hostname = os.uname().nodename
                header = f"Bench Runner Monitor ({hostname})\n{'=' * 30}\n\n"
                message = header + "\n\n".join(alerts)
                send_telegram(message)

        except Exception as exc:
            logger.error("Monitor loop error: %s", exc, exc_info=True)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
