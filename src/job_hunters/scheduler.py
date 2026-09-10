"""The process that runs ingest, score and digest on a schedule.

Runs as its own container, separate from the web process, and is what turns
the commands in `cli.py` into something that happens without you. It registers
three jobs on the schedules declared in `system_config.yaml`:

    ingest   fetch every watched board          (default: every 2h)
    score    judge whatever ingest turned up    (default: every 2h)
    digest   build and send the daily email     (default: daily 08:00)

`discovery` and `backup` are configured but not registered yet: the code they
would call arrives in Phases 5 and 4. Registering a job that cannot run would
only produce a stack trace every week.

Kept separate from `web.py` on purpose. If the scheduler ran inside uvicorn
and the worker count were ever raised above one, every worker would start its
own copy - meaning duplicate digest emails and concurrent ingests. Two
processes makes that structurally impossible."""

from __future__ import annotations

import logging
from datetime import tzinfo
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from . import paths
from .config import ConfigError, load_system_config
from .db import SchemaError, init_db
from .digest import run_digest
from .ingest import run_ingest
from .scoring import run_scoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("job_hunters.scheduler")

_UNITS = {"m": "minutes", "h": "hours", "d": "days"}

SECONDS_PER_MINUTE = 60


def build_trigger(spec: str, timezone: tzinfo) -> BaseTrigger:
    """Turns one schedule string from `system_config.yaml` into an APScheduler trigger."""
    parts = spec.split()
    if parts[0] == "every":
        amount, unit = int(parts[1][:-1]), parts[1][-1]
        return IntervalTrigger(timezone=timezone, **{_UNITS[unit]: amount})
    if parts[0] == "daily":
        hour, minute = parts[1].split(":")
        return CronTrigger(hour=int(hour), minute=int(minute), timezone=timezone)
    if parts[0] == "weekly":
        hour, minute = parts[2].split(":")
        return CronTrigger(
            day_of_week=parts[1], hour=int(hour), minute=int(minute), timezone=timezone
        )
    raise ConfigError(f"Unsupported schedule {spec!r} in system_config.yaml.")


def scheduled_ingest() -> None:
    """Fetches every watched board and logs the outcome instead of raising."""
    try:
        report = run_ingest()
        log.info(
            "ingest: %s companies, %s failed, %s postings, %s new, %s closed, %s new jobs",
            len(report.companies), len(report.failures), report.total("fetched"),
            report.total("new_sources"), report.total("closed"), report.total("new_jobs"),
        )
    except Exception:
        log.exception("ingest failed")


def scheduled_score() -> None:
    """Judges whatever the last ingest turned up and logs the outcome."""
    try:
        report = run_scoring()
        log.info(
            "score: %s texts judged, %s scores written, %s failed, %s carried over",
            report.judged, report.scored, report.failed, report.carried_over,
        )
        if report.aborted:
            log.error("score: stopped early: %s", report.aborted)
    except Exception:
        log.exception("scoring failed")


def scheduled_digest() -> None:
    """Builds and sends the daily email and logs the outcome."""
    try:
        report = run_digest()
        log.info(
            "digest: %s entries sent to %s, %s suppressed",
            report.digest.total, report.sent_to, report.digest.still_open,
        )
    except Exception:
        log.exception("digest failed")


def main() -> int:
    """Starts the scheduler and blocks forever."""
    try:
        config = load_system_config()
        timezone = ZoneInfo(config.timezone)
        interval_grace = config.schedules.misfire_grace_minutes * SECONDS_PER_MINUTE
        digest_grace = config.schedules.digest_misfire_grace_minutes * SECONDS_PER_MINUTE
        jobs = (
            ("ingest", scheduled_ingest, config.schedules.ingest, interval_grace),
            ("score", scheduled_score, config.schedules.score, interval_grace),
            ("digest", scheduled_digest, config.schedules.digest, digest_grace),
        )
        triggers = [
            (name, func, spec, build_trigger(spec, timezone), grace)
            for name, func, spec, grace in jobs
        ]
    except ConfigError as exc:
        log.error("Cannot start: %s", exc)
        return 1

    # Whichever of `web` and `scheduler` starts first creates the schema in the
    # otherwise empty Docker volume. Both calls are idempotent and `create_all`
    # issues "create table if not exists", so the two racing is harmless.
    paths.ensure_runtime_dirs()
    try:
        init_db()
    except SchemaError as exc:
        # Nothing this process does would work, and restarting will not help.
        # Say what to fix rather than crash-looping on a traceback.
        log.error("Cannot start: %s", exc)
        return 1

    scheduler = BlockingScheduler(timezone=timezone)
    for name, func, spec, trigger, grace in triggers:
        scheduler.add_job(
            func,
            trigger,
            id=name,
            name=name,
            # One at a time. A slow ingest must not have a second one start on
            # top of it and fight the first for the same SQLite writer lock.
            max_instances=1,
            # A backlog of missed runs collapses into a single one.
            coalesce=True,
            misfire_grace_time=grace,
        )
        log.info("Registered %s (%s, grace %s min)", name, spec, grace // SECONDS_PER_MINUTE)

    log.info("Scheduler starting (timezone: %s)", timezone)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
