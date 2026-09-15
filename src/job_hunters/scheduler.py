"""The process that runs ingest, digest, discovery and backup on a schedule.

Runs as its own container, separate from the web process, and is what turns
the commands in `cli.py` into something that happens without you. It registers
four jobs on the schedules declared in `system_config.yaml`:

    ingest     fetch every watched board and judge what
               it turned up                               (default: every 2h)
    digest     build and send the daily email             (default: daily 08:00)
    discovery  scan the sources for companies to review   (default: weekly mon 06:00)
    backup     copy the database into `backups/`          (default: weekly sun 02:00)

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
from .backup import backup_database
from .config import ConfigError, load_system_config
from .db import SchemaError, init_db
from .digest import run_digest
from .discovery import run_discovery
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
    try:
        if parts[0] == "every":
            amount, unit = int(parts[1][:-1]), parts[1][-1]
            if amount < 1:
                raise ValueError("The interval must be at least 1")
            return IntervalTrigger(timezone=timezone, **{_UNITS[unit]: amount})
        if parts[0] == "daily":
            hour, minute = parts[1].split(":")
            return CronTrigger(hour=int(hour), minute=int(minute), timezone=timezone)
        if parts[0] == "weekly":
            hour, minute = parts[2].split(":")
            return CronTrigger(
                day_of_week=parts[1], hour=int(hour), minute=int(minute), timezone=timezone
            )
    except (ValueError, KeyError, IndexError) as exc:
        raise ConfigError(f"Unsupported schedule {spec!r} in system_config.yaml: {exc}") from exc
    raise ConfigError(f"Unsupported schedule {spec!r} in system_config.yaml.")


def scheduled_ingest() -> None:
    """Fetches every watched board, then judges what it turned up. Logs both instead of raising.

    Scoring runs after the fetch and in the same job, so it always reads what this ingest
    wrote rather than a snapshot taken while it was still writing. It runs even when the
    fetch failed. The backlog from earlier runs is still worth judging.
    """
    try:
        report = run_ingest()
        log.info(
            "ingest: %s companies, %s failed, %s postings, %s new, %s closed, %s new jobs",
            len(report.companies), len(report.failures), report.total("fetched"),
            report.total("new_sources"), report.total("closed"), report.total("new_jobs"),
        )
    except Exception:
        log.exception("ingest failed")
    scheduled_score()


def scheduled_score() -> None:
    """Judges whatever ingest turned up and logs the outcome. The second half of the ingest job."""
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


def scheduled_discovery() -> None:
    """Scans the discovery sources and queues new companies and logs the outcome."""
    try:
        report = run_discovery()
        log.info(
            "discovery: %s sources, %s failed, %s matched, %s new sightings, "
            "%s new candidates, %s waiting for review",
            len(report.sources), len(report.failures), report.total("matched"),
            report.total("new_sightings"), report.total("new_candidates"), report.pending,
        )
        if report.aborted:
            log.error("discovery: extraction stopped early: %s", report.aborted)
    except Exception:
        log.exception("discovery failed")


def scheduled_backup() -> None:
    """Copies the database into `backups/`, prunes the oldest and logs where it went."""
    try:
        report = backup_database(keep=load_system_config().backup.keep)
        log.info("backup: %s (%.1f MB, %s older removed)", report.path, report.megabytes, len(report.pruned))
    except Exception:
        log.exception("backup failed")


def main() -> int:
    """Starts the scheduler and blocks forever."""
    try:
        config = load_system_config()
        timezone = ZoneInfo(config.timezone)
        ingest_grace = config.schedules.ingest_misfire_grace_minutes * SECONDS_PER_MINUTE
        digest_grace = config.schedules.digest_misfire_grace_minutes * SECONDS_PER_MINUTE
        discovery_grace = config.schedules.discovery_misfire_grace_minutes * SECONDS_PER_MINUTE
        backup_grace = config.schedules.backup_misfire_grace_minutes * SECONDS_PER_MINUTE
        jobs = (
            ("ingest", scheduled_ingest, config.schedules.ingest, ingest_grace),
            ("digest", scheduled_digest, config.schedules.digest, digest_grace),
            ("discovery", scheduled_discovery, config.schedules.discovery, discovery_grace),
            ("backup", scheduled_backup, config.schedules.backup, backup_grace),
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
