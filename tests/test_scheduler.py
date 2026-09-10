"""Tests for the process that runs ingest, score and digest on a schedule."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from job_hunters import scheduler as scheduler_module
from job_hunters.config import ConfigError, SchedulesConfig
from job_hunters.scheduler import (
    build_trigger,
    scheduled_digest,
    scheduled_ingest,
    scheduled_score,
)

LISBON = ZoneInfo("Europe/Lisbon")


def test_an_interval_schedule_becomes_an_interval_trigger() -> None:
    """Every 2h is two hours apart and not twice a day."""
    trigger = build_trigger("every 2h", LISBON)
    assert isinstance(trigger, IntervalTrigger)
    assert trigger.interval.total_seconds() == 2 * 3600


@pytest.mark.parametrize(
    ("spec", "seconds"),
    [("every 30m", 1800), ("every 2h", 7200), ("every 1d", 86400)],
)
def test_every_interval_unit_is_understood(spec: str, seconds: int) -> None:
    """Minutes, hours and days are the three the config regex allows."""
    assert build_trigger(spec, LISBON).interval.total_seconds() == seconds


def test_a_daily_schedule_fires_at_the_declared_hour() -> None:
    """An 08:00 digest has to actually be 08:00 in the configured timezone."""
    trigger = build_trigger("daily 08:00", LISBON)
    assert isinstance(trigger, CronTrigger)
    fields = {field.name: str(field) for field in trigger.fields}
    assert fields["hour"] == "8" and fields["minute"] == "0"


def test_a_weekly_schedule_carries_its_day() -> None:
    """"weekly sun 02:00" is a Sunday and losing the day would make it daily."""
    trigger = build_trigger("weekly sun 02:00", LISBON)
    fields = {field.name: str(field) for field in trigger.fields}
    assert fields["day_of_week"] == "sun"
    assert fields["hour"] == "2" and fields["minute"] == "0"


@pytest.mark.parametrize("spec", ["every 2h", "daily 08:00", "weekly sun 02:00"])
def test_a_trigger_keeps_the_declared_timezone_and_not_the_machine_one(spec: str) -> None:
    """The timezone comes from `system_config.yaml`, which is the point of declaring it."""
    tokyo = ZoneInfo("Asia/Tokyo")
    scheduler = BackgroundScheduler(timezone=tokyo)
    job = scheduler.add_job(lambda: None, build_trigger(spec, tokyo), id=spec)
    assert str(job.trigger.timezone) == "Asia/Tokyo"


def test_every_default_schedule_in_the_config_can_be_built() -> None:
    """The config regex and this parser have to keep agreeing about what is legal."""
    defaults = SchedulesConfig()
    specs = defaults.specs()
    assert set(specs) == {"ingest", "score", "digest", "discovery", "backup"}
    for spec in specs.values():
        assert build_trigger(spec, LISBON) is not None


def test_a_schedule_this_parser_does_not_know_is_a_config_error() -> None:
    """Config validates the shape first, so this guards the two drifting apart."""
    with pytest.raises(ConfigError):
        build_trigger("fortnightly tue 09:00", LISBON)


@pytest.mark.parametrize(
    ("job", "target"),
    [
        (scheduled_ingest, "run_ingest"),
        (scheduled_score, "run_scoring"),
        (scheduled_digest, "run_digest"),
    ],
)
def test_a_failing_job_is_logged_and_does_not_escape(job, target, monkeypatch, caplog) -> None:
    """One bad morning must be a loud line in the log and not the end of the scheduler."""
    def raise_it(**_kwargs):
        raise RuntimeError("the board is on fire")

    monkeypatch.setattr(scheduler_module, target, raise_it)
    job()
    assert "the board is on fire" in caplog.text
