"""The scheduler process.

Runs as its own container, separate from the web process. Phase 3 registers
the real jobs here (ingest, score, digest, discovery, backup). For now this
only starts an empty scheduler and blocks, which is enough to prove the
service comes up and reads the right timezone.

Kept separate from `web.py` on purpose. If the scheduler ran inside uvicorn
and the worker count were ever raised above one, every worker would start its
own copy - meaning duplicate digest emails and concurrent ingests. Two
processes makes that structurally impossible. See plan section 3.11.
"""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler

from .config import ConfigError, load_system_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("job_hunters.scheduler")


def main() -> int:
    """Starts the scheduler and blocks forever."""
    # The timezone comes from `system_config.yaml` rather than the TZ environment
    # variable, so schedules are driven by the same declared value everywhere.
    try:
        timezone = ZoneInfo(load_system_config().timezone)
    except ConfigError as exc:
        log.error("Cannot start: %s", exc)
        return 1

    scheduler = BlockingScheduler(timezone=timezone)

    log.info("Scheduler starting (timezone: %s)", timezone)
    log.info("No jobs registered yet. Phase 3 adds them.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
