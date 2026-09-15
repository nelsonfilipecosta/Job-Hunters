"""Timestamped backups of the database."""

from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from . import paths
from .models import Base, utcnow

log = logging.getLogger("job_hunters.backup")

FILENAME_PREFIX = "job_hunters"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
# Only files this code named are ever pruned. Anything else in the directory is left alone.
_BACKUP_NAME = re.compile(rf"^{FILENAME_PREFIX}-\d{{8}}-\d{{6}}(?:-\d+)?\.db$")


class BackupError(Exception):
    """The backup could not be written or could not be read back afterwards."""


@dataclass(frozen=True)
class BackupReport:
    """What one `job-hunters backup` wrote."""

    source: Path
    path: Path
    bytes_written: int
    tables: int
    # Tables declared by `models.py` that the backup does not hold.
    missing: tuple[str, ...] = ()
    # Older backups removed to stay within `keep`.
    pruned: tuple[Path, ...] = ()

    @property
    def megabytes(self) -> float:
        """The size in megabytes rather than in bytes."""
        return self.bytes_written / (1024 * 1024)


def backup_database(
    destination_dir: Path | None = None,
    *,
    db_path: Path | None = None,
    now: datetime | None = None,
    keep: int | None = None,
) -> BackupReport:
    """Writes one timestamped backup of the database and reads it back to prove it opens.

    With `keep`, the oldest backups beyond that many are then removed. Pruning happens only
    after the new copy has passed its check, so the directory is never emptier than it was
    found and a failed backup removes nothing.
    """
    source = db_path or paths.DB_PATH
    if not source.is_file():
        raise BackupError(
            f"There is no database at {source} to back up. "
            f"Run `job-hunters init-db` first."
        )
    directory = destination_dir or paths.BACKUP_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = _free_name(directory, now or utcnow())

    try:
        with (
            closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as origin,
            closing(sqlite3.connect(target)) as copy,
        ):
            origin.backup(copy)
            # `journal_mode` is stored inside the database file, so the backup comes out of
            # `backup()` still in WAL mode and as three files (`.db`, `-wal` and `-shm`).
            # Switching it to the older rollback-journal mode folds everything back into
            # a single `.db` file. 
            copy.execute("PRAGMA journal_mode=DELETE")
    except sqlite3.Error as exc:
        _remove(target)
        raise BackupError(f"Could not backup {source} to {target}: {exc}") from exc

    present = _verify(target)
    report = BackupReport(
        source=source,
        path=target,
        bytes_written=target.stat().st_size,
        tables=len(present),
        missing=tuple(sorted(set(Base.metadata.tables) - present)),
    )
    if report.missing:
        log.warning(
            "%s holds no %s table(s): the database it came from is older than this code.",
            target, ", ".join(report.missing),
        )
    log.info("Backed up %s to %s (%.1f MB)", source, target, report.megabytes)
    if keep is not None:
        report = replace(report, pruned=_prune(directory, keep, target))
    return report


def _prune(directory: Path, keep: int, latest: Path) -> tuple[Path, ...]:
    """Removes the oldest backups so that `keep` remain. Returns what was removed."""
    backups = sorted(
        path for path in directory.iterdir()
        if path.is_file() and _BACKUP_NAME.match(path.name) and path != latest
    )
    surplus = len(backups) + 1 - keep
    removed: list[Path] = []
    for path in backups[:max(surplus, 0)]:
        try:
            path.unlink()
        except OSError as exc:
            # The new copy is written and checked. A file that will not go is a warning.
            log.warning("Could not remove old backup %s: %s", path, exc.strerror or exc)
            continue
        removed.append(path)
        log.info("Removed old backup %s (keeping %s)", path, keep)
    return tuple(removed)


def _remove(target: Path) -> None:
    """Deletes a failed backup and any journal SQLite left beside it."""
    for path in (target, Path(f"{target}-wal"), Path(f"{target}-shm"), Path(f"{target}-journal")):
        path.unlink(missing_ok=True)


def _free_name(directory: Path, moment: datetime) -> Path:
    """Picks a filename for the backup and appends a counter if the name is already taken.

    The timestamp in the name is only precise to the second, so two backups started in
    the same second would otherwise collide - easy to hit by running a manual backup
    right after a scheduled one.
    """
    stamp = moment.strftime(TIMESTAMP_FORMAT)
    candidate = directory / f"{FILENAME_PREFIX}-{stamp}.db"
    attempt = 2
    while candidate.exists():
        candidate = directory / f"{FILENAME_PREFIX}-{stamp}-{attempt}.db"
        attempt += 1
    return candidate


def _verify(target: Path) -> set[str]:
    """Opens the backup, checks its pages and returns the tables it holds.

    A backup that will not open or does not pass SQLite's own page check is not a
    backup. It is an error and the file is removed rather than left in `backups/`.
    """
    try:
        with closing(sqlite3.connect(f"file:{target}?mode=ro", uri=True)) as copy:
            verdict = copy.execute("PRAGMA quick_check").fetchone()
            present = {
                row[0]
                for row in copy.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
    except sqlite3.Error as exc:
        _remove(target)
        raise BackupError(f"The backup at {target} could not be reopened: {exc}") from exc

    if not verdict or verdict[0] != "ok":
        _remove(target)
        raise BackupError(
            f"The backup written to {target} did not pass SQLite's own check "
            f"({verdict[0] if verdict else 'no answer'}) and has been removed. "
        )
    return present & set(Base.metadata.tables)
