"""Tests for the timestamped backups of the database."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from job_hunters import db as db_module
from job_hunters.backup import BackupError, backup_database
from job_hunters.models import Company

NOW = datetime(2026, 9, 10, 2, 0, tzinfo=UTC)


def _rows(path: Path, query: str) -> list[tuple]:
    """Reads the backup back through a fresh connection of its own."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        return connection.execute(query).fetchall()


def test_a_backup_carries_the_rows_that_were_in_the_database(
    session: Session, company: Company, tmp_path: Path
) -> None:
    """What was in the database is in the backup."""
    report = backup_database(tmp_path / "out", db_path=_db_path(), now=NOW)

    assert report.path.is_file()
    assert _rows(report.path, "SELECT slug FROM companies") == [("acme",)]


def test_the_name_says_when_it_was_taken(
    session: Session, company: Company, tmp_path: Path
) -> None:
    """A directory of backups is only useful if you can tell which is which."""
    report = backup_database(tmp_path / "out", db_path=_db_path(), now=NOW)
    assert report.path.name == "job_hunters-20260910-020000.db"


def test_two_backups_in_one_second_do_not_overwrite_each_other(
    session: Session, company: Company, tmp_path: Path
) -> None:
    """Running the command twice by hand must not discard the first backup."""
    first = backup_database(tmp_path / "out", db_path=_db_path(), now=NOW)
    second = backup_database(tmp_path / "out", db_path=_db_path(), now=NOW)

    assert first.path != second.path
    assert first.path.is_file() and second.path.is_file()


def test_the_backup_directory_is_created_if_it_is_not_there(
    session: Session, company: Company, tmp_path: Path
) -> None:
    """`backups/` is a bind mount that may not exist yet on a fresh clone."""
    target = tmp_path / "never" / "existed"
    report = backup_database(target, db_path=_db_path(), now=NOW)
    assert report.path.parent == target


def test_a_backup_is_readable_on_its_own_and_names_its_tables(
    session: Session, company: Company, tmp_path: Path
) -> None:
    """A backup missing half the schema would look like a file and restore like nothing."""
    report = backup_database(tmp_path / "out", db_path=_db_path(), now=NOW)
    assert report.tables == 8
    assert report.bytes_written > 0


def test_a_write_in_flight_does_not_reach_a_half_written_copy(
    session: Session, company: Company, tmp_path: Path
) -> None:
    """The reason this uses SQLite's backup API and not `cp`: WAL keeps the newest rows elsewhere."""
    session.add(Company(slug="beta", name="Beta", ats_type="lever",
                        ats_config={"token": "beta"}, tier="lab"))
    session.commit()

    report = backup_database(tmp_path / "out", db_path=_db_path(), now=NOW)
    slugs = {row[0] for row in _rows(report.path, "SELECT slug FROM companies")}
    assert slugs == {"acme", "beta"}, "a plain file backup would have missed the committed row"


def test_a_backup_is_one_file_and_not_three(
    session: Session, company: Company, tmp_path: Path
) -> None:
    """A WAL backup is a database plus two journals and only one of them looks like a backup."""
    report = backup_database(tmp_path / "out", db_path=_db_path(), now=NOW)
    assert [path.name for path in (tmp_path / "out").iterdir()] == [report.path.name]


def test_backing_up_nothing_says_so_rather_than_writing_an_empty_file(
    tmp_path: Path,
) -> None:
    """An empty file in `backups/` would look exactly like a backup that worked."""
    with pytest.raises(BackupError, match="init-db"):
        backup_database(tmp_path / "out", db_path=tmp_path / "absent.db", now=NOW)
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


def test_a_database_older_than_the_code_is_still_copied_and_reported(
    session: Session, company: Company, tmp_path: Path
) -> None:
    """That is the database you most want a backup of since the repair for it drops a table."""
    source = _db_path()
    with sqlite3.connect(source) as connection:
        connection.execute("DROP TABLE digest_appearances")

    report = backup_database(tmp_path / "out", db_path=source, now=NOW)
    assert report.path.is_file()
    assert report.missing == ("digest_appearances",)
    assert report.tables == 7


def test_a_copy_that_fails_halfway_leaves_nothing_behind(tmp_path: Path) -> None:
    """A half-written file in `backups/` is worse than none because it reads as a backup."""
    not_a_database = tmp_path / "junk.db"
    not_a_database.write_bytes(b"this is not a database")

    with pytest.raises(BackupError, match="Could not backup"):
        backup_database(tmp_path / "out", db_path=not_a_database, now=NOW)
    assert list((tmp_path / "out").iterdir()) == []


def test_a_copy_that_will_not_reopen_is_removed(tmp_path: Path) -> None:
    """The check exists to catch a file that appeared without being restorable."""
    from job_hunters.backup import _verify

    unreadable = tmp_path / "job_hunters-20260910-020000.db"
    unreadable.write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)

    with pytest.raises(BackupError):
        _verify(unreadable)
    assert not unreadable.exists()


def _db_path() -> Path:
    """Where the `session` fixture put this test's throwaway database."""
    return Path(db_module.get_engine().url.database or "")
