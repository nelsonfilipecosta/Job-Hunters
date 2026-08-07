"""SQLite engine, connection pragmas and session helpers.

Two processes share this database: `web` and `scheduler` (plan section 3.11).
WAL mode is what makes that safe: readers never block the writer and the writer
never blocks readers. The pragmas below are not optional decoration. Each one
fixes a specific way SQLite would otherwise misbehave here.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from . import paths
from .models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _apply_pragmas(dbapi_connection, _connection_record) -> None:
    """Run on every new connection.

    `journal_mode` is stored in the database file itself, so setting it repeatedly
    is a no-op. The other three are per-connection and must be set every time.
    """
    cursor = dbapi_connection.cursor()
    # Concurrent readers alongside one writer. The reason two services can share
    # one file at all.
    cursor.execute("PRAGMA journal_mode=WAL")
    # Wait up to 5s for a lock instead of raising "database is locked" instantly.
    cursor.execute("PRAGMA busy_timeout=5000")
    # SQLite by default has foreign key enforcement OFF. Turning it ON is what
    # makes the ForeignKey declarations in `models.py`` mean anything at runtime.
    cursor.execute("PRAGMA foreign_keys=ON")
    # With WAL, NORMAL is durable against process crashes (only a power loss can
    # lose the last transactions) and much faster than FULL.
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def get_engine(db_path: Path | None = None, echo: bool = False) -> Engine:
    """Return the process-wide engine or create it on first use.
    
    An engine is the object in SQLAlchemy representing "the database". It
    knows the file location and manages a pool of connections. Nothing talks
    to SQLite without going through it."""
    global _engine
    if _engine is None:
        target = db_path or paths.DB_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{target}", echo=echo, future=True)
        event.listen(_engine, "connect", _apply_pragmas)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session maker or create it on first use.
    
    A session is a workspace for one unit of work: you add objects, query, then
    commit or roll back. A session factory is a pre-configured mould for producing
    them."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(db_path: Path | None = None) -> Path:
    """Create any missing tables. Existing tables are left untouched.

    `create_all` does not migrate: if a later phase adds a column, this will not
    add it to an existing database. Until there is history worth keeping, the
    fix is to delete the file and re-run. Alembic becomes worth its weight once
    real application data is in there.
    """
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return Path(engine.url.database or "")


def reset_engine() -> None:
    """Drop the cached engine. Used by tests that point at a temporary database."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
