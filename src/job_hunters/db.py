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

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

from . import paths
from .models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


class SchemaError(Exception):
    """The database on disk cannot answer what this code is about to ask it.

    Either it was never initialised or it is older than this code and is
    missing a column. `require_current_schema` tells the two apart.
    """


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
    them.

    The schema is checked here rather than only in `init_db`, because most
    commands never call `init_db` at all - they just open a session and query.
    The factory is built once per process, so this costs one look at SQLite's
    own metadata for the whole run."""

    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        require_current_schema(engine)
        _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
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


def missing_tables(engine: Engine) -> list[str]:
    """Tables declared in `models.py` that the database on disk does not have."""
    present = set(inspect(engine).get_table_names())
    return [name for name in Base.metadata.tables if name not in present]


def missing_columns(engine: Engine) -> dict[str, list[str]]:
    """Columns declared in `models.py` that a table on disk does not have.

    Only tables that already exist are inspected. Extra columns in the database are ignored.
    """
    inspector = inspect(engine)
    present = set(inspector.get_table_names())
    gaps: dict[str, list[str]] = {}
    for name, table in Base.metadata.tables.items():
        if name not in present:
            continue
        have = {column["name"] for column in inspector.get_columns(name)}
        if absent := [c.name for c in table.columns if c.name not in have]:
            gaps[name] = absent
    return gaps


def require_current_schema(engine: Engine) -> None:
    """Raises `SchemaError` when the database cannot answer what this code will ask.

    Two different faults, with two different repairs. A table that does not
    exist yet is a database nobody has initialised and `init-db` fixes it
    outright. A table that exists without one of its columns is a database
    older than this code and `create_all` will never repair that. It issues
    "create table if not exists" and stops.

    Both are checked here rather than only in `init_db`, because most commands
    never call `init_db` at all. They open a session and query.
    """
    absent = missing_tables(engine)
    if absent:
        what = (
            "has no tables at all"
            if len(absent) == len(Base.metadata.tables)
            else f"is missing the {', '.join(absent)} table"
            f"{'s' if len(absent) > 1 else ''}"
        )
        raise SchemaError(
            f"The database at {engine.url.database} {what}. "
            f"Run `job-hunters init-db` to create the schema."
        )
    gaps = missing_columns(engine)
    if not gaps:
        return
    listed = "\n".join(f"  {table}: {', '.join(columns)}" for table, columns in gaps.items())
    raise SchemaError(
        f"The database at {engine.url.database} is older than this code and is "
        f"missing column(s):\n{listed}\n"
        f"`init-db` creates missing tables but never adds a column to a table that "
        f"already exists. Drop the table(s) above and run `job-hunters init-db` "
        f"again to have them rebuilt - but look at what is in them first, because "
        f"dropping one discards its rows."
    )


def init_db(db_path: Path | None = None) -> Path:
    """Create any missing tables. Existing tables are left untouched.

    `create_all` does not migrate: if a later phase adds a column, this will not
    add it to an existing database. Until there is history worth keeping, the
    fix is to delete the file and re-run. Alembic becomes worth its weight once
    real application data is in there."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    require_current_schema(engine)
    return Path(engine.url.database or "")


def reset_engine() -> None:
    """Drop the cached engine. Used by tests that point at a temporary database."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
