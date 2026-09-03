"""Shared setup for every test in this directory.

pytest loads this file automatically before running anything beside it. It is
never imported by name and the name `conftest.py` is what makes that happen.

What it holds are fixtures: named pieces of setup a test asks for by writing the
name in its own signature. A test declared as `def test_x(session)` gets handed
whatever the `session` fixture below produces. Keeping them here means the
database setup is written once rather than repeated in every test file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from job_hunters import db as db_module
from job_hunters.models import Company


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    """A session against a throwaway database with the real pragmas applied.

    Every test gets its own empty database file, so no test can see or corrupt
    another's rows. `tmp_path` is a built-in pytest fixture supplying a fresh
    temporary directory per test.

    The engine is reset on both sides because `db.py` caches it for the whole
    process. Without clearing it, every later test would keep using the first
    test's database. Code before `yield` is setup and code after it is teardown.
    """
    db_module.reset_engine()
    db_module.init_db(tmp_path / "test.db")
    factory = db_module.get_session_factory()
    with factory() as active:
        yield active
    db_module.reset_engine()


@pytest.fixture
def company(session: Session) -> Company:
    """A saved Company row for tests that need something to attach jobs to.

    Foreign keys are enforced, so a job cannot be inserted without a company
    that really exists. Taking `session` as an argument is how one fixture
    depends on another. Pytest builds the database first, then this row in it.
    """
    entry = Company(slug="acme", name="Acme", ats_type="greenhouse",
                    ats_config={"token": "acme"}, tier="lab")
    session.add(entry)
    session.commit()
    return entry
