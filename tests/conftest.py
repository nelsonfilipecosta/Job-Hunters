"""Shared setup for every test in this directory."""

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


from job_hunters.sources.base import FetchResult, RawPosting


def make_posting(
    source_job_id: str,
    title: str = "Research Scientist",
    *,
    source: str = "greenhouse",
    location: str | None = "Lisbon, Portugal",
    description: str = "We do post-training and evals.",
    url: str | None = None,
    **hints,
) -> RawPosting:
    """A RawPosting with sensible defaults for building fetch results by hand."""
    return RawPosting(
        source=source,
        source_job_id=source_job_id,
        title=title,
        url=url or f"https://example.test/{source}/{source_job_id}",
        location_raw=location,
        description=description,
        raw={"id": source_job_id, "title": title, "location": location, "body": description},
        **hints,
    )


class FakeAdapter:
    """An adapter that returns a scripted sequence of results - one per fetch() call.

    The last result repeats once the script runs out, so a test can say
    "fetch these three postings, then fetch them again" by scripting one result.
    """

    def __init__(self, source: str, *results: FetchResult) -> None:
        """Scripts one FetchResult per call to fetch(), in order."""
        self.source = source
        self._results = list(results)
        self.calls = 0

    @classmethod
    def returning(cls, source: str, *postings: RawPosting) -> "FakeAdapter":
        """A FakeAdapter that always succeeds with these postings, once ingested."""
        return cls(source, FetchResult.ok(list(postings)))

    def fetch(self, company) -> FetchResult:
        """Returns the next scripted result, repeating the last one once exhausted."""
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]


def ok(*postings: RawPosting) -> FetchResult:
    """A successful FetchResult carrying these postings."""
    return FetchResult.ok(list(postings))


def failed(error: str = "HTTP 429 for https://example.test") -> FetchResult:
    """A failed FetchResult with a plausible default error message."""
    return FetchResult.failed(error)
