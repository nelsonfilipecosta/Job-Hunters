"""Shared setup for every test in this directory."""

from __future__ import annotations

import html
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from sqlalchemy.orm import Session

from job_hunters import db as db_module
from job_hunters.judge import Verdict
from job_hunters.models import Company, Job, JobSource


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
    raw: dict | None = None,
    **hints,
) -> RawPosting:
    """A RawPosting with sensible defaults for building fetch results by hand."""
    url = url or f"https://example.test/{source}/{source_job_id}"
    return RawPosting(
        source=source,
        source_job_id=source_job_id,
        title=title,
        url=url,
        location_raw=location,
        description=description,
        raw=raw or _raw_payload(source, source_job_id, title, url, location, description, hints),
        **hints,
    )


def _raw_payload(
    source: str, source_job_id: str, title: str, url: str, location: str | None,
    description: str, hints: dict,
) -> dict:
    """A payload shaped as the board would return it, so `replay_posting` rebuilds the same posting.

    Scoring reads each posting's text back from `raw_json` through the real
    adapters, so a fake posting has to store what those adapters expect.
    """
    if source == "lever":
        return {
            "id": source_job_id, "text": title, "hostedUrl": url,
            "categories": {"location": location, "allLocations": [location] if location else []},
            "descriptionPlain": description,
            "country": hints.get("country_code"), "workplaceType": hints.get("workplace_type"),
        }
    if source == "ashby":
        return {
            "id": source_job_id, "title": title, "jobUrl": url, "location": location,
            "descriptionPlain": description, "workplaceType": hints.get("workplace_type"),
            "isRemote": hints.get("is_remote"), "isListed": True,
        }
    # Greenhouse: HTML that has itself been HTML-escaped.
    return {
        "id": source_job_id, "title": title, "absolute_url": url,
        "location": {"name": location},
        "content": html.escape(f"<p>{html.escape(description)}</p>"),
    }


def make_source(
    company: Company,
    job: Job | None = None,
    *,
    source: str = "greenhouse",
    source_job_id: str = "1",
    content_hash: str = "c" * 64,
    raw_hash: str | None = None,
) -> JobSource:
    """An unsaved JobSource with both hashes filled for tests that need a posting row."""
    return JobSource(
        company_id=company.id,
        job_id=job.id if job is not None else None,
        source=source,
        source_job_id=source_job_id,
        raw_hash=raw_hash or content_hash,
        content_hash=content_hash,
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


def verdict(score: int = 80, **overrides) -> Verdict:
    """A complete Verdict with one score and boring defaults for everything else."""
    fields = dict(
        score=score, summary="A role. It fits.", rationale="Because.",
        matched_areas=["post-training"], concerns=[], work_authorization="eligible",
    )
    fields.update(overrides)
    return Verdict(**fields)


def api_error(kind: type[anthropic.APIStatusError], status: int, message: str = "nope") -> anthropic.APIStatusError:
    """A real SDK status error since the judge classifies them by type."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return kind(message, response=httpx.Response(status, request=request), body=None)


class FakeAnthropic:
    """Stands in for `anthropic.Anthropic`. Records every request and answers from a script."""

    def __init__(self, *script) -> None:
        """Scripts one answer per call, defaulting to a verdict of 80 for everything."""
        self.requests: list[dict] = []
        self._script = list(script) or [verdict()]
        self.messages = self  # so `client.messages.parse(...)` lands on `parse` below

    def parse(self, **kwargs):
        """Returns the next scripted answer, shaped like the SDK's parsed message."""
        self.requests.append(kwargs)
        item = self._script[min(len(self.requests), len(self._script)) - 1]
        if isinstance(item, BaseException):
            raise item
        answer = item(kwargs) if callable(item) else item
        first = len(self.requests) == 1
        usage = SimpleNamespace(
            input_tokens=1500, output_tokens=200,
            cache_creation_input_tokens=4300 if first else 0,
            cache_read_input_tokens=0 if first else 4300,
        )
        return SimpleNamespace(parsed_output=answer, usage=usage, stop_reason="end_turn")
