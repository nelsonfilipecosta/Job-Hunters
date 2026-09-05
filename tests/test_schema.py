"""Tests that the database enforces what the schema promises.

Every table in `models.py` claims something: that the same posting cannot be
stored twice, that a job cannot be judged twice under one prompt version or that
a foreign key must point at a row that exists. A claim is only real if the
database actually refuses the thing it forbids, so each test here tries to break
one rule and checks that the attempt fails.

Most tests therefore expect an error. `pytest.raises(IntegrityError)` means the
test passes when the database rejects the write and fails if the write got
through. If one of these ever fails, the guarantee was never real.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from conftest import make_source
from job_hunters import db as db_module
from job_hunters.models import (
    Application,
    ApplicationEvent,
    ApplicationStatus,
    Company,
    DigestAppearance,
    EventKind,
    Job,
    JobSource,
    Score,
)

EXPECTED_TABLES = {
    "companies",
    "job_sources",
    "jobs",
    "scores",
    "applications",
    "application_events",
    "digest_appearances",
    "fetch_runs",
}


def test_all_tables_created(session: Session) -> None:
    """All eight tables are created and nothing else is."""
    assert set(inspect(session.get_bind()).get_table_names()) == EXPECTED_TABLES


def test_wal_mode_is_enabled(session: Session) -> None:
    """WAL mode is on. Without it, `web` and `scheduler` would block each other."""
    assert session.execute(text("PRAGMA journal_mode")).scalar() == "wal"


def test_foreign_keys_are_enforced(session: Session) -> None:
    """A job cannot reference a company that does not exist."""
    assert session.execute(text("PRAGMA foreign_keys")).scalar() == 1

    session.add(Job(canonical_key="orphan", company_id=9999, title="Nowhere"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_source_identity_is_unique(session: Session, company: Company) -> None:
    """The same posting fetched twice is stored once, so re-ingest is idempotent."""
    for _ in range(2):
        session.add(make_source(company, source="greenhouse", source_job_id="4001"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_same_id_from_different_sources_is_allowed(
    session: Session, company: Company
) -> None:
    """Identical job ids on two different boards stay two separate postings."""
    session.add(make_source(company, source="greenhouse", source_job_id="1"))
    session.add(make_source(company, source="lever", source_job_id="1"))
    session.commit()
    assert session.query(JobSource).count() == 2


def test_canonical_key_is_unique(session: Session, company: Company) -> None:
    """Two jobs cannot share a canonical key, which is what deduplication needs."""
    for _ in range(2):
        session.add(Job(canonical_key="same-key", company_id=company.id, title="RS"))
    with pytest.raises(IntegrityError):
        session.commit()


def _make_job(session: Session, company: Company, key: str = "k") -> Job:
    """Save one job for the tests below to hang postings and applications off."""
    job = Job(canonical_key=key, company_id=company.id, title="Research Scientist",
              content_hash="c" * 64)
    session.add(job)
    session.commit()
    return job


def _make_source(
    session: Session, company: Company, job: Job, source_job_id: str = "1",
    content_hash: str = "c" * 64,
) -> JobSource:
    """Save one posting of a job for the scoring tests to hang judgements off."""
    source = make_source(company, job, source_job_id=source_job_id, content_hash=content_hash)
    session.add(source)
    session.commit()
    return source


def test_a_posting_is_scored_once_per_content_and_prompt_version(
    session: Session, company: Company
) -> None:
    """Unchanged posting text is never judged twice under the same prompt version."""
    source = _make_source(session, company, _make_job(session, company))
    for _ in range(2):
        session.add(
            Score(source_id=source.id, score=80, prompt_version=1,
                  model="claude-haiku-4-5", content_hash="c" * 64)
        )
    with pytest.raises(IntegrityError):
        session.commit()


def test_a_new_prompt_version_can_rescore_the_same_content(
    session: Session, company: Company
) -> None:
    """Bumping the prompt version lets the same posting text be judged again."""
    source = _make_source(session, company, _make_job(session, company))
    session.add(Score(source_id=source.id, score=80, prompt_version=1,
                      model="claude-haiku-4-5", content_hash="c" * 64))
    session.add(Score(source_id=source.id, score=65, prompt_version=2,
                      model="claude-haiku-4-5", content_hash="c" * 64))
    session.commit()
    assert session.query(Score).count() == 2


def test_each_posting_of_a_job_is_judged_on_its_own_row(
    session: Session, company: Company
) -> None:
    """Two postings of one job are scored separately, even with identical text, and both reach `job.scores`."""
    job = _make_job(session, company)
    first = _make_source(session, company, job, source_job_id="1")
    second = _make_source(session, company, job, source_job_id="2")
    session.add(Score(source_id=first.id, score=45, prompt_version=1,
                      model="claude-haiku-4-5", content_hash="c" * 64))
    session.add(Score(source_id=second.id, score=80, prompt_version=1,
                      model="claude-haiku-4-5", content_hash="c" * 64))
    session.commit()
    session.refresh(job)
    assert sorted(s.score for s in job.scores) == [45, 80]


def test_at_most_one_application_per_job(session: Session, company: Company) -> None:
    """The same job cannot be applied to twice."""
    job = _make_job(session, company)
    for _ in range(2):
        session.add(Application(job_id=job.id, status=ApplicationStatus.APPLIED))
    with pytest.raises(IntegrityError):
        session.commit()


def test_a_job_appears_at_most_once_per_digest(
    session: Session, company: Company
) -> None:
    """The same job cannot be recorded twice in one day's digest."""
    job = _make_job(session, company)
    for _ in range(2):
        session.add(
            DigestAppearance(job_id=job.id, digest_date=date(2026, 8, 6),
                             section="priority", position=1)
        )
    with pytest.raises(IntegrityError):
        session.commit()


def test_application_events_form_an_ordered_timeline(
    session: Session, company: Company
) -> None:
    """Events come back in the order they happened, not the order they were saved."""
    job = _make_job(session, company)
    application = Application(job_id=job.id, status=ApplicationStatus.IN_PROCESS)
    session.add(application)
    session.commit()

    from datetime import UTC, datetime

    session.add_all(
        [
            ApplicationEvent(application_id=application.id, event=EventKind.TECHNICAL,
                             occurred_at=datetime(2026, 3, 1, tzinfo=UTC)),
            ApplicationEvent(application_id=application.id, event=EventKind.APPLIED,
                             occurred_at=datetime(2026, 2, 1, tzinfo=UTC)),
            ApplicationEvent(application_id=application.id,
                             event=EventKind.RECRUITER_SCREEN,
                             occurred_at=datetime(2026, 2, 14, tzinfo=UTC)),
        ]
    )
    session.commit()
    session.refresh(application)

    assert [event.event for event in application.events] == [
        EventKind.APPLIED,
        EventKind.RECRUITER_SCREEN,
        EventKind.TECHNICAL,
    ]


def test_many_sightings_collapse_to_one_job(session: Session, company: Company) -> None:
    """One job can be reached from several postings, which is what dedup produces."""
    job = _make_job(session, company)
    session.add_all(
        [
            make_source(company, job, source="greenhouse", source_job_id="1"),
            make_source(company, job, source="remoteok", source_job_id="9"),
        ]
    )
    session.commit()
    session.refresh(job)
    assert len(job.sources) == 2
    assert {source.source for source in job.sources} == {"greenhouse", "remoteok"}


def test_init_db_is_idempotent(tmp_path) -> None:
    """Creating the schema twice is safe, since it runs at every container start."""
    db_module.reset_engine()
    target = tmp_path / "twice.db"
    db_module.init_db(target)
    db_module.init_db(target)
    db_module.reset_engine()
    assert target.exists()
