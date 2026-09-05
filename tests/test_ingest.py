"""Tests for fetching boards into the database.

Nothing here touches the network: `FakeAdapter` scripts each fetch's result.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import FakeAdapter, failed, make_posting, ok
from job_hunters.config import CompanyEntry
from job_hunters import ingest as ingest_module
from job_hunters.ingest import ingest_company, run_ingest, sync_companies
from job_hunters.models import Company, FetchRun, FetchStatus, Job, JobSource, as_utc
from job_hunters.sources.base import parse_iso_datetime

NOW = datetime(2026, 9, 1, tzinfo=UTC)
LATER = NOW + timedelta(hours=2)


def _counts(session: Session) -> tuple[int, int]:
    """The (job_sources, jobs) row counts after reloading from the database."""
    session.expire_all()
    return (len(session.scalars(select(JobSource)).all()), len(session.scalars(select(Job)).all()))


def _open_ids(session: Session) -> set[str]:
    """The source_job_ids of every job_sources row still marked open."""
    session.expire_all()
    return {s.source_job_id for s in session.scalars(select(JobSource)) if s.is_open}


def test_ingesting_twice_changes_nothing(session: Session, company: Company) -> None:
    """Identical counts after the second run. This is the gate."""
    adapter = FakeAdapter.returning(
        "greenhouse",
        make_posting("1", "Research Scientist"),
        make_posting("2", "Research Engineer"),
        make_posting("3", "Member of Technical Staff"),
    )
    first = ingest_company(session, company, adapter, NOW)
    session.commit()
    before = _counts(session)

    second = ingest_company(session, company, adapter, LATER)
    session.commit()
    after = _counts(session)

    assert before == after == (3, 3)
    assert first.new_sources == 3 and first.new_jobs == 3
    assert second.new_sources == 0 and second.new_jobs == 0 and second.updated_sources == 0
    assert all(s.last_seen.replace(tzinfo=UTC) == LATER for s in session.scalars(select(JobSource)))


def test_run_ingest_end_to_end_is_idempotent(session: Session) -> None:
    """Through the top-level entry point with the watchlist sync included."""
    watchlist = [CompanyEntry(slug="acme", name="Acme", ats="greenhouse", token="acme", tier="lab")]
    adapters = {"greenhouse": FakeAdapter.returning(
        "greenhouse", make_posting("1", "Research Scientist"), make_posting("2", "Research Engineer")
    )}

    run_ingest(adapters=adapters, watchlist=watchlist)
    before = _counts(session)
    report = run_ingest(adapters=adapters, watchlist=watchlist)
    after = _counts(session)

    assert before == after == (2, 2)
    assert not report.failures
    assert report.total("new_sources") == 0


def test_a_missing_posting_is_closed_after_a_successful_non_empty_fetch(
    session: Session, company: Company
) -> None:
    """A posting absent from a good and non-empty fetch is closed."""
    adapter = FakeAdapter(
        "greenhouse",
        ok(make_posting("1"), make_posting("2"), make_posting("3")),
        ok(make_posting("1"), make_posting("3")),  # 2 is gone
    )
    ingest_company(session, company, adapter, NOW)
    report = ingest_company(session, company, adapter, LATER)
    session.commit()
    assert _open_ids(session) == {"1", "3"}
    assert report.closed == 1


def test_a_failed_fetch_closes_nothing(session: Session, company: Company) -> None:
    """Condition 1 of the rule: a 429 or 404 must not look like a hiring freeze."""
    adapter = FakeAdapter("greenhouse", ok(make_posting("1"), make_posting("2")), failed("HTTP 429"))
    ingest_company(session, company, adapter, NOW)
    report = ingest_company(session, company, adapter, LATER)
    session.commit()

    assert _open_ids(session) == {"1", "2"}
    assert report.failed and "429" in report.error
    runs = session.scalars(select(FetchRun).order_by(FetchRun.id)).all()
    assert [r.status for r in runs] == [FetchStatus.OK, FetchStatus.FAILED]
    assert runs[1].error == "HTTP 429" and runs[1].item_count == 0


def test_a_successful_but_empty_fetch_closes_nothing(session: Session, company: Company) -> None:
    """Condition 2: an empty list with HTTP 200 is what a wrong token returns."""
    adapter = FakeAdapter("greenhouse", ok(make_posting("1"), make_posting("2")), ok())
    ingest_company(session, company, adapter, NOW)
    report = ingest_company(session, company, adapter, LATER)
    session.commit()

    assert _open_ids(session) == {"1", "2"}
    assert not report.failed and report.fetched == 0 and report.closed == 0


def test_a_posting_that_returns_is_reopened(session: Session, company: Company) -> None:
    """A posting that disappears and comes back is reopened and not duplicated."""
    adapter = FakeAdapter(
        "greenhouse",
        ok(make_posting("1", "Research Scientist"), make_posting("2", "Research Engineer")),
        ok(make_posting("1", "Research Scientist")),
        ok(make_posting("1", "Research Scientist"), make_posting("2", "Research Engineer")),
    )
    for i in range(3):
        ingest_company(session, company, adapter, NOW + timedelta(hours=i))
    session.commit()
    assert _open_ids(session) == {"1", "2"}
    assert _counts(session) == (2, 2), "reopening must not duplicate anything"


def test_closing_is_scoped_to_the_source_that_was_fetched(session: Session, company: Company) -> None:
    """A Greenhouse fetch says nothing about postings that came from Ashby."""
    ingest_company(session, company, FakeAdapter.returning("ashby", make_posting("a", source="ashby")), NOW)
    ingest_company(session, company, FakeAdapter.returning("greenhouse", make_posting("g")), LATER)
    session.commit()
    assert _open_ids(session) == {"a", "g"}


def test_an_adapter_that_raises_is_recorded_as_a_failure_not_a_crash(
    session: Session, company: Company
) -> None:
    """An adapter bug is caught and recorded - not left to crash the whole run."""

    class Broken:
        source = "greenhouse"

        def fetch(self, company):
            """Always raises, simulating a bug inside an adapter."""
            raise RuntimeError("bug in adapter")

    ingest_company(session, company, FakeAdapter.returning("greenhouse", make_posting("1")), NOW)
    report = ingest_company(session, company, Broken(), LATER)
    session.commit()
    assert report.failed and "RuntimeError" in report.error
    assert _open_ids(session) == {"1"}


def test_a_changed_posting_updates_the_job_and_its_hashes(session: Session, company: Company) -> None:
    """New description text updates the job and changes its content hash."""
    adapter = FakeAdapter(
        "greenhouse",
        ok(make_posting("1", "Research Scientist", description="v1")),
        ok(make_posting("1", "Research Scientist", description="v2 with RLHF")),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    job = session.scalar(select(Job))
    source = session.scalar(select(JobSource))
    hash_before = job.content_hash
    source_hashes_before = (source.raw_hash, source.content_hash)
    assert source.content_hash == hash_before, "the primary posting's text hash is the job's"

    report = ingest_company(session, company, adapter, LATER)
    session.commit()
    session.expire_all()
    job = session.scalar(select(Job))
    source = session.scalar(select(JobSource))
    assert job.description == "v2 with RLHF"
    assert job.content_hash != hash_before
    assert source.raw_hash != source_hashes_before[0], "the payload changed"
    assert source.content_hash != source_hashes_before[1], "so did the judged text"
    assert report.updated_sources == 1


def test_a_raw_only_change_leaves_the_text_hash_alone(session: Session, company: Company) -> None:
    """A board bumping an internal field changes `raw_hash` but not `content_hash`, so no rescore."""
    words = dict(title="Research Scientist", location="Lisbon, Portugal", description="Same text.")
    adapter = FakeAdapter(
        "greenhouse",
        ok(make_posting("1", raw={"id": "1", "updated_at": "2026-08-01T00:00:00Z"}, **words)),
        ok(make_posting("1", raw={"id": "1", "updated_at": "2026-09-01T00:00:00Z"}, **words)),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    source = session.scalar(select(JobSource))
    before = (source.raw_hash, source.content_hash)

    report = ingest_company(session, company, adapter, LATER)
    session.commit()
    session.expire_all()
    source = session.scalar(select(JobSource))
    assert source.raw_hash != before[0], "the payload did change"
    assert source.content_hash == before[1], "but nothing the judge reads did"
    assert report.updated_sources == 1


def test_locations_are_parsed_into_regions_and_work_mode(session: Session, company: Company) -> None:
    """A posting's location and hints end up parsed onto its job."""
    adapter = FakeAdapter.returning(
        "lever",
        make_posting("1", "RS", source="lever", location="London, United Kingdom",
                     country_code="GB", workplace_type="hybrid"),
        make_posting("2", "RS", source="lever", location="Remote", country_code="PT",
                     workplace_type="remote"),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    jobs = {j.location_raw: j for j in session.scalars(select(Job))}
    assert (jobs["London, United Kingdom"].region, jobs["London, United Kingdom"].work_mode) == ("uk", "hybrid")
    assert (jobs["Remote"].region, jobs["Remote"].regions, jobs["Remote"].work_mode) == ("portugal", ["portugal"], "remote")


def test_the_raw_payload_is_kept(session: Session, company: Company) -> None:
    """The board's untouched JSON is stored alongside the parsed fields."""
    posting = make_posting("1")
    ingest_company(session, company, FakeAdapter.returning("greenhouse", posting), NOW)
    session.commit()
    source = session.scalar(select(JobSource))
    assert source.raw_json == posting.raw
    assert source.url == posting.url


def test_sync_companies_upserts_and_deactivates(session: Session) -> None:
    """A watchlist entry upserts by slug and a removed one is deactivated."""
    first = [
        CompanyEntry(slug="a", name="A", ats="greenhouse", token="a", tier="lab"),
        CompanyEntry(slug="b", name="B", ats="ashby", token="b", tier="infra"),
    ]
    sync_companies(session, first)
    session.commit()
    assert {c.slug: c.active for c in session.scalars(select(Company))} == {"a": True, "b": True}

    second = [CompanyEntry(slug="a", name="A Renamed", ats="lever", token="a2", tier="lab")]
    sync_companies(session, second)
    session.commit()
    session.expire_all()
    rows = {c.slug: c for c in session.scalars(select(Company))}
    assert rows["a"].name == "A Renamed" and rows["a"].ats_type == "lever"
    assert rows["a"].ats_config == {"token": "a2"}
    assert rows["b"].active is False, "removed from the watchlist: deactivated, not deleted"


def test_run_ingest_only_fetches_the_requested_slugs(session: Session) -> None:
    """`only` restricts a run to the named companies and nothing else."""
    watchlist = [
        CompanyEntry(slug="a", name="A", ats="greenhouse", token="a"),
        CompanyEntry(slug="b", name="B", ats="greenhouse", token="b"),
    ]
    adapter = FakeAdapter.returning("greenhouse", make_posting("1"))
    report = run_ingest(adapters={"greenhouse": adapter}, watchlist=watchlist, only=["b"])
    assert [c.slug for c in report.companies] == ["b"]
    assert adapter.calls == 1


def test_an_unsupported_ats_is_reported_not_raised(session: Session) -> None:
    """An ATS type with no adapter is a recorded failure, not a crash."""
    watchlist = [CompanyEntry(slug="nv", name="NVIDIA", ats="workday",
                              ats_config={"tenant": "nv", "wd": "wd5", "site": "X"}, tier="bigtech")]
    report = run_ingest(adapters={}, watchlist=watchlist)
    assert len(report.failures) == 1 and "no adapter" in report.failures[0].error.lower()
    assert session.scalar(select(FetchRun)).status == FetchStatus.FAILED


def test_same_title_siblings_share_a_job_but_keep_their_own_text_hash(
    session: Session, company: Company
) -> None:
    """The canonical key is (company, title, region): two source rows and one job, each with its own text hash."""
    adapter = FakeAdapter.returning(
        "greenhouse",
        make_posting("1", description="Runs research infrastructure programs."),
        make_posting("2", description="Coordinates safety evaluations."),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    assert _counts(session) == (2, 1)
    hashes = {s.content_hash for s in session.scalars(select(JobSource))}
    assert len(hashes) == 2, "each posting is judged on its own text"


def test_posted_at_survives_a_round_trip_through_sqlite(session: Session, company: Company) -> None:
    """SQLite hands back naive datetimes. Comparing them to aware ones must not crash."""
    posted = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    adapter = FakeAdapter.returning("greenhouse", make_posting("1", posted_at=posted))
    ingest_company(session, company, adapter, NOW)
    session.commit()
    session.expire_all()  # forces the next access to reload from SQLite: naive
    ingest_company(session, company, adapter, LATER)  # must not raise
    session.commit()
    job = session.scalar(select(Job))
    assert job.posted_at.replace(tzinfo=UTC) == posted


def test_a_posting_dated_without_an_offset_does_not_crash_the_second_run(
    session: Session, company: Company
) -> None:
    """A board timestamp with no timezone is treated as UTC instead of raising TypeError."""
    naive = parse_iso_datetime("2026-08-01T10:00:00")
    assert naive is not None and naive.tzinfo is not None, "adapters must return aware datetimes"

    adapter = FakeAdapter.returning("greenhouse", make_posting("1", posted_at=naive))
    ingest_company(session, company, adapter, NOW)
    session.commit()
    session.expire_all()
    report = ingest_company(session, company, adapter, LATER)  # must not raise
    session.commit()
    assert not report.failed
    assert as_utc(session.scalar(select(Job)).posted_at) == naive


def test_a_crash_while_reconciling_one_company_does_not_abort_the_others(
    session: Session, monkeypatch
) -> None:
    """Recorded as a failed `fetch_run` for that company. Every other company still runs."""
    watchlist = [
        CompanyEntry(slug="a", name="A", ats="greenhouse", token="a"),
        CompanyEntry(slug="b", name="B", ats="greenhouse", token="b"),
        CompanyEntry(slug="c", name="C", ats="greenhouse", token="c"),
    ]
    real_upsert = ingest_module._upsert_posting

    def exploding_upsert(session, company, posting, now, report):
        """Stands in for `_upsert_posting` and blows up only for company "b"."""
        if company.slug == "b":
            raise ValueError("simulated bug")
        return real_upsert(session, company, posting, now, report)

    monkeypatch.setattr(ingest_module, "_upsert_posting", exploding_upsert)

    class PerCompany:  # ids must differ per company, as they do on real boards
        source = "greenhouse"

        def fetch(self, company):
            """Returns one posting whose id is unique to the fetching company."""
            return ok(make_posting(f"{company.slug}-1"))

    report = run_ingest(adapters={"greenhouse": PerCompany()}, watchlist=watchlist)

    assert [c.slug for c in report.companies] == ["a", "b", "c"]
    assert [c.failed for c in report.companies] == [False, True, False]
    assert "ValueError" in report.companies[1].error
    runs = {r.company_id: r for r in session.scalars(select(FetchRun))}
    b = session.scalar(select(Company).where(Company.slug == "b"))
    assert runs[b.id].status == FetchStatus.FAILED and "simulated bug" in runs[b.id].error
    assert _counts(session) == (2, 2), "a and c were ingested; b's transaction rolled back"


def test_an_id_collision_across_companies_is_skipped_not_reassigned(session: Session) -> None:
    """`(source, source_job_id)` is the identity anchor and is global per source."""
    watchlist = [
        CompanyEntry(slug="a", name="A", ats="greenhouse", token="a"),
        CompanyEntry(slug="c", name="C", ats="greenhouse", token="c"),
    ]
    adapter = FakeAdapter.returning("greenhouse", make_posting("1"))  # same id for both
    report = run_ingest(adapters={"greenhouse": adapter}, watchlist=watchlist)

    assert _counts(session) == (1, 1)
    assert [c.skipped for c in report.companies] == [0, 1]
    owner = session.scalar(select(Company).where(Company.slug == "a"))
    assert session.scalar(select(JobSource)).company_id == owner.id
