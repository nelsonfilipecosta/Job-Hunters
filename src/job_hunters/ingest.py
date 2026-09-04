"""Fetches every watched board and reconciles what it finds into the database.

One run does, per active company:

    1. Record a `fetch_runs` row and call the company's adapter.
    2. If the fetch failed, record why and touch nothing else.
    3. If the fetch succeeded, upsert every posting into `job_sources` (keyed on
       `(source, source_job_id)`, so a second run updates `last_seen` instead of
       inserting again) and attach each to a `jobs` row via `dedup.resolve_job`.
    4. Only if the fetch succeeded and returned at least one posting, mark every
       posting from this source that was not in the response as closed.

Each company is its own transaction. One board failing or one bug in one
adapter must never roll back or abort the others.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import CompanyEntry, load_watchlist
from .db import session_scope
from .dedup import resolve_job
from .models import Company, FetchRun, FetchStatus, Job, JobSource, as_utc, utcnow
from .normalize import content_hash, parse_location, raw_hash
from .sources import get_adapter
from .sources.base import FetchResult, JobSource as JobSourceAdapter, RawPosting

log = logging.getLogger("job_hunters.ingest")


@dataclass
class CompanyReport:
    """What one company's fetch did."""

    slug: str
    source: str
    status: str
    error: str | None = None
    fetched: int = 0
    new_sources: int = 0
    updated_sources: int = 0
    closed: int = 0
    new_jobs: int = 0
    # Postings refused because their (source, id) already belongs to another company.
    skipped: int = 0

    @property
    def failed(self) -> bool:
        """True when the board could not be read, which is what suppresses closing."""
        return self.status == FetchStatus.FAILED


@dataclass
class IngestReport:
    """What a whole run did. One entry per company."""

    companies: list[CompanyReport] = field(default_factory=list)

    @property
    def failures(self) -> list[CompanyReport]:
        """The companies whose fetch failed, which is what sets the exit code."""
        return [c for c in self.companies if c.failed]

    def total(self, attr: str) -> int:
        """Adds one counter (`fetched`, `new_jobs`, ...) across every company."""
        return sum(getattr(c, attr) for c in self.companies)


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


def sync_companies(session: Session, watchlist: list[CompanyEntry]) -> int:
    """Makes the `companies` table mirror the watchlist. Returns how many are active.

    Upserts by slug. A company removed from the watchlist is deactivated, not
    deleted. Its jobs and any application history stay, it just stops being
    fetched.
    """
    wanted = {entry.slug: entry for entry in watchlist}
    existing = {c.slug: c for c in session.scalars(select(Company)).all()}

    for slug, entry in wanted.items():
        values = entry.to_company_kwargs()
        company = existing.get(slug)
        if company is None:
            session.add(Company(**values))
        else:
            for key, value in values.items():
                setattr(company, key, value)
    for slug, company in existing.items():
        if slug not in wanted and company.active:
            company.active = False
            log.info("%s: no longer in the watchlist - deactivated", slug)
    session.flush()
    return sum(1 for e in watchlist if e.active)


# ---------------------------------------------------------------------------
# One company
# ---------------------------------------------------------------------------


def _upsert_posting(
    session: Session, company: Company, posting: RawPosting, now: datetime, report: CompanyReport
) -> None:
    """Stores one posting. Upserts its `job_sources` row and attaches it to a job.

    Every counter on `report` is incremented here. Nothing is deleted or closed.
    That happens in `ingest_company` once the whole board has been read.
    """
    parsed = parse_location(
        posting.location_raw,
        country_code=posting.country_code,
        workplace_type=posting.workplace_type,
        is_remote=posting.is_remote,
        hints=posting.location_hints,
    )
    payload_hash = raw_hash(posting.raw)

    source = session.scalar(
        select(JobSource).where(
            JobSource.source == posting.source,
            JobSource.source_job_id == posting.source_job_id,
        )
    )
    if source is not None and source.company_id not in (None, company.id):
        # The (source, source_job_id) pair already belongs to another company.
        # Greenhouse ids are global integers and Lever/Ashby use UUIDs, so this
        # should never happen. But if it did, reusing the row would move a job
        # from one company to another. Refuse and keep both.
        log.warning("%s: %s:%s already belongs to company id %s; skipped",
                    company.slug, posting.source, posting.source_job_id, source.company_id)
        report.skipped += 1
        return
    if source is None:
        source = JobSource(
            company_id=company.id,
            source=posting.source,
            source_job_id=posting.source_job_id,
            url=posting.url,
            raw_json=posting.raw,
            content_hash=payload_hash,
            first_seen=now,
            last_seen=now,
            is_open=True,
        )
        session.add(source)
        session.flush()
        report.new_sources += 1
    else:
        source.last_seen = now
        source.is_open = True  # a posting that vanished and came back is open again
        if source.content_hash != payload_hash:
            source.raw_json = posting.raw
            source.content_hash = payload_hash
            source.url = posting.url
            report.updated_sources += 1

    current = session.get(Job, source.job_id) if source.job_id is not None else None
    job, created = resolve_job(
        session, company.id, company.name, posting, parsed, now, current=current
    )
    if created:
        report.new_jobs += 1
    source.job_id = job.id
    if job.primary_source_id is None:
        job.primary_source_id = source.id

    # Only the primary sighting is allowed to rewrite the job's displayed fields.
    # Otherwise two boards describing one role would take turns overwriting it.
    if job.primary_source_id == source.id:
        _refresh_job(job, company, posting, parsed, now)
    else:
        job.last_seen = now
        if posting.posted_at and (job.posted_at is None or posting.posted_at < as_utc(job.posted_at)):
            job.posted_at = posting.posted_at


def _refresh_job(
    job: Job, company: Company, posting: RawPosting, parsed, now: datetime
) -> None:
    """Copies the primary posting's fields onto the job, but only if its text changed.

    Guarding on `content_hash` keeps the job untouched when a board returns the same words
    again, so the judge does not pay to rescore an unchanged role.
    """
    job.last_seen = now
    new_hash = content_hash(company.name, posting.title, posting.location_raw, posting.description)
    if job.content_hash != new_hash:
        job.title = posting.title
        job.location_raw = posting.location_raw
        job.region = parsed.region
        job.regions = list(parsed.regions)
        job.work_mode = parsed.work_mode
        job.description = posting.description
        job.apply_url = posting.url
        job.content_hash = new_hash
    if posting.posted_at and (job.posted_at is None or posting.posted_at < as_utc(job.posted_at)):
        job.posted_at = posting.posted_at


def ingest_company(
    session: Session, company: Company, adapter: JobSourceAdapter, now: datetime | None = None
) -> CompanyReport:
    """Fetches one company's board and reconciles it. Records a `fetch_runs` row either way."""
    now = now or utcnow()
    report = CompanyReport(slug=company.slug, source=str(adapter.source), status=FetchStatus.FAILED)
    run = FetchRun(company_id=company.id, source=str(adapter.source), started_at=now, status=FetchStatus.FAILED)
    session.add(run)

    try:
        result: FetchResult = adapter.fetch(company)
    except Exception as exc:
        result = FetchResult.failed(f"Adapter raised {type(exc).__name__}: {exc}")
        log.exception("%s: adapter raised", company.slug)

    run.finished_at = utcnow()
    if not result.succeeded:
        run.error = result.error
        report.error = result.error
        log.warning("%s: fetch failed: %s", company.slug, result.error)
        return report

    run.status = FetchStatus.OK
    run.item_count = len(result.items)
    report.status = FetchStatus.OK
    report.fetched = len(result.items)

    if not result.items:
        log.warning("%s: board returned no postings. Leaving existing ones open.", company.slug)
        return report

    seen: set[str] = set()
    for posting in result.items:
        seen.add(posting.source_job_id)
        _upsert_posting(session, company, posting, now, report)

    stale = session.scalars(
        select(JobSource).where(
            JobSource.company_id == company.id,
            JobSource.source == str(adapter.source),
            JobSource.is_open.is_(True),
            JobSource.source_job_id.not_in(seen),
        )
    ).all()
    for source in stale:
        source.is_open = False
        report.closed += 1
    session.flush()
    return report


# ---------------------------------------------------------------------------
# The whole watchlist
# ---------------------------------------------------------------------------


def run_ingest(
    *,
    adapters: Mapping[str, JobSourceAdapter] | None = None,
    only: Iterable[str] | None = None,
    watchlist: list[CompanyEntry] | None = None,
) -> IngestReport:
    """Syncs the watchlist, then fetches every active company - each in its own transaction.

    `adapters` maps an ATS type to a ready adapter. Anything not supplied is built with `get_adapter`.
    """
    watchlist = watchlist if watchlist is not None else load_watchlist()
    with session_scope() as session:
        sync_companies(session, watchlist)

    with session_scope() as session:
        slugs = list(
            session.scalars(
                select(Company.slug).where(Company.active.is_(True)).order_by(Company.slug)
            )
        )
    wanted = set(only) if only else None
    if wanted is not None:
        slugs = [s for s in slugs if s in wanted]

    cache: dict[str, JobSourceAdapter] = dict(adapters or {})
    report = IngestReport()
    for slug in slugs:
        try:
            with session_scope() as session:
                company = session.scalar(select(Company).where(Company.slug == slug))
                try:
                    adapter = cache.get(company.ats_type) or get_adapter(company.ats_type)
                    cache[company.ats_type] = adapter
                except KeyError as exc:
                    _record_failure(session, company, company.ats_type, str(exc))
                    report.companies.append(CompanyReport(
                        slug=slug, source=company.ats_type, status=FetchStatus.FAILED, error=str(exc)))
                    continue
                report.companies.append(ingest_company(session, company, adapter))
        except Exception as exc:
            error = f"ingest crashed: {type(exc).__name__}: {exc}"
            log.exception("%s: %s", slug, error)
            with session_scope() as session:
                company = session.scalar(select(Company).where(Company.slug == slug))
                _record_failure(session, company, company.ats_type, error)
            report.companies.append(CompanyReport(
                slug=slug, source=company.ats_type, status=FetchStatus.FAILED, error=error))
    return report


def _record_failure(session: Session, company: Company, source: str, error: str) -> None:
    """Writes a failed `fetch_runs` row so a crash is visible instead of silent."""
    session.add(FetchRun(company_id=company.id, source=source, status=FetchStatus.FAILED,
                         error=error, finished_at=utcnow()))
