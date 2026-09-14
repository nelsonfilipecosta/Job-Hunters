"""The scan that turns the discovery sources into a list of companies to review.

One run per enabled source:

    1. Record a `fetch_runs` row and fetch whatever the site lists.
    2. Prefilter every posting with the vocabulary scoring uses, minus
       location and supporting keywords. Location is left out because the
       question here is which companies hire for this work and not where
       this one role sits. And the supporting keywords are left out because
       nobody judges what passes here except the person reading the list.
       An excluded title word drops a posting and only an included title or
       a strong keyword keeps one.
    3. Store each kept posting that is new as a `job_sources` row with no
       company and no job. That row is the memory that stops a posting being
       processed twice. A posting seen before only has its `last_seen` moved.
    4. Name the company. The structured sources carry it as a field. Hacker
       News (HN) is prose, so a small model reads it out with `extract.py`,
       up to a cap per run. As tructured posting that left the field empty
       takes the same route.
    5. Attach the sighting to a candidate. Skip a company already watched,
       merge into an existing candidate by name or by board, or create one
       and probe the three ATS patterns for its board.

Discovery postings never become jobs and never reach the digest. The ATS
posting arrives on the next ingest once the company is watched.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import AppConfig, CompanyEntry, ConfigError, load_all
from .db import session_scope
from .discover import Discovery, best_board, probe, token_from_url
from .extract import ExtractError, Extractor
from .judge import Usage, make_client
from .models import (
    CandidateCompany,
    CandidateStatus,
    Company,
    FetchRun,
    FetchStatus,
    JobSource,
    as_utc,
    utcnow,
)
from .normalize import content_hash, normalize_company, raw_hash
from .scoring import Prefilter
from .sources import get_discovery_source
from .sources.base import DiscoverySource, FetchResult, RawPosting, default_client

log = logging.getLogger("job_hunters.discovery")

# A candidate keeps the distinct roles and the latest sightings it was seen
# with. Enough for a reviewer to place it and no more.
MAX_ROLES = 12
MAX_EVIDENCE = 30

Prober = Callable[[str, tuple[str, ...]], list[Discovery]]

PROBE_TIMEOUT_SECONDS = 15


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class SourceReport:
    """What one source's scan did."""

    source: str
    status: str
    error: str | None = None
    fetched: int = 0
    # Postings the prefilter kept and how many of those were already stored.
    matched: int = 0
    seen_before: int = 0
    new_sightings: int = 0
    # Prose postings the model named, could not name, failed on or that the cap left for next run.
    extracted: int = 0
    unattributed: int = 0
    extraction_failed: int = 0
    capped: int = 0
    # Where the named sightings went.
    new_candidates: int = 0
    seen_again: int = 0
    already_watched: int = 0
    boards_found: int = 0

    @property
    def failed(self) -> bool:
        """True when the site could not be read."""
        return self.status == FetchStatus.FAILED


@dataclass
class DiscoveryReport:
    """What a whole run did. One entry per source."""

    sources: list[SourceReport] = field(default_factory=list)
    cap: int = 0
    dry_run: bool = False
    usage: Usage = field(default_factory=Usage)
    aborted: str | None = None
    reconciled: int = 0
    pending: int = 0

    @property
    def failures(self) -> list[SourceReport]:
        """The sources whose fetch failed which is what sets the exit code."""
        return [s for s in self.sources if s.failed]

    def total(self, attr: str) -> int:
        """Adds one counter (`matched`, `new_candidates`, ...) across every source."""
        return sum(getattr(s, attr) for s in self.sources)


# ---------------------------------------------------------------------------
# What is already watched
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Watched:
    """The companies discovery must not list because they are watched already.

    Matched by name (normalized) and by board. Companies removed from the watchlist
    stay in `companies` deactivated so that the loop does not keep proposing them.
    """

    by_key: Mapping[str, str]
    by_board: Mapping[tuple[str, str], str]

    @classmethod
    def build(cls, watchlist: Iterable[CompanyEntry], companies: Iterable[Company] = ()) -> Watched:
        """Indexes the watchlist and the `companies` table by name and by board."""
        by_key: dict[str, str] = {}
        by_board: dict[tuple[str, str], str] = {}
        for entry in watchlist:
            _index(by_key, by_board, entry.slug, entry.name, entry.ats.value, entry.ats_config)
        for company in companies:
            _index(by_key, by_board, company.slug, company.name, company.ats_type, company.ats_config)
        return cls(by_key, by_board)

    def slug_for(self, key: str, board: tuple[str, str] | None) -> str | None:
        """The watchlist slug this name or board belongs to or None when unwatched."""
        if key in self.by_key:
            return self.by_key[key]
        if board is not None and board in self.by_board:
            return self.by_board[board]
        return None


def _index(by_key: dict, by_board: dict, slug: str, name: str, ats: str, ats_config: dict) -> None:
    """Adds one watched company under its name, its slug and its board."""
    for text in (name, slug):
        key = normalize_company(text)
        if key:
            by_key.setdefault(key, slug)
    token = (ats_config or {}).get("token")
    if token:
        by_board.setdefault((ats, str(token).lower()), slug)


def reconcile(session: Session, watched: Watched, now: datetime) -> int:
    """Marks every pending candidate the watchlist now lists as approved. Returns how many."""
    resolved = 0
    for candidate in session.scalars(
        select(CandidateCompany).where(CandidateCompany.status == CandidateStatus.PENDING)
    ).all():
        board = (
            (candidate.ats_type, candidate.ats_token)
            if candidate.ats_type and candidate.ats_token
            else None
        )
        slug = watched.slug_for(candidate.name_key, board)
        if slug is None:
            continue
        candidate.status = CandidateStatus.APPROVED
        candidate.decided_at = now
        candidate.slug = slug
        resolved += 1
    session.flush()
    return resolved


# ---------------------------------------------------------------------------
# One source
# ---------------------------------------------------------------------------


@dataclass
class _Budget:
    """How many model calls this run may still make and why it may make none."""

    remaining: int
    stopped: str | None = None


class _LazyExtractor:
    """Builds the extractor on first use so a run with no prose never needs a key."""

    def __init__(self, config: AppConfig, extractor: Extractor | None) -> None:
        """Keeps the config to build from or a ready extractor (tests)."""
        self._config = config
        self._extractor = extractor

    def get(self) -> Extractor:
        """The extractor. Built from `ANTHROPIC_API_KEY` if it was not supplied."""
        if self._extractor is None:
            try:
                api_key = self._config.secrets.require("anthropic_api_key")
            except ConfigError as exc:
                raise ExtractError(str(exc), fatal=True) from exc
            self._extractor = Extractor(make_client(api_key), self._config.system.models.extract)
        return self._extractor


def scan_source(
    session: Session,
    adapter: DiscoverySource,
    *,
    prefilter: Prefilter,
    watched: Watched,
    extractor: _LazyExtractor,
    budget: _Budget,
    prober: Prober,
    reprobe_after: timedelta,
    report: DiscoveryReport,
    now: datetime | None = None,
    dry_run: bool = False,
) -> SourceReport:
    """Fetches one source and lists what it finds. Records a `fetch_runs` row unless dry."""
    now = now or utcnow()
    source = str(adapter.source)
    outcome = SourceReport(source=source, status=FetchStatus.FAILED)
    run = FetchRun(company_id=None, source=source, started_at=now, status=FetchStatus.FAILED)
    if not dry_run:
        session.add(run)

    try:
        result: FetchResult = adapter.fetch()
    except Exception as exc:
        result = FetchResult.failed(f"Source raised {type(exc).__name__}: {exc}")
        log.exception("%s: source raised", source)

    run.finished_at = utcnow()
    if not result.succeeded:
        run.error = result.error
        outcome.error = result.error
        log.warning("%s: fetch failed: %s", source, result.error)
        return outcome

    run.status = FetchStatus.OK
    run.item_count = len(result.items)
    outcome.status = FetchStatus.OK
    outcome.fetched = len(result.items)

    for posting in result.items:
        if prefilter.title_exclusion(posting.title) is not None:
            continue
        if prefilter.strong_term(posting.title, posting.description) is None:
            continue
        outcome.matched += 1

        existing = session.scalar(
            select(JobSource).where(
                JobSource.source == posting.source,
                JobSource.source_job_id == posting.source_job_id,
            )
        )
        if existing is not None:
            outcome.seen_before += 1
            if not dry_run:
                existing.last_seen = now
            continue
        outcome.new_sightings += 1
        if dry_run:
            continue

        name, roles, careers_url = posting.company_name, [posting.title], None
        if name is None:
            if budget.stopped is not None or budget.remaining <= 0:
                outcome.capped += 1
                continue
            try:
                extraction, usage = extractor.get().extract(posting.description)
            except ExtractError as exc:
                outcome.extraction_failed += 1
                log.warning("%s:%s: %s", source, posting.source_job_id, exc)
                if exc.fatal:
                    budget.stopped = str(exc)
                    report.aborted = str(exc)
                    break
                continue
            budget.remaining -= 1
            report.usage = report.usage + usage
            outcome.extracted += 1
            name, roles, careers_url = extraction.company, extraction.roles, extraction.careers_url

        _store_sighting(session, posting, name, now)
        if name is None or not normalize_company(name):
            outcome.unattributed += 1
        else:
            _attach(
                session, posting, name, roles, careers_url,
                watched=watched, prober=prober, reprobe_after=reprobe_after, now=now, report=outcome,
            )
        session.commit()
    return outcome


def _store_sighting(session: Session, posting: RawPosting, company_name: str | None, now: datetime) -> JobSource:
    """Records one kept posting as a sighting: a `job_sources` row with no company and no job."""
    row = JobSource(
        company_id=None,
        job_id=None,
        source=posting.source,
        source_job_id=posting.source_job_id,
        url=posting.url,
        raw_json=posting.raw,
        raw_hash=raw_hash(posting.raw),
        content_hash=content_hash(
            company_name or "", posting.title, posting.location_raw, posting.description
        ),
        first_seen=now,
        last_seen=now,
        is_open=True,
    )
    session.add(row)
    session.flush()
    return row


def _attach(
    session: Session,
    posting: RawPosting,
    name: str,
    roles: list[str],
    careers_url: str | None,
    *,
    watched: Watched,
    prober: Prober,
    reprobe_after: timedelta,
    now: datetime,
    report: SourceReport,
) -> None:
    """Puts one named sighting on the candidate it belongs to, creating and probing one if needed."""
    name = name.strip()[:200]
    key = normalize_company(name)
    if key in watched.by_key:
        report.already_watched += 1
        return

    candidate = session.scalar(select(CandidateCompany).where(CandidateCompany.name_key == key))
    if candidate is not None and candidate.status == CandidateStatus.APPROVED:
        # Approved means in the watchlist. Possibly under a name the file spells differently.
        report.already_watched += 1
        return
    if candidate is None:
        board = _resolve(name, careers_url, prober)
        if board is not None and (board.ats, board.token) in watched.by_board:
            report.already_watched += 1
            return
        if board is not None:
            # The same board under another spelling of the name is the same company.
            candidate = session.scalar(
                select(CandidateCompany).where(
                    CandidateCompany.ats_type == board.ats,
                    CandidateCompany.ats_token == board.token,
                )
            )
        if candidate is None:
            candidate = CandidateCompany(
                name=name, name_key=key, sightings=0, roles=[], evidence=[],
                first_seen=now, last_seen=now,
            )
            session.add(candidate)
            report.new_candidates += 1
        else:
            report.seen_again += 1
        _set_board(candidate, board, now)
        if board is not None:
            report.boards_found += 1
    else:
        report.seen_again += 1
        stale = candidate.probed_at is None or now - as_utc(candidate.probed_at) >= reprobe_after
        if candidate.ats_type is None and stale:
            board = _resolve(name, careers_url, prober)
            _set_board(candidate, board, now)
            if board is not None:
                report.boards_found += 1

    _note_sighting(candidate, posting, roles, careers_url, now)
    session.flush()


def _resolve(name: str, careers_url: str | None, prober: Prober) -> Discovery | None:
    """The board to watch for this company if any of the three ATS patterns answers."""
    hint = token_from_url(careers_url)
    hits = prober(name, (hint[1],) if hint else ())
    return best_board(hits, hint)


def _set_board(candidate: CandidateCompany, board: Discovery | None, now: datetime) -> None:
    """Records what the probe found or that it found nothing, as well as when it looked."""
    candidate.probed_at = now
    if board is None:
        return
    candidate.ats_type = board.ats
    candidate.ats_token = board.token
    candidate.board_url = board.url
    candidate.board_jobs = board.job_count


def _note_sighting(
    candidate: CandidateCompany,
    posting: RawPosting,
    roles: list[str],
    careers_url: str | None,
    now: datetime,
) -> None:
    """Adds one sighting to a candidate. Lists are reassigned and not appended to, so SQLAlchemy sees the change."""
    candidate.sightings = (candidate.sightings or 0) + 1
    candidate.last_seen = now
    if careers_url and not candidate.careers_url:
        candidate.careers_url = careers_url
    merged = list(candidate.roles or [])
    for role in roles:
        role = (role or "").strip()
        if role and role not in merged:
            merged.append(role)
    candidate.roles = merged[:MAX_ROLES]
    evidence = list(candidate.evidence or [])
    evidence.append({
        "source": str(posting.source),
        "source_job_id": posting.source_job_id,
        "title": posting.title[:200],
        "url": posting.url,
        "seen": now.isoformat(),
    })
    candidate.evidence = evidence[-MAX_EVIDENCE:]


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


def run_discovery(
    *,
    sources: Mapping[str, DiscoverySource] | None = None,
    only: Iterable[str] | None = None,
    dry_run: bool = False,
    extractor: Extractor | None = None,
    prober: Prober | None = None,
    config: AppConfig | None = None,
    now: datetime | None = None,
) -> DiscoveryReport:
    """Scans every enabled source, each in its own transaction. What `job-hunters scan` runs.

    `sources` maps a source name to a ready adapter. Anything not supplied is built
    with `get_discovery_source`. `only` narrows the enabled sources and never switches
    on one the config has off. `prober` stands in for `discover.probe` (tests) and the
    real one shares a single HTTP client across every company the run probes. A dry
    run fetches and prefilters but calls no model, probes no board and writes nothing.
    """
    config = config or load_all()
    settings = config.system.discovery
    now = now or utcnow()
    names = settings.sources.enabled()
    if only is not None:
        wanted = set(only)
        names = [name for name in names if name in wanted]

    report = DiscoveryReport(cap=settings.max_extractions_per_run, dry_run=dry_run)
    prefilter = Prefilter(config.search_profile)
    budget = _Budget(remaining=settings.max_extractions_per_run)
    lazy = _LazyExtractor(config, extractor)
    reprobe_after = timedelta(days=settings.reprobe_after_days)
    cache: dict[str, DiscoverySource] = dict(sources or {})

    with session_scope() as session:
        # Two views of "watched". The file alone decides what counts as approved.
        # The file plus the table decides what is never proposed (a company removed
        # from the file stays in the table deactivated on purpose).
        listed = Watched.build(config.watchlist)
        watched = Watched.build(config.watchlist, session.scalars(select(Company)).all())
        if not dry_run:
            report.reconciled = reconcile(session, listed, now)

    with default_client(timeout=PROBE_TIMEOUT_SECONDS) as client:
        if prober is None:
            def prober(name: str, hints: tuple[str, ...]) -> list[Discovery]:
                """The real probe with the tokens a careers link named tried first."""
                return probe(name, client, hints=hints)

        for name in names:
            try:
                with session_scope() as session:
                    adapter = cache.get(name) or get_discovery_source(name)
                    cache[name] = adapter
                    report.sources.append(scan_source(
                        session, adapter, prefilter=prefilter, watched=watched, extractor=lazy,
                        budget=budget, prober=prober, reprobe_after=reprobe_after, report=report,
                        now=now, dry_run=dry_run,
                    ))
            except Exception as exc:
                error = f"scan crashed: {type(exc).__name__}: {exc}"
                log.exception("%s: %s", name, error)
                if not dry_run:
                    with session_scope() as session:
                        session.add(FetchRun(company_id=None, source=name, status=FetchStatus.FAILED,
                                             error=error, finished_at=utcnow()))
                report.sources.append(SourceReport(source=name, status=FetchStatus.FAILED, error=error))

    with session_scope() as session:
        report.pending = session.scalar(
            select(func.count()).select_from(CandidateCompany)
            .where(CandidateCompany.status == CandidateStatus.PENDING)
        ) or 0
    return report
