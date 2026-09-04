"""SQLAlchemy definitions of the eight database tables.

This module is purely declarative. It describes the tables, their columns and
the sets of values their text columns may hold. It opens no connection and runs
no query. `db.py` builds the engine and `Base.metadata.create_all()` turns
these classes into real tables.

    companies           organizations whose job boards are polled
    job_sources         postings as each board returned them
    jobs                deduplicated openings: one row per real opening
    scores              LLM judgements of how well a job matches the search
    applications        current state of every job applied to
    application_events  dated history behind each application
    digest_appearances  which jobs were included in which digest email
    fetch_runs          the outcome of every attempt to fetch a board

Two of those tables hold postings, which is deliberate. `job_sources` keeps one
row per board a posting appeared on. `jobs` keeps one row per real opening after
deduplication. Separating them is what lets the same board be fetched repeatedly
without creating duplicates and one opening listed on three boards collapse
into a single entry.

The StrEnum classes below (AtsType, WorkMode and the rest) list the allowed
values for individual columns. They are Python constants, not database CHECK
constraints: SQLite cannot alter a constraint in place, so a database-level enum
would mean rebuilding the table whenever a later phase adds a value.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship
)


def utcnow() -> datetime:
    """Current time in UTC.

    Everything is stored in UTC and converted only for display. SQLite has no
    native timestamp type and does not preserve the offset, so values read back
    are naive - treat them as UTC.
    """
    return datetime.now(UTC)


def as_utc(value: datetime | None) -> datetime | None:
    """Makes a datetime comparable regardless of where it came from.

    Values read back from SQLite are naive (see `utcnow`), while values built
    in code are aware. Python refuses to compare the two, so anything that
    compares timestamps calls this first and treats naive as UTC.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------


class AtsType(StrEnum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"


class Tier(StrEnum):
    LAB = "lab"
    BIGTECH = "bigtech"
    INFRA = "infra"
    DISCOVERED = "discovered"  # added by the promotion loop


class SourceKind(StrEnum):
    """Where a sighting came from. ATS boards and aggregators share one table."""

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    HN = "hn"
    REMOTEOK = "remoteok"
    ARBEITNOW = "arbeitnow"
    REMOTIVE = "remotive"


class WorkMode(StrEnum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"
    UNKNOWN = "unknown"


class LocationFit(StrEnum):
    PRIORITY = "priority"
    ACCEPTABLE = "acceptable"
    UNKNOWN = "unknown"
    EXCLUDED = "excluded"


class WorkAuthStatus(StrEnum):
    """The judge's *contradiction detection* only.

    `blocked` means the description explicitly conflicts with declared status
    ("must hold US citizenship"), not that the model reasoned about immigration.
    """

    ELIGIBLE = "eligible"
    UNCLEAR = "unclear"
    BLOCKED = "blocked"


class ApplicationStatus(StrEnum):
    INTERESTED = "interested"
    APPLIED = "applied"
    IN_PROCESS = "in_process"
    REJECTED = "rejected"
    OFFER = "offer"
    WITHDRAWN = "withdrawn"
    GHOSTED = "ghosted"
    DISMISSED = "dismissed"


class EventKind(StrEnum):
    """Append-only timeline entries. Order here is the usual progression."""

    APPLIED = "applied"
    RECRUITER_SCREEN = "recruiter_screen"
    TECHNICAL = "technical"
    ONSITE = "onsite"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    GHOSTED = "ghosted"
    NOTE = "note"


class FetchStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class DigestSection(StrEnum):
    PRIORITY = "priority"
    REMOTE = "remote"
    WORTH_CHECKING = "worth_checking"
    FOLLOW_UP = "follow_up"
    STILL_OPEN = "still_open"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class Company(Base):
    """Organizations whose job boards are polled.

    Each row shows an organization whose job board is polled for openings. It
    holds the connection details its applicant tracking system needs to be
    fetched. Mirrors one entry in `config/companies_watchlist.yaml`.
    """

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    ats_type: Mapped[str] = mapped_column(String(20))

    # Adapter-specific connection details. Greenhouse/Lever/Ashby need only
    # {"token": "..."}; Workday needs {"tenant", "wd", "site"}. A JSON column
    # keeps one adapter's requirements from leaking into the others' columns.
    ats_config: Mapped[dict] = mapped_column(JSON, default=dict)

    tier: Mapped[str] = mapped_column(String(20), default=Tier.DISCOVERED)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job_sources: Mapped[list[JobSource]] = relationship(back_populates="company")
    jobs: Mapped[list[Job]] = relationship(back_populates="company")

    def __repr__(self) -> str:
        return f"<Company {self.slug} ({self.ats_type})>"


class JobSource(Base):
    """Postings as each board returned them.

    Each row shows one posting as it appeared on one job board, stored exactly
    as that board returned it. Several rows here can describe the same
    real-world opening, which is why the deduplicated version lives in `jobs`.
    """

    __tablename__ = "job_sources"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Null for aggregator sightings (HN, RemoteOK), which have no watchlist company.
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"), index=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), index=True)

    source: Mapped[str] = mapped_column(String(20))
    source_job_id: Mapped[str] = mapped_column(String(200))
    url: Mapped[str | None] = mapped_column(Text)

    # The untouched payload. Keeping it means a normalization bug can be fixed
    # and replayed without re-fetching every board.
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)

    # Hash of the *raw* payload. It answers "did the source change?" and is
    # distinct from Job.content_hash, which answers "must we rescore?".
    content_hash: Mapped[str] = mapped_column(String(64), index=True)

    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Set false only when a fetch succeeded and did not contain this posting.
    # See critical correctness rule in plan section 3.4.
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    company: Mapped[Company | None] = relationship(back_populates="job_sources")
    job: Mapped[Job | None] = relationship(
        back_populates="sources", foreign_keys=[job_id]
    )

    # Avoid a duplicate sighting when a second fetch returns the same posting.
    # Keeping this combination of columns unique is what makes re-ingest idempotent.
    __table_args__ = (
        UniqueConstraint("source", "source_job_id", name="uq_job_sources_identity"),
    )

    def __repr__(self) -> str:
        return f"<JobSource {self.source}:{self.source_job_id}>"


class Job(Base):
    """Deduplicated job openings.

    Each row shows one real-world job opening, deduplicated across every board
    it appeared on. This is the table that the scoring, the digest and the
    application tracking all read from.
    """

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)

    # sha256(normalized company | normalized title | region) - the deterministic
    # first pass of deduplication. See plan section 3.5.
    canonical_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    title: Mapped[str] = mapped_column(String(500))

    location_raw: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str] = mapped_column(String(30), default="unknown", index=True)
    # Every country token the location string resolved to. `region` above is the
    # first of them or "other"/"unknown" when none did.
    regions: Mapped[list] = mapped_column(JSON, default=list)
    work_mode: Mapped[str] = mapped_column(String(20), default=WorkMode.UNKNOWN)

    description: Mapped[str | None] = mapped_column(Text)
    apply_url: Mapped[str | None] = mapped_column(Text)

    # Hash of everything the judge reads: title, company, location and description.
    # Computed over normalized text, with whitespace collapsed and boilerplate
    # stripped, so a recruiter fixing a typo does not trigger a rescore. See plan
    # section 3.6.
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    # Which sighting is canonical for display and apply_url. Deliberately not a
    # foreign key: jobs and job_sources reference each other and SQLite cannot
    # add a constraint after the fact, so declaring both directions would make
    # create_all unorderable. The real relation lives on JobSource.job_id.
    primary_source_id: Mapped[int | None] = mapped_column(Integer)

    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    company: Mapped[Company] = relationship(back_populates="jobs")
    sources: Mapped[list[JobSource]] = relationship(
        back_populates="job", foreign_keys=[JobSource.job_id]
    )
    scores: Mapped[list[Score]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    application: Mapped[Application | None] = relationship(
        back_populates="job", uselist=False
    )

    def __repr__(self) -> str:
        return f"<Job {self.id} {self.title!r}>"


class Score(Base):
    """LLM judgements of how well a job matches the search.

    Each row shows one judgement of one job against the search profile, kept as
    a 0-100 score plus a short written explanation. A job is judged again only
    when its text changes or the scoring prompt is revised.
    """

    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)

    score: Mapped[int] = mapped_column(Integer, index=True)
    summary: Mapped[str | None] = mapped_column(Text)
    rationale: Mapped[str | None] = mapped_column(Text)
    matched_areas: Mapped[list] = mapped_column(JSON, default=list)
    concerns: Mapped[list] = mapped_column(JSON, default=list)

    work_authorization: Mapped[str] = mapped_column(String(20), default=WorkAuthStatus.UNCLEAR)
    # Computed in code and stored alongside the judgement for convenience.
    location_fit: Mapped[str] = mapped_column(String(20), default=LocationFit.UNKNOWN)

    # Which version of the prompt produced the judgement.
    prompt_version: Mapped[int] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(60))
    content_hash: Mapped[str] = mapped_column(String(64))
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[Job] = relationship(back_populates="scores")

    # Avoid a duplicate judgement when a second scoring run returns the same result.
    # The unique constraint enforces 'score once per (job_id, content_hash, prompt_version)'
    # in the database to avoid accidental duplicates. See plan section 3.6.
    __table_args__ = (
        UniqueConstraint(
            "job_id", "content_hash", "prompt_version", name="uq_scores_once_per_version"
        ),
    )

    def __repr__(self) -> str:
        return f"<Score job={self.job_id} {self.score} v{self.prompt_version}>"


class Application(Base):
    """Current state of every job applied to.

    Each row shows a job acted on and where it currently stands (applied,
    interviewing or rejected) and the paths to any CV or cover letter generated
    for it. At most one row per job. The application history lives in
    `application_events`.
    """

    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), unique=True)

    status: Mapped[str] = mapped_column(
        String(20), default=ApplicationStatus.INTERESTED, index=True
    )
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cv_path: Mapped[str | None] = mapped_column(Text)
    cover_letter_path: Mapped[str | None] = mapped_column(Text)
    last_followup_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[Job] = relationship(back_populates="application")
    events: Mapped[list[ApplicationEvent]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="ApplicationEvent.occurred_at",
    )

    def __repr__(self) -> str:
        return f"<Application job={self.job_id} {self.status}>"


class ApplicationEvent(Base):
    """Dated history behind each application.

    Each row shows one dated step in an application history: applying, a
    recruiter screen, an interview, an offer or a rejection. Rows are only ever
    added and never changed, so the order and timing of a hiring process are
    preserved instead of being overwritten.
    """

    __tablename__ = "application_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id"), index=True
    )
    event: Mapped[str] = mapped_column(String(30))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    notes: Mapped[str | None] = mapped_column(Text)

    application: Mapped[Application] = relationship(back_populates="events")

    def __repr__(self) -> str:
        return f"<Event {self.event} app={self.application_id}>"


class DigestAppearance(Base):
    """Which jobs were included in which digest email.

    Each row shows that a job was included in one day's digest email and where
    in it. Counting these is what stops a job you neither applied to nor
    dismissed from reappearing in every digest forever.
    """

    __tablename__ = "digest_appearances"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    digest_date: Mapped[date] = mapped_column(Date, index=True)
    section: Mapped[str] = mapped_column(String(30))
    position: Mapped[int] = mapped_column(Integer)

    # The score at the time it was shown, so a materially changed score can
    # reset the appearance counter (`reset_on_score_delta`).
    score_at_appearance: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("job_id", "digest_date", name="uq_digest_once_per_day"),
    )

    def __repr__(self) -> str:
        return f"<Appearance job={self.job_id} {self.digest_date}>"


class FetchRun(Base):
    """The outcome of every attempt to fetch a board.

    Each row shows one attempt to fetch one job board and how it went. Without
    this record a failed fetch looks identical to a board with no open roles -
    and would wrongly mark every job at that company as closed.
    """

    __tablename__ = "fetch_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"), index=True)
    source: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), index=True)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_fetch_runs_company_started", "company_id", "started_at"),)

    def __repr__(self) -> str:
        return f"<FetchRun {self.source} {self.status} n={self.item_count}>"
