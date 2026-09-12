"""What an action link does and what the dashboard tracks.

Two halves of one subject. `perform` is the only place in this project that
writes an `applications` row and the dashboard functions below are the only
place that reads the tracking tables in aggregate. Keeping them together so
that the rules for what a status means are written once.

Every action is idempotent. A link sits in an inbox for weeks and gets clicked
twice, forwarded or opened again from a search months later. So `perform` decides
from the state it finds rather than from what the link says. Clicking "Applied"
on a job already applied to changes nothing and says so, rather than writing a
second row or raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .actions import Action
from .config import AppConfig, RepeatSuppressionConfig, SearchProfile
from .digest import ACTIONED_STATUSES, appearances_since_change, suppression_state
from .models import (
    Application,
    ApplicationEvent,
    ApplicationStatus,
    Company,
    DigestAppearance,
    EventKind,
    Job,
    JobSource,
    Score,
    as_utc,
    utcnow,
)
from .scoring import best_scores

# Every status that means an application was actually sent. `interested` is the
# row a job gets before anything happens to it and `dismissed` is the row that
# says it never will be, so both are outside this set. Derived rather than
# listed, so a status added later has to be placed deliberately.
APPLIED_STATUSES: frozenset[str] = frozenset(
    set(ApplicationStatus)
    - {ApplicationStatus.INTERESTED, ApplicationStatus.DISMISSED}
)

# Actions whose code arrives in Phase 6. Listed rather than tested for by name
# so that removing one from here is all it takes to switch it on.
DEFERRED_ACTIONS: frozenset[str] = frozenset(
    {Action.DRAFT_CV, Action.DRAFT_COVER_LETTER}
)

# The stages an application passes through, in order, for the funnel.
FUNNEL_STAGES: tuple[str, ...] = (
    EventKind.APPLIED, EventKind.RECRUITER_SCREEN, EventKind.TECHNICAL,
    EventKind.ONSITE, EventKind.OFFER,
)

# What counts as somebody at the company answering.
RESPONSE_EVENTS: frozenset[str] = frozenset(
    {EventKind.RECRUITER_SCREEN, EventKind.TECHNICAL, EventKind.ONSITE,
     EventKind.OFFER, EventKind.REJECTED}
)


class UnknownJob(Exception):
    """A signed link naming a job that is no longer in the database."""


# ---------------------------------------------------------------------------
# What one page shows about one job
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobCard:
    """The few facts a confirmation page shows about a job."""

    job_id: int
    title: str
    company: str
    location: str
    work_mode: str
    apply_url: str | None
    score: int | None
    summary: str
    status: str | None
    applied_at: datetime | None

    @property
    def is_settled(self) -> bool:
        """True once the job has been acted on and left the digest."""
        return self.status is not None and self.status != ApplicationStatus.INTERESTED


def job_card(session: Session, job_id: int, prompt_version: int) -> JobCard:
    """Everything a page needs about one job or `UnknownJob` if it is gone."""
    row = session.execute(
        select(Job, Company.name)
        .join(Company, Job.company_id == Company.id)
        .where(Job.id == job_id)
    ).first()
    if row is None:
        raise UnknownJob(f"No job {job_id} in this database.")
    job, company_name = row
    score = best_score_for(session, job_id, prompt_version)
    application = _application_for(session, job_id)
    return JobCard(
        job_id=job.id,
        title=job.title,
        company=company_name,
        location=job.location_raw or "not stated",
        work_mode=job.work_mode,
        apply_url=_winning_url(score, job),
        score=score.score if score is not None else None,
        summary=(score.summary or "").strip() if score is not None else "",
        status=application.status if application is not None else None,
        applied_at=as_utc(application.applied_at) if application is not None else None,
    )


def _winning_url(score: Score | None, job: Job) -> str | None:
    """The posting the digest linked to, which is the one whose text was judged.

    A job can have several open postings and the one displayed is not always the
    one that scored best. The email deliberately links to the winner, so the page
    that email opens has to link to the same posting or the summary beside it
    would describe text that link does not show.
    """
    if score is None or score.source is None:
        return job.apply_url
    return score.source.url or job.apply_url


def best_score_for(session: Session, job_id: int, prompt_version: int) -> Score | None:
    """The best score for one job or `None` if it has never been scored."""
    return session.scalars(
        select(Score)
        .join(JobSource, Score.source_id == JobSource.id)
        .where(
            JobSource.job_id == job_id,
            JobSource.is_open.is_(True),
            Score.prompt_version == prompt_version,
            Score.content_hash == JobSource.content_hash,
        )
        .order_by(Score.score.desc(), Score.id)
    ).first()


# ---------------------------------------------------------------------------
# Performing an action
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """What one confirmed action did, in the words the page will use."""

    changed: bool
    headline: str
    detail: str
    status: str | None = None


def can_confirm(action: Action, card: JobCard) -> tuple[bool, str]:
    """Whether the confirm page offers a button and the sentence explaining it either way."""
    if action in DEFERRED_ACTIONS:
        return False, (
            "Drafting arrives in Phase 6. Nothing has been queued and this page "
            "would rather say so than pretend."
        )
    if action is Action.APPLIED:
        if card.status in APPLIED_STATUSES:
            return False, f"Already recorded as {card.status}. There is nothing left to do."
        return True, "Records the application and starts its timeline."
    if card.status == ApplicationStatus.DISMISSED:
        return False, "Already dismissed. It is not in the digest."
    if card.status in APPLIED_STATUSES:
        return False, (
            f"This job is recorded as {card.status}. Dismissing it would discard that "
            f"and it is already out of the digest, so this link does nothing here."
        )
    return True, "Takes it out of the digest for good. Nothing is deleted."


def perform(
    session: Session, action: Action, job_id: int, *, now: datetime | None = None
) -> Outcome:
    """Carries out one confirmed action on one job. Safe to call twice."""
    now = now or utcnow()
    if action in DEFERRED_ACTIONS:
        return Outcome(
            changed=False,
            headline="Not built yet",
            detail=(
                "Drafting a CV or a cover letter is Phase 6. Nothing was queued and "
                "nothing was written."
            ),
        )
    application = _application_for(session, job_id)
    if action is Action.APPLIED:
        return _mark_applied(session, application, job_id, now)
    return _dismiss(session, application, job_id)


def _mark_applied(
    session: Session, application: Application | None, job_id: int, now: datetime
) -> Outcome:
    """Records the application and opens its timeline or reports that it already exists."""
    if application is not None and application.status in APPLIED_STATUSES:
        when = as_utc(application.applied_at)
        on_date = f" on {when:%d %B %Y}" if when is not None else ""
        return Outcome(
            changed=False,
            headline="Already recorded",
            detail=(
                f"This job was already marked {application.status}{on_date}. "
                f"Clicking the link again changed nothing."
            ),
            status=application.status,
        )
    if application is None:
        application = Application(job_id=job_id)
        session.add(application)
    application.status = ApplicationStatus.APPLIED
    application.applied_at = now
    session.flush()
    if not _has_event(session, application.id, EventKind.APPLIED):
        session.add(
            ApplicationEvent(
                application_id=application.id, event=EventKind.APPLIED, occurred_at=now
            )
        )
    return Outcome(
        changed=True,
        headline="Marked as applied",
        detail=(
            "It will not appear in another digest. Add what happens next from the "
            "dashboard as it happens."
        ),
        status=ApplicationStatus.APPLIED,
    )


def _dismiss(session: Session, application: Application | None, job_id: int) -> Outcome:
    """Takes a job out of the digest, unless it was applied to."""
    if application is not None and application.status in APPLIED_STATUSES:
        return Outcome(
            changed=False,
            headline="Nothing was changed",
            detail=(
                f"This job is recorded as {application.status}. Dismissing it would "
                f"discard that record and it is already out of the digest, so the "
                f"dismissal was refused rather than carried out."
            ),
            status=application.status,
        )
    if application is not None and application.status == ApplicationStatus.DISMISSED:
        return Outcome(
            changed=False,
            headline="Already dismissed",
            detail="Clicking the link again changed nothing.",
            status=ApplicationStatus.DISMISSED,
        )
    if application is None:
        application = Application(job_id=job_id)
        session.add(application)
    application.status = ApplicationStatus.DISMISSED
    return Outcome(
        changed=True,
        headline="Dismissed",
        detail=(
            "It will not appear in another digest. Nothing was deleted. The job, its "
            "postings and its scores are all still there. An 'Applied' link for the "
            "same job still works if you change your mind."
        ),
        status=ApplicationStatus.DISMISSED,
    )


def _application_for(session: Session, job_id: int) -> Application | None:
    """This job's application row, of which there can only ever be one."""
    return session.scalar(select(Application).where(Application.job_id == job_id))


def _has_event(session: Session, application_id: int, event: str) -> bool:
    """Whether this application already carries an event of this kind."""
    return session.scalar(
        select(ApplicationEvent.id)
        .where(
            ApplicationEvent.application_id == application_id,
            ApplicationEvent.event == event,
        )
        .limit(1)
    ) is not None


# ---------------------------------------------------------------------------
# The dashboard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineEvent:
    """One dated step of one application as the timeline prints it."""

    event: str
    occurred_at: datetime
    notes: str | None

    @property
    def label(self) -> str:
        """The event name in the words a person uses for it."""
        return self.event.replace("_", " ")


@dataclass(frozen=True)
class TrackedApplication:
    """One row of the pipeline with the timeline behind it."""

    job_id: int
    title: str
    company: str
    status: str
    applied_at: datetime | None
    created_at: datetime
    source: str
    tailored: bool
    apply_url: str | None
    events: tuple[TimelineEvent, ...] = ()

    @property
    def sent(self) -> bool:
        """Whether an application was actually made, regardless of how it came to be recorded.

        `perform` writes the date and the first event together, so for anything
        this project created the two agree. Accepting either keeps the funnel
        and the response rate counting the same set of applications when a row
        is edited by hand, which is the only way one of the two can go missing.
        """
        return self.applied_at is not None or any(
            event.event == EventKind.APPLIED for event in self.events
        )

    @property
    def responded(self) -> bool:
        """Whether anyone at the company ever answered."""
        return any(event.event in RESPONSE_EVENTS for event in self.events)

    @property
    def last_event(self) -> TimelineEvent | None:
        """The most recent step, which is usually the only one worth showing folded."""
        return self.events[-1] if self.events else None


@dataclass(frozen=True)
class Stage:
    """One step of the funnel and how many applications ever reached it."""

    name: str
    count: int
    share_of_applied: float

    @property
    def label(self) -> str:
        """The stage name in the words a person uses for it."""
        return self.name.replace("_", " ")


@dataclass(frozen=True)
class Rate:
    """How often one slice of the applications got an answer."""

    label: str
    applied: int
    responded: int

    @property
    def rate(self) -> float:
        """The share that got an answer or zero when nothing was sent yet."""
        return self.responded / self.applied if self.applied else 0.0


@dataclass(frozen=True)
class OpenRole:
    """One job above the threshold that has been neither applied to nor dismissed."""

    job_id: int
    title: str
    company: str
    location: str
    work_mode: str
    score: int
    apply_url: str | None
    appearances: int
    suppressed: bool = False


@dataclass(frozen=True)
class Dashboard:
    """Everything the dashboard page shows (already counted)."""

    generated_at: datetime
    threshold: int
    pipeline: dict[str, int] = field(default_factory=dict)
    applications: tuple[TrackedApplication, ...] = ()
    dismissed: int = 0
    funnel: tuple[Stage, ...] = ()
    by_source: tuple[Rate, ...] = ()
    by_tailoring: tuple[Rate, ...] = ()
    open_roles: tuple[OpenRole, ...] = ()
    open_total: int = 0

    @property
    def open_suppressed(self) -> int:
        """How many of the listed roles the digest has stopped showing."""
        return sum(1 for role in self.open_roles if role.suppressed)

    @property
    def applied_total(self) -> int:
        """How many applications have actually been sent."""
        return sum(1 for a in self.applications if a.sent)

    @property
    def response_rate(self) -> float:
        """The share of sent applications that got an answer."""
        answered = sum(1 for a in self.applications if a.sent and a.responded)
        return answered / self.applied_total if self.applied_total else 0.0

    @property
    def is_empty(self) -> bool:
        """True before the first link has ever been clicked."""
        return not self.applications and self.dismissed == 0


def build_dashboard(
    session: Session,
    config: AppConfig,
    *,
    now: datetime | None = None,
    open_limit: int = 200,
) -> Dashboard:
    """Counts the pipeline, the funnel and the response rates in one pass over the tables."""
    now = now or utcnow()
    profile = config.search_profile
    tracked = _tracked_applications(session)
    active = tuple(a for a in tracked if a.status != ApplicationStatus.DISMISSED)
    dismissed = sum(1 for a in tracked if a.status == ApplicationStatus.DISMISSED)
    open_roles, open_total = _open_roles(
        session, profile, config.system.digest.repeat_suppression, limit=open_limit
    )
    return Dashboard(
        generated_at=now,
        threshold=profile.scoring.threshold,
        pipeline=_pipeline(active),
        applications=active,
        dismissed=dismissed,
        funnel=_funnel(active),
        by_source=_rates(active, lambda a: a.source),
        by_tailoring=_rates(active, lambda a: "tailored" if a.tailored else "not tailored"),
        open_roles=open_roles,
        open_total=open_total,
    )


def _tracked_applications(session: Session) -> tuple[TrackedApplication, ...]:
    """Every application with its job, its board and its timeline (newest first)."""
    rows = session.execute(
        select(Application, Job, Company.name)
        .join(Job, Application.job_id == Job.id)
        .join(Company, Job.company_id == Company.id)
        .order_by(Application.created_at.desc(), Application.id.desc())
    ).all()
    sources = _primary_sources(session, [job.id for _, job, _ in rows])
    return tuple(
        TrackedApplication(
            job_id=job.id,
            title=job.title,
            company=company_name,
            status=application.status,
            applied_at=as_utc(application.applied_at),
            created_at=as_utc(application.created_at),
            source=sources.get(job.id, "unknown"),
            tailored=bool(application.cv_path or application.cover_letter_path),
            apply_url=job.apply_url,
            events=tuple(
                TimelineEvent(e.event, as_utc(e.occurred_at), e.notes)
                for e in application.events
            ),
        )
        for application, job, company_name in rows
    )


def _primary_sources(session: Session, job_ids: list[int]) -> dict[int, str]:
    """Which board each job is displayed from (keyed by job)."""
    if not job_ids:
        return {}
    rows = session.execute(
        select(JobSource.job_id, JobSource.id, JobSource.source, Job.primary_source_id)
        .join(Job, JobSource.job_id == Job.id)
        .where(JobSource.job_id.in_(job_ids))
        .order_by(JobSource.id)
    ).all()
    sources: dict[int, str] = {}
    for job_id, source_id, source, primary_id in rows:
        if source_id == primary_id or job_id not in sources:
            sources[job_id] = source
    return sources


def _pipeline(applications: tuple[TrackedApplication, ...]) -> dict[str, int]:
    """How many applications sit at each status, in the order a search progresses."""
    order = [
        ApplicationStatus.INTERESTED, ApplicationStatus.APPLIED,
        ApplicationStatus.IN_PROCESS, ApplicationStatus.OFFER,
        ApplicationStatus.REJECTED, ApplicationStatus.GHOSTED,
        ApplicationStatus.WITHDRAWN,
    ]
    counts = {status: 0 for status in order}
    for application in applications:
        counts[application.status] = counts.get(application.status, 0) + 1
    return {status: count for status, count in counts.items() if count}


def _funnel(applications: tuple[TrackedApplication, ...]) -> tuple[Stage, ...]:
    """How many applications ever reached each stage.

    Counted from the events and not from the current status. An application that was
    rejected after an onsite still reached the onsite. The first stage is counted as
    `sent` rather than from its event, so that this table and the response rate below
    are always shares of the same number.
    """
    reached = {stage: 0 for stage in FUNNEL_STAGES}
    for application in applications:
        kinds = {event.event for event in application.events}
        for stage in FUNNEL_STAGES:
            if stage in kinds:
                reached[stage] += 1
    reached[EventKind.APPLIED] = sum(1 for a in applications if a.sent)
    applied = reached[EventKind.APPLIED]
    return tuple(
        Stage(stage, count, count / applied if applied else 0.0)
        for stage, count in reached.items()
    )


def _rates(applications: tuple[TrackedApplication, ...], key) -> tuple[Rate, ...]:
    """Response rate per slice (biggest first) over applications that were sent."""
    counts: dict[str, list[int]] = {}
    for application in (a for a in applications if a.sent):
        row = counts.setdefault(key(application), [0, 0])
        row[0] += 1
        row[1] += 1 if application.responded else 0
    return tuple(
        Rate(label, applied, responded)
        for label, (applied, responded) in sorted(
            counts.items(), key=lambda item: (-item[1][0], item[0])
        )
    )


def _open_roles(
    session: Session,
    profile: SearchProfile,
    suppression: RepeatSuppressionConfig,
    *,
    limit: int,
) -> tuple[tuple[OpenRole, ...], int]:
    """Everything above the threshold that has been neither applied to nor dismissed."""
    winners = best_scores(session, profile.scoring.prompt_version)
    shortlist = {
        job_id: score
        for job_id, score in winners.items()
        if score.score >= profile.scoring.threshold
    }
    if not shortlist:
        return (), 0
    actioned = set(
        session.scalars(
            select(Application.job_id).where(
                Application.job_id.in_(list(shortlist)),
                Application.status.in_(ACTIONED_STATUSES),
            )
        )
    )
    remaining = [job_id for job_id in shortlist if job_id not in actioned]
    if not remaining:
        return (), 0
    history = _appearance_history(session, remaining)
    rows = session.execute(
        select(Job, Company.name)
        .join(Company, Job.company_id == Company.id)
        .where(Job.id.in_(remaining))
    ).all()
    roles = []
    for job, company_name in rows:
        score = shortlist[job.id]
        shown = appearances_since_change(
            history.get(job.id, []), score.score, job.content_hash, suppression
        )
        # What the next digest would do with it, which is what decides whether
        # this row is here because you have not got to it or because the email
        # has stopped asking.
        is_suppressed, _ = suppression_state(shown + 1, suppression)
        roles.append(
            OpenRole(
                job_id=job.id,
                title=job.title,
                company=company_name,
                location=job.location_raw or "not stated",
                work_mode=job.work_mode,
                score=score.score,
                apply_url=_winning_url(score, job),
                appearances=shown,
                suppressed=is_suppressed,
            )
        )
    roles.sort(key=lambda role: (-role.score, role.company.lower(), role.title.lower()))
    return tuple(roles[:limit]), len(roles)


def _appearance_history(
    session: Session, job_ids: list[int]
) -> dict[int, list[DigestAppearance]]:
    """Every digest these jobs have been in (keyed by job and newest first)."""
    history: dict[int, list[DigestAppearance]] = {}
    rows = session.scalars(
        select(DigestAppearance)
        .where(DigestAppearance.job_id.in_(job_ids))
        .order_by(DigestAppearance.digest_date.desc(), DigestAppearance.id.desc())
    )
    for appearance in rows:
        history.setdefault(appearance.job_id, []).append(appearance)
    return history
