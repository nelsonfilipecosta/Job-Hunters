"""The daily email.

A template renderer over a SQL query with no LLM anywhere in it. Everything
it shows was decided earlier - the judge wrote the score and the summary, the
normalizer parsed the location. In order:

    1. Take each job's best score among its currently open postings
       (`scoring.best_scores`), which is also what decides which posting the
       entry links to.
    2. Drop anything below `scoring.threshold` and anything already acted on
       (a job you applied to or dismissed does not come back).
    3. Section each survivor on its location fit plus the judge's work
       authorization finding. A job no location rule accepts is dropped here.
    4. Apply repeat suppression so a job you neither applied to nor dismissed
       stops filling the email forever.
    5. Render, send and record one `digest_appearances` row per entry - which
       is what step 4 counts on the next run.

Sections mirror `LocationFit`, so one vocabulary describes both. `unknown`
routes to "Worth checking" rather than being dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .actions import ActionLink, action_links
from .config import AppConfig, RepeatSuppressionConfig, load_all
from .db import session_scope
from .mailer import Message, Sender, SmtpSender
from .models import (
    Application,
    ApplicationStatus,
    Company,
    DigestAppearance,
    DigestSection,
    Job,
    LocationFit,
    Score,
    WorkAuthStatus,
)
from .scoring import LocationMatch, best_scores, location_match
from .templating import render as render_template

ACTIONED_STATUSES: frozenset[str] = frozenset(
    set(ApplicationStatus) - {ApplicationStatus.INTERESTED}
)


@dataclass(frozen=True)
class SectionSpec:
    """One section of the email: which location fit lands in it and how it reads."""

    key: str
    icon: str
    title: str
    blurb: str


SECTION_SPECS: dict[str, SectionSpec] = {
    LocationFit.PRIORITY: SectionSpec(
        DigestSection.PRIORITY, "🥇", "Priority",
        "Matches a priority rule in your location profile.",
    ),
    LocationFit.ACCEPTABLE: SectionSpec(
        DigestSection.ACCEPTABLE, "✅", "Acceptable",
        "Matches an acceptable rule in your location profile.",
    ),
    LocationFit.UNKNOWN: SectionSpec(
        DigestSection.WORTH_CHECKING, "⚠️", "Worth Checking",
        "The location or the work authorization could not be settled from the "
        "posting. Each entry says which.",
    ),
}


@dataclass(frozen=True)
class LinkFactory:
    """Everything one run needs to sign a link, so entries do not carry it about."""

    base_url: str
    secret: str
    ttl_days: int
    now: datetime

    def for_job(self, job_id: int) -> tuple[ActionLink, ...]:
        """The four signed links that sit under one entry."""
        return action_links(
            self.base_url, self.secret, job_id, ttl_days=self.ttl_days, now=self.now
        )


@dataclass(frozen=True)
class Entry:
    """One job as the email shows it."""

    job_id: int
    company: str
    title: str
    location: str
    work_mode: str
    region: str
    score: int
    summary: str
    concerns: tuple[str, ...]
    work_authorization: str
    apply_url: str | None
    links: tuple[ActionLink, ...]
    # How many digests this job has appeared in, counting this one.
    appearance: int
    demoted: bool
    content_hash: str | None
    # Which section this entry is in and the country whose rule put it there.
    fit: str = ""
    matched_place: str | None = None

    @property
    def section_note(self) -> str:
        """Why this entry is in the section it is in when the line above does not show it."""
        if self.fit == LocationFit.UNKNOWN:
            return "Location could not be settled."
        if self.work_authorization != WorkAuthStatus.ELIGIBLE:
            return f"The location qualifies{self._via}. The work authorization does not."
        return f"{self.fit} via {self.matched_place}" if self._via else ""

    @property
    def _via(self) -> str:
        """Return the country that matched the rules when it is not the one on display."""
        if not self.matched_place or self.matched_place == self.region:
            return ""
        return f" (via {self.matched_place})"

    @property
    def repeat_note(self) -> str:
        """The dismiss nudge or nothing while the job is still new."""
        if not self.demoted:
            return ""
        return f"{_ordinal(self.appearance)} time — dismiss?"


@dataclass
class Section:
    """One rendered section and the entries in it, already in display order."""

    spec: SectionSpec
    entries: list[Entry] = field(default_factory=list)


@dataclass
class Digest:
    """One day's email ready to render."""

    digest_date: date
    sections: list[Section]
    threshold: int
    still_open: int
    still_open_url: str
    generated_at: datetime

    @property
    def entries(self) -> list[Entry]:
        """Every entry in the body across all sections."""
        return [entry for section in self.sections for entry in section.entries]

    @property
    def total(self) -> int:
        """How many jobs the body shows."""
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        """True when nothing cleared the threshold that was not already suppressed."""
        return self.total == 0

    def subject(self) -> str:
        """The subject line, which is most of what gets read on a phone."""
        stamp = f"{self.digest_date:%a %d %b}"
        if self.is_empty:
            return f"Job-Hunters {stamp}: nothing new"
        priority = sum(
            len(s.entries) for s in self.sections if s.spec.key == DigestSection.PRIORITY
        )
        roles = f"{self.total} role{'s' if self.total != 1 else ''}"
        return f"Job-Hunters {stamp}: {roles}, {priority} priority"


_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(number: int) -> str:
    """Convert the number of appearances into its ordinal form."""
    if 11 <= number % 100 <= 13:
        return f"{number}th"
    return f"{number}{_ORDINAL_SUFFIXES.get(number % 10, 'th')}"


# ---------------------------------------------------------------------------
# Repeat suppression
# ---------------------------------------------------------------------------


def appearances_since_change(
    history: list[DigestAppearance],
    score: int,
    content_hash: str | None,
    suppression: RepeatSuppressionConfig,
) -> int:
    """How often this job has been shown since the last time it materially changed."""
    count = 0
    for appearance in history:
        if appearance.content_hash_at_appearance != content_hash:
            break
        previous = appearance.score_at_appearance
        if previous is not None and abs(score - previous) > suppression.reset_on_score_delta:
            break
        count += 1
    return count


def suppression_state(appearance: int, suppression: RepeatSuppressionConfig) -> tuple[bool, bool]:
    """Whether this appearance is suppressed and whether it is demoted."""
    if not suppression.enabled:
        return False, False
    if appearance > suppression.suppress_after:
        return True, True
    return False, appearance > suppression.demote_after


# ---------------------------------------------------------------------------
# Building one digest
# ---------------------------------------------------------------------------


def _section_for(fit: str, work_authorization: str) -> SectionSpec | None:
    """Which section an entry belongs in or None when it does not belong at all."""
    if fit == LocationFit.EXCLUDED:
        return None
    if work_authorization != WorkAuthStatus.ELIGIBLE:
        return SECTION_SPECS[LocationFit.UNKNOWN]
    return SECTION_SPECS[fit]


def build_digest(
    session: Session,
    config: AppConfig,
    secret: str,
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> Digest:
    """Selects, sections and orders everything one day's email should show."""
    now = now or datetime.now(UTC)
    today = today or _local_today(config.system.timezone, now)
    profile = config.search_profile
    suppression = config.system.digest.repeat_suppression
    base_url = config.system.base_url
    links = LinkFactory(base_url, secret, config.system.actions.token_ttl_days, now)

    winners = best_scores(session, profile.scoring.prompt_version)
    shortlist = {
        job_id: score
        for job_id, score in winners.items()
        if score.score >= profile.scoring.threshold
    }
    if not shortlist:
        return Digest(
            digest_date=today,
            sections=_empty_sections(),
            threshold=profile.scoring.threshold,
            still_open=0,
            still_open_url=f"{base_url}/",
            generated_at=now,
        )

    job_ids = list(shortlist)
    actioned = set(
        session.scalars(
            select(Application.job_id).where(
                Application.job_id.in_(job_ids),
                Application.status.in_(ACTIONED_STATUSES),
            )
        )
    )
    history = _appearance_history(session, job_ids, today)
    rows = session.execute(
        select(Job, Company.name)
        .join(Company, Job.company_id == Company.id)
        .where(Job.id.in_(job_ids))
    ).all()

    sections = {spec.key: Section(spec) for spec in _ordered_specs()}
    suppressed = 0
    for job, company_name in rows:
        if job.id in actioned:
            continue
        score = shortlist[job.id]
        match = location_match(profile, job.region, job.regions or [], job.work_mode)
        spec = _section_for(match.fit, score.work_authorization)
        if spec is None:
            continue

        appearance = 1 + appearances_since_change(
            history.get(job.id, []), score.score, job.content_hash, suppression
        )
        is_suppressed, demoted = suppression_state(appearance, suppression)
        if is_suppressed:
            suppressed += 1
            continue
        sections[spec.key].entries.append(
            _entry(job, company_name, score, links, appearance, demoted, match)
        )

    for section in sections.values():
        # Demoted entries sink to the bottom of their own section.
        section.entries.sort(key=lambda e: (e.demoted, -e.score, e.company.lower(), e.title.lower()))
    return Digest(
        digest_date=today,
        sections=list(sections.values()),
        threshold=profile.scoring.threshold,
        still_open=suppressed,
        still_open_url=f"{base_url}/",
        generated_at=now,
    )


def _entry(
    job: Job,
    company_name: str,
    score: Score,
    links: LinkFactory,
    appearance: int,
    demoted: bool,
    match: LocationMatch,
) -> Entry:
    """Turns one job and its winning judgement into a row of the email.

    The apply link comes from the posting that won rather than from the job's
    own `apply_url`, so the summary beside it was written about the text that
    link opens.
    """
    posting = score.source
    return Entry(
        job_id=job.id,
        company=company_name,
        title=job.title,
        location=job.location_raw or "not stated",
        work_mode=job.work_mode,
        region=job.region,
        score=score.score,
        summary=(score.summary or "").strip(),
        concerns=tuple(score.concerns or ()),
        work_authorization=score.work_authorization,
        apply_url=(posting.url if posting is not None else None) or job.apply_url,
        links=links.for_job(job.id),
        appearance=appearance,
        demoted=demoted,
        content_hash=job.content_hash,
        fit=match.fit,
        matched_place=match.place,
    )


def _ordered_specs() -> list[SectionSpec]:
    """The sections in the order they are printed."""
    return [SECTION_SPECS[fit] for fit in
            (LocationFit.PRIORITY, LocationFit.ACCEPTABLE, LocationFit.UNKNOWN)]


def _empty_sections() -> list[Section]:
    """Empty sections, so an empty digest renders through the same template."""
    return [Section(spec) for spec in _ordered_specs()]


def _appearance_history(
    session: Session, job_ids: list[int], today: date
) -> dict[int, list[DigestAppearance]]:
    """Every earlier appearance of these jobs, newest first, keyed by job.

    Today's own rows are excluded so that re-running the digest on one day
    reads the same history the first run did instead of counting itself.
    """
    history: dict[int, list[DigestAppearance]] = {}
    rows = session.scalars(
        select(DigestAppearance)
        .where(
            DigestAppearance.job_id.in_(job_ids),
            DigestAppearance.digest_date < today,
        )
        .order_by(DigestAppearance.digest_date.desc(), DigestAppearance.id.desc())
    )
    for appearance in rows:
        history.setdefault(appearance.job_id, []).append(appearance)
    return history


def _local_today(timezone: str, now: datetime) -> date:
    """The time at the timezone where you are not where UTC is."""
    return now.astimezone(ZoneInfo(timezone)).date()


def record_appearances(session: Session, digest: Digest) -> int:
    """Writes one `digest_appearances` row per entry that was actually sent.

    This is the memory repeat suppression reads on the next run, so it is
    written only after a send succeeds. Any rows already carrying today's date
    are replaced, which keeps a resend on the same day from being counted as a
    second appearance and from colliding with the one-per-day constraint.

    Suppressed jobs deliberately get no row. They were shown as a count and not
    as an entry, so their counter freezes where it is.
    """
    entries = digest.entries
    if not entries:
        return 0
    session.execute(
        delete(DigestAppearance).where(
            DigestAppearance.job_id.in_([e.job_id for e in entries]),
            DigestAppearance.digest_date == digest.digest_date,
        )
    )
    for section in digest.sections:
        for position, entry in enumerate(section.entries, start=1):
            session.add(
                DigestAppearance(
                    job_id=entry.job_id,
                    digest_date=digest.digest_date,
                    section=section.spec.key,
                    position=position,
                    score_at_appearance=entry.score,
                    content_hash_at_appearance=entry.content_hash,
                )
            )
    return len(entries)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(digest: Digest, template: str) -> str:
    """Renders one digest through `templates/<template>`."""
    return render_template(template, digest=digest)


def render_bodies(digest: Digest) -> tuple[str, str]:
    """Both bodies of the email as plain text and HTML rendered together."""
    return render(digest, "digest.txt"), render(digest, "digest.html")


def render_message(
    digest: Digest, to: str, sender: str, bodies: tuple[str, str] | None = None
) -> Message:
    """The whole email: subject, plain text and HTML, addressed and ready to send."""
    text, html = bodies if bodies is not None else render_bodies(digest)
    return Message(to=to, sender=sender, subject=digest.subject(), text=text, html=html)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


@dataclass
class DigestReport:
    """What one `job-hunters digest` did."""

    digest: Digest
    sent_to: str | None = None
    recorded: int = 0
    html: str = ""


def run_digest(
    *,
    dry_run: bool = False,
    sender: Sender | None = None,
    config: AppConfig | None = None,
) -> DigestReport:
    """Builds the digest and, unless this is a dry run, sends it and records it.

    A dry run still signs real links, because a preview whose links differ from
    the ones that go out cannot tell you whether the real ones work. It needs
    `ACTION_TOKEN_SECRET` for that, but no mail credentials.
    """
    config = config or load_all()
    secret = config.secrets.require("action_token_secret")

    with session_scope() as session:
        digest = build_digest(session, config, secret)

    bodies = render_bodies(digest)
    report = DigestReport(digest=digest, html=bodies[1])
    if dry_run:
        return report

    to = config.secrets.require("digest_to")
    sender = sender or _smtp_sender(config)
    sender.send(render_message(digest, to, config.secrets.optional("digest_from") or to, bodies))
    report.sent_to = to

    with session_scope() as session:
        report.recorded = record_appearances(session, digest)
    return report


def _smtp_sender(config: AppConfig) -> SmtpSender:
    """The real sender, built from `system_config.yaml` and `.env`."""
    secrets = config.secrets
    username = (
        secrets.optional("smtp_username")
        or secrets.optional("digest_from")
        or secrets.require("digest_to")
    )
    return SmtpSender(
        host=config.system.email.smtp_host,
        port=config.system.email.smtp_port,
        username=username,
        password=secrets.require("smtp_password"),
    )
