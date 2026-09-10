"""The scoring pipeline.

Biased towards recall. Every step below would rather waste a cheap step or
a cheap model call than drop a posting that could be worth seeing. In order:

    1. Only open postings not yet judged under the current `prompt_version`
       are considered at all. Everything else here operates on that set.
    2. If the title of a job contains an excluded title word or location, all
       postings of that job are eliminated. This hard prefilter is done once
       per job since every posting of a job shares its title and location.
    3. A generous union keeps multiple postings of a job if the title matches
       an included term or the text description contains a domain keyword.
       It is deliberately noisy. Two postings of one job can have different
       text, so this step cannot be shared like step 2 can.
    4. Survivors are grouped by identical text. Each group is one model call,
       oldest group first, up to `max_llm_scores_per_run` calls. A group that
       does not fit under the cap carries over to the next run untouched.
    5. The judge reads each group's text once and the verdict is written as
       one `scores` row per posting in that group, so postings with identical
       text never cost a second call.
       
A job's own score is not decided here. It is computed on demand by comparing
the best scoring posting among the job's currently open postings through
`best_scores()`. It is computed on demand rather than stored on `jobs` so it
cannot go stale when a posting closes or a sibling is judged later with no
signal to refresh it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from sqlalchemy import exists, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, defer

from . import regions
from .config import SearchProfile, load_all
from .db import session_scope
from .judge import (
    Judge,
    JudgeError,
    PostingText,
    Usage,
    Verdict,
    build_system_prompt,
    load_profile_text,
    make_client,
)
from .models import Company, Job, JobSource, LocationFit, Score, WorkMode
from .normalize import content_hash
from .sources import replay_posting
from .sources.base import RawPosting

log = logging.getLogger("job_hunters.scoring")

MAX_CONSECUTIVE_FAILURES = 5

Replay = Callable[[str, dict], RawPosting]
VerdictCallback = Callable[["Candidate", Verdict, Usage], None]


# ---------------------------------------------------------------------------
# Location fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocationMatch:
    """Where a job falls in the location rules and which place put it there."""

    fit: str
    place: str | None = None


def location_match(
    profile: SearchProfile, region: str, job_regions: Iterable[str], work_mode: str
) -> LocationMatch:
    """Where a job falls in the declared location rules. A pure config lookup.

    `unknown` when the location could not be parsed or the work mode is not
    known, so the job reaches the digest's "Worth checking" section instead of
    being judged on a guess. `excluded` when the location parsed confidently
    and no rule wants it, which includes `other`. A job listed in several
    places counts under the best of them.

    The place that matched is returned alongside the fit. A London role can be
    `acceptable` on the strength of a Toronto office and a digest that shows
    only "London" leaves the reader wondering why it is there at all. Places
    are tried in the order the posting listed them, so the answer is the first
    office that qualifies rather than an arbitrary one.
    """
    if region == regions.UNKNOWN or work_mode == WorkMode.UNKNOWN:
        return LocationMatch(LocationFit.UNKNOWN)
    places = list(job_regions) or [region]
    for fit, rules in (
        (LocationFit.PRIORITY, profile.location.priority),
        (LocationFit.ACCEPTABLE, profile.location.acceptable),
    ):
        for place in places:
            if any(rule.matches(place, work_mode) for rule in rules):
                return LocationMatch(fit, place)
    return LocationMatch(LocationFit.EXCLUDED)


def location_fit(
    profile: SearchProfile, region: str, job_regions: Iterable[str], work_mode: str
) -> str:
    """Just the fit for the callers that do not care which place earned it."""
    return location_match(profile, region, job_regions, work_mode).fit


# ---------------------------------------------------------------------------
# Prefilter
# ---------------------------------------------------------------------------


def _normalize_text(text: str | None) -> str:
    """Lowercase with every run of punctuation or whitespace collapsed to one space."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


class TermMatcher:
    """Whole-word and punctuation-insensitive search for a list of configured terms.

    "post-training" matches "Post Training" and "post-training", but not
    "posttraining". "SFT" matches the token and not "soft". With `prefix=True`
    a term also matches a longer word it starts ("intern" hits "internship").
    """

    def __init__(self, terms: Iterable[str], *, prefix: bool = False) -> None:
        """Compiles one pattern per term in the order the config lists them."""
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for term in terms:
            words = _normalize_text(term).split()
            if not words:
                continue
            body = r"\s+".join(re.escape(word) for word in words)
            tail = "" if prefix else r"\b"
            self._patterns.append((term, re.compile(rf"\b{body}{tail}")))

    def first_match(self, text: str | None) -> str | None:
        """The first configured term found in the text or None."""
        haystack = _normalize_text(text)
        for term, pattern in self._patterns:
            if pattern.search(haystack):
                return term
        return None

    def matches(self, text: str | None) -> bool:
        """True when any configured term appears in the text."""
        return self.first_match(text) is not None


class Prefilter:
    """The prefilter as configured by `search_profile.yaml`."""

    def __init__(self, profile: SearchProfile) -> None:
        """Builds the matchers once since a run applies them to thousands of postings."""
        self.profile = profile
        self._excluded_titles = TermMatcher(profile.titles.exclude)
        self._excluded_seniority = TermMatcher(profile.seniority.exclude, prefix=True)
        self._included_titles = TermMatcher(profile.titles.include)
        self._keywords = TermMatcher([*profile.keywords.strong, *profile.keywords.supporting])

    def title_exclusion(self, title: str) -> str | None:
        """The excluded title or seniority word found in a title or None."""
        return self._excluded_titles.first_match(title) or self._excluded_seniority.first_match(title)

    def exclusion_reason(
        self, title: str, region: str, job_regions: Iterable[str], work_mode: str
    ) -> str | None:
        """Why a job never reaches the judge or None when it may."""
        term = self.title_exclusion(title)
        if term is not None:
            return f"title: {term}"
        if location_fit(self.profile, region, job_regions, work_mode) == LocationFit.EXCLUDED:
            return "location"
        return None

    def matched_term(self, title: str, description: str | None) -> str | None:
        """The term that lets a posting through the generous union or None."""
        return self._included_titles.first_match(title) or self._keywords.first_match(
            f"{title}\n{description or ''}"
        )


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One open posting that survived the prefilter and is ready for the judge."""

    source_id: int
    job_id: int
    content_hash: str
    text: PostingText
    location_fit: str
    matched_on: str


@dataclass
class ScoringReport:
    """What one scoring run did: from open postings down to rows written."""

    open_postings: int = 0
    unjudged_postings: int = 0
    eliminated_title: int = 0
    eliminated_location: int = 0
    unmatched: int = 0
    # Postings whose stored text hash no longer matches their text (see `select_candidates`).
    stale: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    distinct_texts: int = 0
    cap: int = 0
    judged: int = 0
    scored: int = 0
    failed: int = 0
    # Calls after the first that read nothing from the cache. Should be zero.
    cold_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    aborted: str | None = None
    # The judge model this run used. So a caller can report a cache problem
    # against the model actually configured. Empty after a dry run.
    model: str = ""

    @property
    def already_judged(self) -> int:
        """Open postings skipped because their current text was judged before."""
        return self.open_postings - self.unjudged_postings

    @property
    def carried_over(self) -> int:
        """Distinct texts left for the next run (due to the cap or to a failure)."""
        return max(self.distinct_texts - self.judged - self.failed, 0)


def select_candidates(
    session: Session,
    profile: SearchProfile,
    prompt_version: int,
    report: ScoringReport,
    replay: Replay = replay_posting,
) -> list[Candidate]:
    """Runs the prefilter over every open posting not yet judged at its current text.

    The "already judged" test is done in SQL: a `scores` row for this posting
    with its current `content_hash` under this prompt version. Everything
    else is decided in Python on the replayed posting. Eliminations are
    remembered per job since siblings share a title and a region.

    A posting is skipped as stale when the text replayed now hashes to
    something other than the stored `content_hash`. That means the adapter or
    the normalizer changed since the last ingest. Ingest recomputes the hash.
    Judging now would key the score on a hash nothing else knows.
    """
    prefilter = Prefilter(profile)
    open_postings = (JobSource.is_open.is_(True), JobSource.job_id.is_not(None))
    report.open_postings = session.scalar(
        select(func.count()).select_from(JobSource).where(*open_postings)
    ) or 0

    already_judged = exists().where(
        Score.source_id == JobSource.id,
        Score.content_hash == JobSource.content_hash,
        Score.prompt_version == prompt_version,
    )
    rows = session.execute(
        select(JobSource, Job, Company)
        .join(Job, JobSource.job_id == Job.id)
        .join(Company, Job.company_id == Company.id)
        .where(*open_postings, ~already_judged)
        .options(defer(JobSource.raw_json))
        .order_by(JobSource.first_seen, JobSource.id)
    ).all()
    report.unjudged_postings = len(rows)

    reasons: dict[int, str | None] = {}
    candidates: list[Candidate] = []
    for source, job, company in rows:
        if job.id not in reasons:
            reasons[job.id] = prefilter.exclusion_reason(
                job.title, job.region, job.regions or [], job.work_mode
            )
        reason = reasons[job.id]
        if reason is not None:
            if reason.startswith("title"):
                report.eliminated_title += 1
            else:
                report.eliminated_location += 1
            continue

        posting = replay(source.source, source.raw_json)
        text_hash = content_hash(
            company.name, posting.title, posting.location_raw, posting.description
        )
        if text_hash != source.content_hash:
            report.stale += 1
            continue
        matched = prefilter.matched_term(posting.title, posting.description)
        if matched is None:
            report.unmatched += 1
            continue
        candidates.append(
            Candidate(
                source_id=source.id,
                job_id=job.id,
                content_hash=text_hash,
                text=PostingText(
                    company=company.name,
                    title=posting.title,
                    location_raw=posting.location_raw,
                    region=job.region,
                    work_mode=job.work_mode,
                    description=posting.description,
                ),
                location_fit=location_fit(profile, job.region, job.regions or [], job.work_mode),
                matched_on=matched,
            )
        )
    report.candidates = candidates
    report.distinct_texts = len({c.content_hash for c in candidates})
    return candidates


def group_by_text(candidates: Iterable[Candidate]) -> list[list[Candidate]]:
    """Candidates bucketed by text hash, with each bucket in first-seen order."""
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.content_hash, []).append(candidate)
    return list(groups.values())


def judge_candidates(
    session: Session,
    candidates: list[Candidate],
    judge: Judge,
    *,
    prompt_version: int,
    limit: int,
    report: ScoringReport,
    on_verdict: VerdictCallback | None = None,
) -> None:
    """Calls the judge once per distinct text (up to the cap) and writes a `scores` row per posting.

    Each text's rows are committed on their own, so a crash halfway keeps
    what was already paid for. A fatal error (a rejected key or an unknown
    model) stops the run. Any other error skips that text and carries on
    until too many fail in a row.
    """
    report.cap = limit
    report.model = judge.model
    streak = 0
    for siblings in group_by_text(candidates)[:limit]:
        lead = siblings[0]
        try:
            verdict, usage = judge.judge(lead.text)
        except JudgeError as exc:
            report.failed += 1
            streak += 1
            log.warning("%s @ %s: %s", lead.text.title, lead.text.company, exc)
            if exc.fatal:
                report.aborted = str(exc)
                break
            if streak >= MAX_CONSECUTIVE_FAILURES:
                report.aborted = f"{streak} texts failed in a row. Last error: {exc}"
                break
            continue
        streak = 0
        report.judged += 1
        report.usage = report.usage + usage
        if report.judged > 1 and usage.cache_read_input_tokens == 0:
            report.cold_calls += 1

        for candidate in siblings:
            session.add(
                Score(
                    source_id=candidate.source_id,
                    score=verdict.score,
                    summary=verdict.summary,
                    rationale=verdict.rationale,
                    matched_areas=list(verdict.matched_areas),
                    concerns=list(verdict.concerns),
                    work_authorization=verdict.work_authorization,
                    location_fit=candidate.location_fit,
                    prompt_version=prompt_version,
                    model=judge.model,
                    content_hash=candidate.content_hash,
                )
            )
        try:
            session.commit()
        except IntegrityError:
            # Another run judged this text between our select and our insert.
            # The verdict was paid for twice, but the rows exist.
            session.rollback()
            log.warning("%s @ %s: already scored by a concurrent run.", lead.text.title, lead.text.company)
            continue
        report.scored += len(siblings)
        if on_verdict is not None:
            on_verdict(lead, verdict, usage)


def run_scoring(
    *,
    limit: int | None = None,
    dry_run: bool = False,
    judge: Judge | None = None,
    on_verdict: VerdictCallback | None = None,
) -> ScoringReport:
    """The prefilter over the whole database and then the judge up to the cap. What `job-hunters score` runs.

    `dry_run` stops after the prefilter and writes nothing, which needs no API key.
    `judge` lets tests supply a fake. Otherwise one is built from config and `ANTHROPIC_API_KEY`.
    """
    config = load_all()
    profile = config.search_profile
    prompt_version = profile.scoring.prompt_version
    cap = limit if limit is not None else profile.scoring.max_llm_scores_per_run
    report = ScoringReport(cap=cap)

    if not dry_run and judge is None:
        api_key = config.secrets.require("anthropic_api_key")
        judge = Judge(
            make_client(api_key),
            config.system.models.judge,
            build_system_prompt(profile, load_profile_text()),
        )

    with session_scope() as session:
        candidates = select_candidates(session, profile, prompt_version, report)
        if dry_run:
            return report
        judge_candidates(
            session, candidates, judge, prompt_version=prompt_version, limit=cap,
            report=report, on_verdict=on_verdict,
        )
    return report


# ---------------------------------------------------------------------------
# Reading scores back
# ---------------------------------------------------------------------------


def best_scores(session: Session, prompt_version: int) -> dict[int, Score]:
    """The winning judgement per job: the best score among its open postings, keyed by job id.

    Only judgements of a posting's *current* text count (`scores.content_hash`
    equal to `job_sources.content_hash`), so a reworded posting waiting to be
    rescored does not keep an old verdict in play. Ties go to the earliest
    row. Computed on read rather than stored on `jobs` so it cannot go stale
    when a posting closes or a sibling is judged later. The digest reads the
    summary and apply link from the winning posting.
    """
    rows = session.execute(
        select(Score, JobSource.job_id)
        .join(JobSource, Score.source_id == JobSource.id)
        .where(
            JobSource.is_open.is_(True),
            JobSource.job_id.is_not(None),
            Score.prompt_version == prompt_version,
            Score.content_hash == JobSource.content_hash,
        )
        .order_by(Score.score.desc(), Score.id)
    ).all()
    winners: dict[int, Score] = {}
    for score, job_id in rows:
        winners.setdefault(job_id, score)
    return winners
