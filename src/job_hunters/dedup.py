"""Deciding which postings are the same real-world opening.

Two stages:

1. Canonical key: `sha256(company | normalized title | region)`. Two postings
   with the same key are the same job. This catches everything that differs only
   in punctuation, a case, a suffix or a requisition id (see `normalize_title`).
2. Fuzzy pass: when the key finds nothing, compare the normalized title
   against existing jobs at the same company in the same region using
   `token_sort_ratio`. This catches reordering ("Post-Training Research
   Scientist" vs "Research Scientist, Post-Training").

Then a merge policy for when two rows that already exist turn out to be one
job. The survivor is the row with an application, else the row with a score
or else the oldest. Application history is never lost to a later merge.

`token_set_ratio` ignores words present in only one title, which makes "Senior"
and "Staff" invisible to it. And one changed character in a load-bearing word
must not be a match. Every case the plan wanted merged still scores 100 under
the stricter rule, so nothing is lost. A false merge hides a real job and a
missed merge shows a near-duplicate twice. We prefer the latter.
"""

from __future__ import annotations

from datetime import datetime

from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import DigestAppearance, Job, Score, as_utc
from .normalize import ParsedLocation, normalize_company, normalize_title, sha256
from .sources.base import RawPosting

FUZZY_THRESHOLD = 95


def canonical_key(company_name: str, title: str, region: str) -> str:
    """The deterministic identity of a role: company, normalized title and region."""
    return sha256(normalize_company(company_name), normalize_title(title, company_name), region)


def titles_match(normalized_a: str, normalized_b: str, threshold: int = FUZZY_THRESHOLD) -> bool:
    """True when two already normalized titles are the same role modulo word order."""
    return fuzz.token_sort_ratio(normalized_a, normalized_b) >= threshold


def find_fuzzy_match(
    session: Session,
    company_id: int,
    company_name: str,
    normalized_title: str,
    region: str,
    exclude_id: int | None = None,
) -> Job | None:
    """The best existing job at this company and region whose title fuzzily matches."""
    candidates = session.scalars(
        select(Job).where(Job.company_id == company_id, Job.region == region)
    ).all()
    best: Job | None = None
    best_score = 0.0
    for job in candidates:
        if exclude_id is not None and job.id == exclude_id:
            continue
        score = fuzz.token_sort_ratio(normalized_title, normalize_title(job.title, company_name))
        if score >= FUZZY_THRESHOLD and score > best_score:
            best, best_score = job, score
    return best


def _survivor_rank(job: Job) -> tuple:
    """Higher is better: has an application, else has a score or else is older."""
    return (
        job.application is not None,
        bool(job.scores),
        -(as_utc(job.first_seen).timestamp() if job.first_seen else 0.0),
        -(job.id or 0),
    )


def merge_jobs(session: Session, a: Job, b: Job) -> Job:
    """Collapses two jobs into one and returns the survivor.

    Sources, scores and digest appearances move to the survivor, while the other row
    is deleted. Two jobs that both carry an application are never merged so the caller
    gets `a` back untouched.
    """
    if a.id == b.id:
        return a
    if a.application is not None and b.application is not None:
        return a
    keep, drop = (a, b) if _survivor_rank(a) >= _survivor_rank(b) else (b, a)
    for source in list(drop.sources):
        drop.sources.remove(source)
        keep.sources.append(source)
    existing_scores = {(s.content_hash, s.prompt_version) for s in keep.scores}
    for score in list(drop.scores):
        drop.scores.remove(score)
        if (score.content_hash, score.prompt_version) in existing_scores:
            session.delete(score)
        else:
            keep.scores.append(score)
    existing_days = {
        d.digest_date for d in session.scalars(
            select(DigestAppearance).where(DigestAppearance.job_id == keep.id)
        )
    }
    for appearance in session.scalars(
        select(DigestAppearance).where(DigestAppearance.job_id == drop.id)
    ).all():
        if appearance.digest_date in existing_days:
            session.delete(appearance)
        else:
            appearance.job_id = keep.id
    if drop.application is not None:
        application = drop.application
        drop.application = None
        keep.application = application
    if keep.primary_source_id is None:
        keep.primary_source_id = drop.primary_source_id
    if drop.posted_at and (keep.posted_at is None or as_utc(drop.posted_at) < as_utc(keep.posted_at)):
        keep.posted_at = drop.posted_at

    session.flush()
    session.delete(drop)
    session.flush()
    return keep


def resolve_job(
    session: Session,
    company_id: int,
    company_name: str,
    posting: RawPosting,
    parsed: ParsedLocation,
    now: datetime,
    current: Job | None = None,
) -> tuple[Job, bool]:
    """Finds or creates the job a posting belongs to. Returns (job, created).

    `current` is the job this posting's source row already points at, if any.
    Passing it lets a retitled posting rename its job's key in place instead of
    creating a second job and immediately merging it.
    """
    key = canonical_key(company_name, posting.title, parsed.region)
    if current is not None and current.canonical_key == key:
        return current, False

    normalized = normalize_title(posting.title, company_name)
    match = session.scalar(select(Job).where(Job.canonical_key == key))
    if match is None:
        match = find_fuzzy_match(
            session, company_id, company_name, normalized, parsed.region,
            exclude_id=current.id if current else None,
        )

    if current is None:
        if match is not None:
            return match, False
        job = Job(
            canonical_key=key,
            company_id=company_id,
            title=posting.title,
            region=parsed.region,
            regions=list(parsed.regions),
            work_mode=parsed.work_mode,
            first_seen=now,
            last_seen=now,
        )
        session.add(job)
        session.flush()
        return job, True

    # The posting is known but its key changed (title or region edited).
    if match is None or match.id == current.id:
        current.canonical_key = key
        return current, False
    survivor = merge_jobs(session, current, match)
    if survivor.canonical_key != key:
        survivor.canonical_key = key
    return survivor, False
