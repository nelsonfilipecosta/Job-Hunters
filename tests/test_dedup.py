"""Tests for deciding which postings are the same real-world opening."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import FakeAdapter, make_posting, ok
from job_hunters.dedup import (
    FUZZY_THRESHOLD,
    canonical_key,
    find_fuzzy_match,
    merge_jobs,
    resolve_job,
    titles_match,
)
from job_hunters.ingest import ingest_company
from job_hunters.models import Application, ApplicationStatus, Company, Job, JobSource, Score
from job_hunters.normalize import normalize_title, parse_location

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _count(session: Session, model) -> int:
    """How many rows of this model exist."""
    return len(session.scalars(select(model)).all())


def test_the_same_role_from_two_sources_collapses_to_one_job(
    session: Session, company: Company
) -> None:
    """Two boards, two source rows and one job. This is the gate."""
    greenhouse = FakeAdapter.returning(
        "greenhouse", make_posting("gh-1", "Research Scientist, Post-Training", source="greenhouse")
    )
    ashby = FakeAdapter.returning(
        "ashby", make_posting("ab-9", "Research Scientist - Post Training", source="ashby")
    )
    ingest_company(session, company, greenhouse, NOW)
    ingest_company(session, company, ashby, NOW)
    session.commit()

    assert _count(session, JobSource) == 2
    assert _count(session, Job) == 1
    job = session.scalar(select(Job))
    assert {s.source for s in job.sources} == {"greenhouse", "ashby"}
    assert job.primary_source_id == min(s.id for s in job.sources)


def test_canonical_key_is_stable_across_cosmetic_differences() -> None:
    """Case, punctuation and separator differences produce the same key."""
    a = canonical_key("Acme", "Research Scientist, Post-Training (Remote)", "portugal")
    b = canonical_key("ACME", "research scientist - post training", "portugal")
    assert a == b


def test_canonical_key_separates_regions_and_companies() -> None:
    """The same title at a different region or company gets a different key."""
    lisbon = canonical_key("Acme", "Research Scientist", "portugal")
    assert lisbon != canonical_key("Acme", "Research Scientist", "us")
    assert lisbon != canonical_key("Other", "Research Scientist", "portugal")


def test_reordered_titles_match() -> None:
    """Swapping the order of a title's two halves still fuzzy-matches."""
    a = normalize_title("Post-Training Research Scientist")
    b = normalize_title("Research Scientist, Post-Training")
    assert titles_match(a, b)


def test_a_comma_versus_a_hyphen_separator_still_matches() -> None:
    """The same words split by a comma or a hyphen are still the same title."""
    a = normalize_title("Research Scientist, Post-Training")
    b = normalize_title("Research Scientist - Post Training")
    assert titles_match(a, b)


def test_seniority_is_never_fuzzed_away() -> None:
    """A seniority prefix makes a title a different role."""
    base = normalize_title("Research Scientist, Alignment")
    assert not titles_match(normalize_title("Senior Research Scientist, Alignment"), base)
    assert not titles_match(normalize_title("Staff Research Scientist, Alignment"), base)


def test_one_changed_word_in_a_specialism_is_a_different_role() -> None:
    """Post-training and pre-training must never fuzzy-match."""
    post = normalize_title("Research Scientist, Post-Training")
    pre = normalize_title("Research Scientist, Pre-Training")
    assert not titles_match(post, pre), "Post- and Pre-Training are different jobs"


def test_different_disciplines_never_match() -> None:
    """Scientist vs Engineer and Inference vs Training are never the same role."""
    assert not titles_match(normalize_title("Research Scientist"), normalize_title("Research Engineer"))
    assert not titles_match(
        normalize_title("Software Engineer, Inference"), normalize_title("Software Engineer, Training")
    )


def test_threshold_is_the_one_the_evidence_supports() -> None:
    """95 and not 92: at 92 'Post-Training' vs 'Pre-Training' (92.1) would merge."""
    assert FUZZY_THRESHOLD == 95


def test_fuzzy_match_never_crosses_regions(session: Session, company: Company) -> None:
    """A candidate in the wrong region is never returned - even with an identical title."""
    lisbon = Job(canonical_key="k1", company_id=company.id, title="Research Scientist",
                 region="portugal", first_seen=NOW)
    session.add(lisbon)
    session.commit()
    assert find_fuzzy_match(session, company.id, company.name,
                            normalize_title("Research Scientist"), "us") is None
    assert find_fuzzy_match(session, company.id, company.name,
                            normalize_title("Research Scientist"), "portugal") is lisbon


def test_a_retitled_posting_renames_its_job_instead_of_creating_a_second(
    session: Session, company: Company
) -> None:
    """Same source id, but edited title: still one job with its key updated in place."""
    adapter = FakeAdapter(
        "greenhouse",
        ok(make_posting("gh-1", "Research Scientist")),
        ok(make_posting("gh-1", "Senior Research Scientist")),
    )
    ingest_company(session, company, adapter, NOW)
    ingest_company(session, company, adapter, NOW + timedelta(hours=2))
    session.commit()

    assert _count(session, Job) == 1
    job = session.scalar(select(Job))
    assert job.title == "Senior Research Scientist"
    assert job.canonical_key == canonical_key(company.name, "Senior Research Scientist", "portugal")


def test_a_retitle_that_lands_on_an_existing_job_merges_them(
    session: Session, company: Company
) -> None:
    """Two postings become the same title: the two jobs merge and both sources kept."""
    first = FakeAdapter.returning("greenhouse", make_posting("gh-1", "Research Scientist, Evals"),
                                  make_posting("gh-2", "Research Scientist, Post-Training"))
    later = FakeAdapter.returning("greenhouse", make_posting("gh-1", "Research Scientist, Post-Training"),
                                  make_posting("gh-2", "Research Scientist, Post-Training"))
    ingest_company(session, company, first, NOW)
    session.commit()
    assert _count(session, Job) == 2
    ingest_company(session, company, later, NOW + timedelta(hours=2))
    session.commit()
    assert _count(session, Job) == 1
    assert _count(session, JobSource) == 2
    assert all(s.job_id is not None for s in session.scalars(select(JobSource)))


def _job(session: Session, company: Company, key: str, first_seen: datetime) -> Job:
    """A saved Job row with the given canonical key and first-seen time."""
    job = Job(canonical_key=key, company_id=company.id, title="Research Scientist",
              region="portugal", first_seen=first_seen)
    session.add(job)
    session.flush()
    return job


def test_the_job_with_an_application_survives_a_merge(session: Session, company: Company) -> None:
    """A job with application history always wins the merge regardless of age."""
    older = _job(session, company, "a", NOW - timedelta(days=30))
    newer = _job(session, company, "b", NOW)
    session.add(Application(job_id=newer.id, status=ApplicationStatus.APPLIED))
    session.commit()

    survivor = merge_jobs(session, older, newer)
    session.commit()
    assert survivor.id == newer.id, "the applied-to job must survive, even though it is newer"
    assert _count(session, Job) == 1
    assert session.scalar(select(Application)).job_id == newer.id


def test_the_job_with_a_score_survives_over_an_unscored_one(session: Session, company: Company) -> None:
    """A scored job outranks an unscored one (second in the merge priority)."""
    older = _job(session, company, "a", NOW - timedelta(days=30))
    newer = _job(session, company, "b", NOW)
    session.add(Score(job_id=newer.id, score=80, prompt_version=1, model="m", content_hash="c" * 64))
    session.commit()
    assert merge_jobs(session, older, newer).id == newer.id


def test_otherwise_the_oldest_survives(session: Session, company: Company) -> None:
    """With neither job applied-to or scored, the older one is kept."""
    older = _job(session, company, "a", NOW - timedelta(days=30))
    newer = _job(session, company, "b", NOW)
    session.commit()
    assert merge_jobs(session, newer, older).id == older.id


def test_two_applied_to_jobs_are_never_merged(session: Session, company: Company) -> None:
    """Merging would lose one application history. Refuse and keep both."""
    a = _job(session, company, "a", NOW)
    b = _job(session, company, "b", NOW)
    session.add_all([Application(job_id=a.id, status=ApplicationStatus.APPLIED),
                     Application(job_id=b.id, status=ApplicationStatus.APPLIED)])
    session.commit()
    assert merge_jobs(session, a, b) is a
    assert _count(session, Job) == 2


def test_merge_moves_sources_and_scores_to_the_survivor(session: Session, company: Company) -> None:
    """Both scored, so the older survives. The other's source and score move to it."""
    keep = _job(session, company, "a", NOW - timedelta(days=1))
    drop = _job(session, company, "b", NOW)
    session.add_all([
        Score(job_id=keep.id, score=70, prompt_version=1, model="m", content_hash="c" * 64),
        JobSource(company_id=company.id, job_id=drop.id, source="ashby", source_job_id="x", content_hash="h" * 64),
        Score(job_id=drop.id, score=60, prompt_version=1, model="m", content_hash="d" * 64),
    ])
    session.commit()
    session.expire_all()
    keep, drop = session.get(Job, keep.id), session.get(Job, drop.id)

    survivor = merge_jobs(session, keep, drop)
    session.commit()
    session.expire_all()
    survivor = session.get(Job, survivor.id)
    assert survivor.id == keep.id
    assert [s.source for s in survivor.sources] == ["ashby"], "the source moved"
    assert sorted(s.score for s in survivor.scores) == [60, 70], "the score moved, nothing was lost"
    assert _count(session, Job) == 1 and _count(session, Score) == 2


def test_resolve_job_creates_with_parsed_location(session: Session, company: Company) -> None:
    """A first sighting creates a job with the location already parsed onto it."""
    posting = make_posting("1", "Research Scientist", location="Zürich, CH")
    parsed = parse_location(posting.location_raw)
    job, created = resolve_job(session, company.id, company.name, posting, parsed, NOW)
    assert created and job.region == "switzerland" and job.regions == ["switzerland"]
    again, created_again = resolve_job(session, company.id, company.name, posting, parsed, NOW)
    assert not created_again and again.id == job.id
