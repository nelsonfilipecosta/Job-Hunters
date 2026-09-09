"""Tests for the scoring pipeline.

`FakeAnthropic` stands in for the API, so nothing here costs money. Postings
reach the database through the same ingest code as in production, so the replay
of `raw_json` back into text is exercised too.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import anthropic
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from conftest import FakeAdapter, FakeAnthropic, api_error, make_posting, ok, verdict
from job_hunters.config import SearchProfile
from job_hunters.ingest import ingest_company
from job_hunters.judge import Judge, Verdict
from job_hunters.models import Company, JobSource, LocationFit, Score
from job_hunters.scoring import (
    Prefilter,
    ScoringReport,
    TermMatcher,
    best_scores,
    judge_candidates,
    location_fit,
    run_scoring,
    select_candidates,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)

PROFILE = SearchProfile.model_validate(
    {
        "titles": {
            "include": ["research scientist", "research engineer", "post-training"],
            "exclude": ["sales", "account executive", "counsel"],
        },
        "keywords": {"strong": ["RLHF", "DPO", "post-training", "SFT"], "supporting": ["evaluation"]},
        "seniority": {"include": ["mid", "senior"], "exclude": ["intern", "director"]},
        "location": {
            "base": "portugal",
            "priority": [{"work_modes": ["onsite", "hybrid"], "regions": ["portugal", "switzerland"]}],
            "acceptable": [{"work_modes": ["remote"], "regions": ["eu", "uk", "us"]}],
            "work_authorization": {"have": ["eu", "switzerland"], "need_sponsorship": ["uk", "us"]},
        },
        "scoring": {"threshold": 60, "max_llm_scores_per_run": 300, "prompt_version": 1,
                    "rubric": "Weight post-training.",
                    "bands": [{"low": 60, "high": 100, "meaning": "A fit."},
                              {"low": 0, "high": 59, "meaning": "Not a fit."}]},
    }
)


def _judge(*script) -> tuple[Judge, FakeAnthropic]:
    """A judge answering from the script, plus the fake client to inspect its requests."""
    client = FakeAnthropic(*script)
    return Judge(client, "claude-haiku-4-5", "stable prefix"), client


def _score_all(session: Session, judge: Judge, limit: int = 300) -> ScoringReport:
    """The prefilter then the judge over the whole test database and under the test profile."""
    report = ScoringReport()
    candidates = select_candidates(session, PROFILE, 1, report)
    judge_candidates(session, candidates, judge, prompt_version=1, limit=limit, report=report)
    return report


def _title_of(request: dict) -> str:
    """The `Title:` line of the posting a request carried."""
    return request["messages"][0]["content"].split("\n")[1]


def test_location_fit_is_a_pure_config_lookup() -> None:
    """Priority, acceptable, excluded and unknown all come from the declared rules."""
    assert location_fit(PROFILE, "portugal", ["portugal"], "onsite") == LocationFit.PRIORITY
    assert location_fit(PROFILE, "us", ["us"], "remote") == LocationFit.ACCEPTABLE
    assert location_fit(PROFILE, "us", ["us"], "onsite") == LocationFit.EXCLUDED
    assert location_fit(PROFILE, "other", [], "remote") == LocationFit.EXCLUDED
    assert location_fit(PROFILE, "unknown", [], "remote") == LocationFit.UNKNOWN
    assert location_fit(PROFILE, "us", ["us"], "unknown") == LocationFit.UNKNOWN


def test_a_job_listed_in_several_places_takes_the_best_fit() -> None:
    """London and Lisbon on one posting is a priority job and not a UK one."""
    assert location_fit(PROFILE, "uk", ["uk", "portugal"], "onsite") == LocationFit.PRIORITY


def test_terms_match_whole_words_regardless_of_punctuation() -> None:
    """A hyphen or a space makes no difference, but a term never matches inside another word."""
    matcher = TermMatcher(["post-training", "SFT"])
    assert matcher.first_match("Research Scientist, Post Training") == "post-training"
    assert matcher.matches("we do post-training")
    assert not matcher.matches("posttraining")
    assert matcher.matches("SFT and RL")
    assert not matcher.matches("soft skills"), "SFT must not match inside another word"


def test_seniority_terms_match_as_word_prefixes() -> None:
    """`intern` must catch `Internship`, which a whole-word title term would not."""
    assert TermMatcher(["intern"], prefix=True).matches("Research Internship")
    assert not TermMatcher(["intern"]).matches("Research Internship")
    assert TermMatcher(["director"], prefix=True).matches("Directors of Research")


def test_excluded_titles_seniority_and_locations_are_hard_eliminations() -> None:
    """The three reasons a job never reaches the judge and the one that is not a reason."""
    prefilter = Prefilter(PROFILE)
    assert prefilter.exclusion_reason("Enterprise Account Executive", "portugal", ["portugal"], "onsite") == "title: account executive"
    assert prefilter.exclusion_reason("Research Internship", "portugal", ["portugal"], "onsite") == "title: intern"
    assert prefilter.exclusion_reason("Research Scientist", "us", ["us"], "onsite") == "location"
    assert prefilter.exclusion_reason("Research Scientist", "unknown", [], "unknown") is None, (
        "an unparsed location is never a reason to eliminate"
    )


def test_the_union_keeps_a_title_match_or_a_keyword_match() -> None:
    """Either signal is enough. Neither means the posting is dropped for free."""
    prefilter = Prefilter(PROFILE)
    assert prefilter.matched_term("Research Scientist", "We sell widgets.") == "research scientist"
    assert prefilter.matched_term("Technical Program Manager", "You will run RLHF data programs.") == "RLHF"
    assert prefilter.matched_term("Technical Program Manager", "You will run compute programs.") is None


def test_every_open_posting_of_a_job_is_judged_on_its_own_text(session: Session, company: Company) -> None:
    """Two siblings share a job but carry different text: two calls, two rows and the job takes the best."""
    adapter = FakeAdapter.returning(
        "greenhouse",
        make_posting("1", "Research Scientist", description="Runs RLHF experiments."),
        make_posting("2", "Research Scientist", description="Coordinates evaluation programs."),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    judge, _ = _judge(lambda request: verdict(90 if "RLHF" in request["messages"][0]["content"] else 40))

    report = _score_all(session, judge)

    assert report.judged == 2 and report.scored == 2
    scores = {s.source.source_job_id: s.score for s in session.scalars(select(Score))}
    assert scores == {"1": 90, "2": 40}
    (job_id,) = set(session.scalars(select(JobSource.job_id)))
    assert best_scores(session, 1)[job_id].score == 90


def test_identical_sibling_texts_share_one_call(session: Session, company: Company) -> None:
    """The same text from two boards is judged once and both postings get a row."""
    ingest_company(session, company, FakeAdapter.returning("greenhouse", make_posting("g1")), NOW)
    ingest_company(session, company, FakeAdapter.returning("ashby", make_posting("a1", source="ashby")), NOW)
    session.commit()
    judge, client = _judge(verdict(75))

    report = _score_all(session, judge)

    assert len(report.candidates) == 2 and report.distinct_texts == 1
    assert report.judged == 1 and report.scored == 2 and len(client.requests) == 1
    assert [s.score for s in session.scalars(select(Score))] == [75, 75]


def test_a_judged_posting_is_not_judged_again(session: Session, company: Company) -> None:
    """Running twice costs one call. This is the cost-discipline gate."""
    ingest_company(session, company, FakeAdapter.returning("greenhouse", make_posting("1")), NOW)
    session.commit()
    judge, client = _judge()

    first = _score_all(session, judge)
    second = _score_all(session, judge)

    assert first.judged == 1
    assert second.judged == 0 and second.already_judged == 1 and second.candidates == []
    assert len(client.requests) == 1


def test_a_reworded_posting_is_judged_again_and_only_the_new_verdict_counts(
    session: Session, company: Company
) -> None:
    """A text change triggers one more call and retires the old verdict from `best_scores`."""
    adapter = FakeAdapter(
        "greenhouse",
        ok(make_posting("1", description="v1 with RLHF")),
        ok(make_posting("1", description="v2 with RLHF and DPO")),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    judge, _ = _judge(verdict(60), verdict(85))
    _score_all(session, judge)

    ingest_company(session, company, adapter, NOW + timedelta(hours=2))
    session.commit()
    report = _score_all(session, judge)

    assert report.judged == 1
    assert [s.score for s in session.scalars(select(Score).order_by(Score.id))] == [60, 85]
    (winner,) = best_scores(session, 1).values()
    assert winner.score == 85


def test_the_cap_takes_the_oldest_posting_first_and_carries_the_rest(
    session: Session, company: Company
) -> None:
    """Oldest-unscored first, so a posting at the back of the queue cannot be starved forever."""
    a = make_posting("a", "Research Scientist, Alignment")
    b = make_posting("b", "Research Engineer, Inference")
    c = make_posting("c", "Post-Training Researcher")
    adapter = FakeAdapter("greenhouse", ok(a), ok(a, b), ok(a, b, c))
    for day in range(3):
        ingest_company(session, company, adapter, NOW + timedelta(days=day))
    session.commit()
    judge, client = _judge()

    report = _score_all(session, judge, limit=2)

    assert report.judged == 2 and report.carried_over == 1 and report.cap == 2
    assert [_title_of(r) for r in client.requests] == [
        "Title: Research Scientist, Alignment",
        "Title: Research Engineer, Inference",
    ]


def test_eliminated_unmatched_and_closed_postings_are_never_candidates(
    session: Session, company: Company
) -> None:
    """Each prefilter counter fires for the right posting and a closed posting drops out."""
    adapter = FakeAdapter(
        "greenhouse",
        ok(
            make_posting("1", "Research Scientist"),
            make_posting("2", "Account Executive"),
            make_posting("3", "Research Scientist", location="Tokyo, Japan"),
            make_posting("4", "Software Engineer, Compute", description="We run GPU clusters."),
        ),
        ok(
            make_posting("2", "Account Executive"),
            make_posting("3", "Research Scientist", location="Tokyo, Japan"),
            make_posting("4", "Software Engineer, Compute", description="We run GPU clusters."),
        ),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    report = ScoringReport()
    candidates = select_candidates(session, PROFILE, 1, report)
    assert [c.text.title for c in candidates] == ["Research Scientist"]
    assert (report.eliminated_title, report.eliminated_location, report.unmatched) == (1, 1, 1)

    ingest_company(session, company, adapter, NOW + timedelta(hours=2))  # posting 1 closes
    session.commit()
    report = ScoringReport()
    assert select_candidates(session, PROFILE, 1, report) == []
    assert report.open_postings == 3


def test_a_fatal_api_error_stops_the_run_before_it_burns_the_cap(
    session: Session, company: Company
) -> None:
    """A rejected key fails every call the same way, so the run stops after the first."""
    adapter = FakeAdapter.returning(
        "greenhouse",
        make_posting("1", "Research Scientist, Alignment"),
        make_posting("2", "Research Engineer, Inference"),
        make_posting("3", "Post-Training Researcher"),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    judge, client = _judge(api_error(anthropic.AuthenticationError, 401, "invalid x-api-key"))

    report = _score_all(session, judge)

    assert report.aborted and "AuthenticationError" in report.aborted
    assert len(client.requests) == 1 and report.failed == 1 and report.judged == 0
    assert session.scalar(select(Score)) is None


def test_a_transient_error_skips_one_text_and_carries_on(session: Session, company: Company) -> None:
    """A rate limit on one posting costs that posting a run and not the whole run."""
    adapter = FakeAdapter.returning(
        "greenhouse",
        make_posting("1", "Research Scientist, Alignment"),
        make_posting("2", "Research Engineer, Inference"),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    judge, _ = _judge(api_error(anthropic.RateLimitError, 429), verdict(70))

    report = _score_all(session, judge)

    assert report.failed == 1 and report.judged == 1 and report.aborted is None
    assert session.scalar(select(func.count()).select_from(Score)) == 1


def test_stale_text_is_skipped_until_ingest_recomputes_its_hash(session: Session, company: Company) -> None:
    """A posting whose stored hash no longer matches its text is left for ingest and not judged."""
    ingest_company(session, company, FakeAdapter.returning("greenhouse", make_posting("1")), NOW)
    session.commit()
    source = session.scalar(select(JobSource))
    source.content_hash = "0" * 64
    session.commit()

    report = ScoringReport()
    assert select_candidates(session, PROFILE, 1, report) == []
    assert report.stale == 1


def test_dry_run_calls_nothing_and_writes_nothing(session: Session, company: Company) -> None:
    """`score --dry-run` reports the prefilter and stops, so it needs no key and costs nothing."""
    ingest_company(session, company, FakeAdapter.returning("greenhouse", make_posting("1")), NOW)
    session.commit()

    report = run_scoring(dry_run=True)

    assert len(report.candidates) == 1 and report.judged == 0
    assert session.scalar(select(Score)) is None


def test_run_scoring_is_idempotent_end_to_end(session: Session, company: Company) -> None:
    """Through the entry point the command uses, with the repository's real search profile."""
    ingest_company(session, company, FakeAdapter.returning("greenhouse", make_posting("1")), NOW)
    session.commit()
    judge, client = _judge(verdict(88))

    first = run_scoring(judge=judge)
    second = run_scoring(judge=judge)

    assert first.scored == 1 and second.judged == 0
    assert len(client.requests) == 1


def test_the_request_carries_one_cached_stable_prefix(session: Session, company: Company) -> None:
    """The system block is marked for caching and identical across calls. Only the posting varies."""
    adapter = FakeAdapter.returning(
        "greenhouse",
        make_posting("1", "Research Scientist, Alignment"),
        make_posting("2", "Research Engineer, Inference"),
    )
    ingest_company(session, company, adapter, NOW)
    session.commit()
    judge, client = _judge()

    report = _score_all(session, judge)

    first, second = client.requests
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["system"] == second["system"], "a prefix that varies can never be read from cache"
    assert first["model"] == "claude-haiku-4-5" and first["output_format"] is Verdict
    assert first["messages"][0]["content"] != second["messages"][0]["content"]
    assert report.cold_calls == 0
    assert report.model == "claude-haiku-4-5", "so a cache warning can name the model in use"


def test_a_score_row_records_fit_model_version_and_the_text_it_judged(
    session: Session, company: Company
) -> None:
    """Everything the digest and the rescoring rule need is on the row."""
    ingest_company(session, company, FakeAdapter.returning("greenhouse", make_posting("1")), NOW)
    session.commit()
    judge, _ = _judge(verdict(91, work_authorization="unclear"))

    _score_all(session, judge)

    row = session.scalar(select(Score))
    source = session.scalar(select(JobSource))
    assert (row.location_fit, row.work_authorization) == (LocationFit.PRIORITY, "unclear")
    assert (row.model, row.prompt_version) == ("claude-haiku-4-5", 1)
    assert row.content_hash == source.content_hash and row.source_id == source.id


def test_best_scores_ignores_closed_postings(session: Session, company: Company) -> None:
    """When the winning posting closes, the job's score falls back to its open sibling."""
    strong = make_posting("1", "Research Scientist", description="Runs RLHF experiments.")
    weak = make_posting("2", "Research Scientist", description="Coordinates evaluation programs.")
    adapter = FakeAdapter("greenhouse", ok(strong, weak), ok(weak))
    ingest_company(session, company, adapter, NOW)
    session.commit()
    judge, _ = _judge(lambda request: verdict(90 if "RLHF" in request["messages"][0]["content"] else 40))
    _score_all(session, judge)
    (job_id,) = set(session.scalars(select(JobSource.job_id)))
    assert best_scores(session, 1)[job_id].score == 90

    ingest_company(session, company, adapter, NOW + timedelta(hours=2))  # the strong one closes
    session.commit()
    assert best_scores(session, 1)[job_id].score == 40
