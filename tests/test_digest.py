"""Tests for the daily email.

Nothing here opens a socket. `RecordingSender` keeps the message instead
of delivering it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import FakeAdapter, make_posting
from job_hunters import digest as digest_module
from job_hunters.actions import ExpiredToken, verify
from job_hunters.config import AppConfig, Secrets, SearchProfile, SystemConfig
from job_hunters.digest import (
    appearances_since_change,
    build_digest,
    record_appearances,
    render,
    render_bodies,
    render_message,
    run_digest,
)
from job_hunters.ingest import ingest_company
from job_hunters.mailer import RecordingSender
from job_hunters.models import (
    Application,
    ApplicationStatus,
    Company,
    DigestAppearance,
    DigestSection,
    Job,
    JobSource,
    LocationFit,
    Score,
)

NOW = datetime(2026, 9, 9, 6, 0, tzinfo=UTC)
TODAY = date(2026, 9, 9)
SECRET = "an-action-token-secret"

PROFILE_DICT = {
    "titles": {"include": ["research scientist"], "exclude": ["sales"]},
    "keywords": {"strong": ["RLHF", "post-training"], "supporting": ["evaluation"]},
    "seniority": {"include": ["senior"], "exclude": ["intern"]},
    "location": {
        "base": "portugal",
        "priority": [{"work_modes": ["onsite", "hybrid"], "regions": ["portugal"]}],
        "acceptable": [{"work_modes": ["remote"], "regions": ["us"]}],
        "work_authorization": {"have": ["eu"], "need_sponsorship": ["us"]},
    },
    "scoring": {
        "threshold": 70, "max_llm_scores_per_run": 300, "prompt_version": 1,
        "rubric": "Weight post-training.",
        "bands": [{"low": 70, "high": 100, "meaning": "A fit."},
                  {"low": 0, "high": 69, "meaning": "Not a fit."}],
    },
}

SYSTEM_DICT = {
    "timezone": "Europe/Lisbon",
    "base_url": "http://localhost:8000",
    "digest": {
        "repeat_suppression": {
            "enabled": True, "demote_after": 2, "suppress_after": 3,
            "reset_on_score_delta": 10,
        }
    },
}


def _config(profile: dict | None = None, system: dict | None = None) -> AppConfig:
    """An AppConfig built from literals, so a test can vary one rule at a time."""
    return AppConfig(
        search_profile=SearchProfile.model_validate(profile or PROFILE_DICT),
        system=SystemConfig.model_validate(system or SYSTEM_DICT),
        watchlist=[],
        secrets=Secrets(_env_file=None),
    )


def _ingest(session: Session, company: Company, *postings, now: datetime = NOW) -> None:
    """Puts postings into the database through the real ingest path."""
    ingest_company(session, company, FakeAdapter.returning("greenhouse", *postings), now)
    session.commit()


def _judge(session: Session, source: JobSource, score: int = 90, **overrides) -> Score:
    """A stored judgement of one posting at that posting's current text."""
    fields = dict(
        source_id=source.id, score=score, summary="A post-training role. It fits.",
        rationale="Because.", matched_areas=["RLHF"], concerns=[],
        work_authorization="eligible", location_fit=LocationFit.PRIORITY,
        prompt_version=1, model="claude-haiku-4-5", content_hash=source.content_hash,
    )
    fields.update(overrides)
    row = Score(**fields)
    session.add(row)
    session.commit()
    return row


def _one_job(session: Session, company: Company, score: int = 90, **overrides) -> Job:
    """One ingested posting in Lisbon, judged at `score`. The common setup."""
    _ingest(session, company, make_posting("1", "Research Scientist", location="Lisbon, Portugal"))
    _judge(session, session.scalar(select(JobSource)), score, **overrides)
    return session.scalar(select(Job))


def _titles(digest, section_key: str) -> list[str]:
    """The entry titles in one section in the order they are rendered."""
    for section in digest.sections:
        if section.spec.key == section_key:
            return [entry.title for entry in section.entries]
    return []


def test_a_job_above_the_threshold_reaches_the_digest(session: Session, company: Company) -> None:
    """The ordinary case: judged well and location matches a priority rule, so it is shown."""
    _one_job(session, company, 90)
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert _titles(digest, DigestSection.PRIORITY) == ["Research Scientist"]
    assert digest.total == 1


def test_a_job_below_the_threshold_is_not_shown(session: Session, company: Company) -> None:
    """Scored and stored, but never emailed. That is what the threshold is for."""
    _one_job(session, company, 69)
    assert build_digest(session, _config(), SECRET, today=TODAY, now=NOW).is_empty


def test_an_applied_job_does_not_come_back(session: Session, company: Company) -> None:
    """Acting on a job is what removes it from the digest."""
    job = _one_job(session, company, 90)
    session.add(Application(job_id=job.id, status=ApplicationStatus.APPLIED))
    session.commit()
    assert build_digest(session, _config(), SECRET, today=TODAY, now=NOW).is_empty


def test_an_untouched_application_row_does_not_hide_a_job(session: Session, company: Company) -> None:
    """`interested` is the status a row is created with and means nothing has happened yet."""
    job = _one_job(session, company, 90)
    session.add(Application(job_id=job.id, status=ApplicationStatus.INTERESTED))
    session.commit()
    assert build_digest(session, _config(), SECRET, today=TODAY, now=NOW).total == 1


def test_the_location_rules_are_read_now_and_not_when_the_score_was_written(
    session: Session, company: Company
) -> None:
    """Narrowing the rules must take effect immediately, with no rescore to trigger it."""
    _one_job(session, company, 90, location_fit=LocationFit.PRIORITY)
    narrowed = {**PROFILE_DICT, "location": {**PROFILE_DICT["location"],
                "priority": [{"work_modes": ["onsite"], "regions": ["switzerland"]}],
                "acceptable": []}}
    assert build_digest(session, _config(narrowed), SECRET, today=TODAY, now=NOW).is_empty


def test_an_acceptable_location_lands_in_its_own_section(session: Session, company: Company) -> None:
    """A remote US role is acceptable rather than priority and the sections say so."""
    _ingest(session, company, make_posting("1", "Research Scientist", location="Remote - US"))
    _judge(session, session.scalar(select(JobSource)))
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert _titles(digest, DigestSection.ACCEPTABLE) == ["Research Scientist"]
    assert _titles(digest, DigestSection.PRIORITY) == []


def test_an_unparsed_location_is_worth_checking_and_never_dropped(
    session: Session, company: Company
) -> None:
    """An unknown location is the parser admitting it does not know and not a reason to hide a job."""
    _ingest(session, company, make_posting("1", "Research Scientist", location="Somewhere"))
    _judge(session, session.scalar(select(JobSource)))
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert _titles(digest, DigestSection.WORTH_CHECKING) == ["Research Scientist"]


def test_an_unclear_work_authorization_moves_a_job_to_worth_checking(
    session: Session, company: Company
) -> None:
    """A posting that may contradict the declared status is surfaced rather than trusted."""
    _one_job(session, company, 90, work_authorization="unclear")
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert _titles(digest, DigestSection.WORTH_CHECKING) == ["Research Scientist"]
    assert _titles(digest, DigestSection.PRIORITY) == []


def test_a_blocked_job_is_shown_rather_than_silently_dropped(
    session: Session, company: Company
) -> None:
    """`blocked` is the judge reading one document and it can be wrong. Nothing is hidden."""
    _one_job(session, company, 90, work_authorization="blocked")
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert _titles(digest, DigestSection.WORTH_CHECKING) == ["Research Scientist"]


def test_entries_are_ordered_by_score_within_a_section(session: Session, company: Company) -> None:
    """The best thing you have not seen yet should be the first thing you read."""
    _ingest(
        session, company,
        make_posting("1", "Research Scientist, Evals", location="Lisbon, Portugal"),
        make_posting("2", "Research Scientist, RLHF", location="Lisbon, Portugal"),
    )
    sources = session.scalars(select(JobSource).order_by(JobSource.source_job_id)).all()
    _judge(session, sources[0], 75)
    _judge(session, sources[1], 95)
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert _titles(digest, DigestSection.PRIORITY) == [
        "Research Scientist, RLHF", "Research Scientist, Evals",
    ]


def test_the_entry_links_to_the_posting_that_won_and_not_the_one_on_display(
    session: Session, company: Company
) -> None:
    """The summary was written about one posting's text, so the link must open that text."""
    _ingest(
        session, company,
        make_posting("1", "Research Scientist", location="Lisbon, Portugal",
                     description="Platform work.", url="https://example.test/primary"),
        make_posting("2", "Research Scientist", location="Lisbon, Portugal",
                     description="RLHF and post-training.", url="https://example.test/sibling"),
    )
    job = session.scalar(select(Job))
    assert len(job.sources) == 2, "same company, title and city: one job, two postings"
    sources = session.scalars(select(JobSource).order_by(JobSource.source_job_id)).all()
    _judge(session, sources[0], 71)
    _judge(session, sources[1], 95)

    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.score == 95
    assert entry.apply_url == "https://example.test/sibling"


def _shown_on(session: Session, job: Job, days: list[int], score: int = 90) -> None:
    """Records this job as already having appeared on the given days of September."""
    for day in days:
        session.add(
            DigestAppearance(
                job_id=job.id, digest_date=date(2026, 9, day),
                section=DigestSection.PRIORITY, position=1,
                score_at_appearance=score, content_hash_at_appearance=job.content_hash,
            )
        )
    session.commit()


def test_a_job_is_left_alone_until_demote_after_is_passed(
    session: Session, company: Company
) -> None:
    """With demote_after 2, the first two showings are ordinary entries."""
    job = _one_job(session, company, 90)
    _shown_on(session, job, [7])
    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.appearance == 2 and not entry.demoted


def test_a_job_shown_too_often_is_demoted_with_a_nudge(
    session: Session, company: Company
) -> None:
    """The third showing is demoted and says so, which is how you know to dismiss it."""
    job = _one_job(session, company, 90)
    _shown_on(session, job, [6, 7])
    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.appearance == 3 and entry.demoted
    assert entry.repeat_note == "3rd time — dismiss?"


def test_demoted_entries_sink_below_the_rest_of_their_section(
    session: Session, company: Company
) -> None:
    """A demoted entry loses its place even when it scores higher than a new one."""
    _ingest(
        session, company,
        make_posting("1", "Research Scientist, Old", location="Lisbon, Portugal"),
        make_posting("2", "Research Scientist, New", location="Lisbon, Portugal"),
    )
    sources = session.scalars(select(JobSource).order_by(JobSource.source_job_id)).all()
    _judge(session, sources[0], 99)
    _judge(session, sources[1], 71)
    old = session.get(JobSource, sources[0].id).job
    _shown_on(session, old, [5, 6], score=99)

    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert _titles(digest, DigestSection.PRIORITY) == [
        "Research Scientist, New", "Research Scientist, Old",
    ]


def test_a_job_shown_past_suppress_after_is_folded_into_a_count(
    session: Session, company: Company
) -> None:
    """The fourth showing leaves the body entirely and becomes part of "Still Open"."""
    job = _one_job(session, company, 90)
    _shown_on(session, job, [5, 6, 7])
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert digest.is_empty
    assert digest.still_open == 1


def test_suppression_can_be_turned_off_entirely(session: Session, company: Company) -> None:
    """`enabled: false` means a job keeps its place however many times it has been shown."""
    job = _one_job(session, company, 90)
    _shown_on(session, job, [3, 4, 5, 6, 7])
    system = {**SYSTEM_DICT, "digest": {"repeat_suppression": {"enabled": False}}}
    entry = build_digest(session, _config(system=system), SECRET, today=TODAY, now=NOW).entries[0]
    assert not entry.demoted


def test_a_moved_score_resets_the_counter(session: Session, company: Company) -> None:
    """A rescore that changes the verdict makes this a different job to read."""
    job = _one_job(session, company, 90)
    _shown_on(session, job, [5, 6, 7], score=70)
    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.appearance == 1, "20 points is more than reset_on_score_delta"
    assert not entry.demoted


def test_a_rewritten_posting_resets_the_counter_even_at_the_same_score(
    session: Session, company: Company
) -> None:
    """A rewrite can be rescored to the same number, so the score alone cannot see one."""
    job = _one_job(session, company, 90)
    _shown_on(session, job, [5, 6, 7], score=90)
    job.content_hash = "f" * 64
    session.commit()
    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.appearance == 1


def test_the_counter_stops_at_the_oldest_appearance_that_still_describes_the_job() -> None:
    """Appearances older than a material change describe the previous version and do not count."""
    def appearance(score: int, digest_hash: str) -> DigestAppearance:
        return DigestAppearance(score_at_appearance=score, content_hash_at_appearance=digest_hash)

    from job_hunters.config import RepeatSuppressionConfig

    history = [appearance(90, "a"), appearance(90, "a"), appearance(50, "a"), appearance(90, "a")]
    suppression = RepeatSuppressionConfig(reset_on_score_delta=10)
    assert appearances_since_change(history, 90, "a", suppression) == 2


def test_sending_records_one_appearance_per_entry(session: Session, company: Company) -> None:
    """The record written today is what suppression reads tomorrow."""
    _one_job(session, company, 90)
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert record_appearances(session, digest) == 1
    session.commit()

    row = session.scalar(select(DigestAppearance))
    assert row.digest_date == TODAY
    assert row.section == DigestSection.PRIORITY
    assert row.position == 1
    assert row.score_at_appearance == 90
    assert row.content_hash_at_appearance == session.scalar(select(Job)).content_hash


def test_a_second_send_on_one_day_is_not_a_second_appearance(
    session: Session, company: Company
) -> None:
    """Re-running the digest must not push a job towards suppression twice in a day."""
    _one_job(session, company, 90)
    for _ in range(2):
        digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
        record_appearances(session, digest)
        session.commit()

    rows = session.scalars(select(DigestAppearance)).all()
    assert len(rows) == 1, "one row per job per day, replaced rather than added to"
    assert digest.entries[0].appearance == 1


def test_a_suppressed_job_gets_no_appearance_row(session: Session, company: Company) -> None:
    """It was shown as a number and not as an entry, so its counter stays where it is."""
    job = _one_job(session, company, 90)
    _shown_on(session, job, [5, 6, 7])
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    record_appearances(session, digest)
    session.commit()
    assert len(session.scalars(select(DigestAppearance)).all()) == 3


def test_the_link_lifetime_comes_from_the_config_and_is_not_hardcoded(
    session: Session, company: Company
) -> None:
    """`actions.token_ttl_days` is the only place the lifetime is decided."""
    _one_job(session, company, 90)
    system = {**SYSTEM_DICT, "actions": {"token_ttl_days": 3}}
    digest = build_digest(session, _config(system=system), SECRET, today=TODAY, now=NOW)

    token = digest.entries[0].links[0].url.rsplit("/", 1)[1]
    assert verify(SECRET, token, now=NOW + timedelta(days=2)).job_id is not None
    with pytest.raises(ExpiredToken):
        verify(SECRET, token, now=NOW + timedelta(days=4))


def test_a_multi_office_job_says_which_office_earned_its_section(
    session: Session, company: Company
) -> None:
    """A job with multiple offices says which one matched the rules and which one is on display."""
    _ingest(session, company, make_posting("1", "Research Scientist",
                                           location="London, UK; New York, US"))
    job = session.scalar(select(Job))
    job.region, job.regions, job.work_mode = "uk", ["uk", "us"], "remote"
    session.commit()
    _judge(session, session.scalar(select(JobSource)))

    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.region == "uk"
    assert entry.matched_place == "us"
    assert entry.section_note == "acceptable via us"
    assert "acceptable via us" in render(
        build_digest(session, _config(), SECRET, today=TODAY, now=NOW), "digest.html"
    )


def test_a_single_office_job_says_nothing_extra(session: Session, company: Company) -> None:
    """A note naming the place that already matches the displayed region would be noise, so it is not printed."""
    _one_job(session, company, 90)
    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.matched_place == entry.region
    assert entry.section_note == ""


def test_an_unparsed_location_says_that_is_why_it_is_worth_checking(
    session: Session, company: Company
) -> None:
    """The section names two possible reasons, so the entry has to say which one applies."""
    _ingest(session, company, make_posting("1", "Research Scientist", location="Somewhere"))
    _judge(session, session.scalar(select(JobSource)))
    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.section_note == "location could not be settled"


def test_a_job_held_back_only_by_its_authorization_says_the_location_was_fine(
    session: Session, company: Company
) -> None:
    """A job in "Worth Checking" only for its authorization must not read as a location the parser fumbled."""
    _one_job(session, company, 90, work_authorization="unclear")
    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.section_note == "the location qualifies; the work authorization does not"


def test_that_answer_still_names_the_office_when_it_is_not_the_obvious_one(
    session: Session, company: Company
) -> None:
    """Both questions can apply at once and answering only one of them is worse than neither."""
    _ingest(session, company, make_posting("1", "Research Scientist", location="London"))
    job = session.scalar(select(Job))
    job.region, job.regions, job.work_mode = "uk", ["uk", "us"], "remote"
    session.commit()
    _judge(session, session.scalar(select(JobSource)), work_authorization="blocked")

    entry = build_digest(session, _config(), SECRET, today=TODAY, now=NOW).entries[0]
    assert entry.section_note == (
        "the location qualifies (via us); the work authorization does not"
    )


def test_every_worth_checking_entry_explains_itself(
    session: Session, company: Company
) -> None:
    """Whatever routed a job here. The reader is never left to guess which of the two it was."""
    _ingest(
        session, company,
        make_posting("1", "Research Scientist, One", location="Somewhere"),
        make_posting("2", "Research Scientist, Two", location="Lisbon, Portugal"),
    )
    sources = session.scalars(select(JobSource).order_by(JobSource.source_job_id)).all()
    _judge(session, sources[0])
    _judge(session, sources[1], work_authorization="blocked")

    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    checking = [s for s in digest.sections if s.spec.key == DigestSection.WORTH_CHECKING][0]
    assert len(checking.entries) == 2
    assert all(entry.section_note for entry in checking.entries)


def test_both_bodies_are_rendered_once_and_reused(session: Session, company: Company) -> None:
    """The preview must be the email that was sent and not a second render of the same digest."""
    _one_job(session, company, 90)
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    text, html = render_bodies(digest)
    message = render_message(digest, "me@example.com", "bot@example.com", (text, html))
    assert message.text is text and message.html is html, "handed over, not re-rendered"


def test_a_dry_run_renders_the_plain_text_body_too(
    session: Session, company: Company, monkeypatch
) -> None:
    """A broken text template must fail the preview and not wait for the one real send."""
    monkeypatch.setenv("ACTION_TOKEN_SECRET", SECRET)
    _one_job(session, company, 90)

    rendered: list[str] = []
    real_render = digest_module.render

    def recording(digest, template):
        """The real renderer with a note of which templates it was asked for."""
        rendered.append(template)
        return real_render(digest, template)

    monkeypatch.setattr(digest_module, "render", recording)
    run_digest(dry_run=True, config=_config())
    assert sorted(rendered) == ["digest.html", "digest.txt"]


def test_the_html_carries_the_entry_and_four_signed_links(
    session: Session, company: Company
) -> None:
    """Every link in the email has to verify against the secret that signed it."""
    job = _one_job(session, company, 90)
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    html = render(digest, "digest.html")

    assert "Research Scientist" in html and "Acme" in html
    entry = digest.entries[0]
    assert len(entry.links) == 4
    for link in entry.links:
        assert link.url.startswith("http://localhost:8000/a/")
        assert verify(SECRET, link.url.rsplit("/", 1)[1], now=NOW).job_id == job.id
        assert link.url in html


def test_the_plain_text_alternative_carries_the_same_entries(
    session: Session, company: Company
) -> None:
    """A text-only client, a screen reader and most spam filters read this one."""
    _one_job(session, company, 90)
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    text = render(digest, "digest.txt")
    assert "Research Scientist" in text and "PRIORITY" in text
    assert "<div" not in text


def test_html_in_a_posting_cannot_break_out_of_the_template(
    session: Session, company: Company
) -> None:
    """A board's text ends up in an email, so it is escaped rather than trusted."""
    _one_job(session, company, 90, summary="<script>alert(1)</script> Fits well.")
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    html = render(digest, "digest.html")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_an_empty_digest_still_renders_and_says_so(session: Session, company: Company) -> None:
    """Silence would be indistinguishable from a broken pipeline, so an empty day is sent."""
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert digest.is_empty
    assert "nothing new" in digest.subject()
    assert "No new roles" in render(digest, "digest.html")
    assert "No new roles" in render(digest, "digest.txt")


def test_the_subject_says_how_much_is_in_it(session: Session, company: Company) -> None:
    """Most of what gets read on a phone is the subject line."""
    _one_job(session, company, 90)
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    assert digest.subject() == "Job-Hunters Wed 09 Sep: 1 role, 1 priority"


def test_the_message_is_addressed_and_carries_both_bodies(
    session: Session, company: Company
) -> None:
    """A `multipart/alternative` with only HTML in it is itself a spam signal."""
    _one_job(session, company, 90)
    digest = build_digest(session, _config(), SECRET, today=TODAY, now=NOW)
    message = render_message(digest, "me@example.com", "bot@example.com")
    assert message.to == "me@example.com" and message.sender == "bot@example.com"
    assert message.text and message.html


def test_run_digest_sends_and_records(session: Session, company: Company, monkeypatch) -> None:
    """The end to end path, with the mail server replaced by a recorder."""
    monkeypatch.setenv("ACTION_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("DIGEST_TO", "me@example.com")
    _one_job(session, company, 90)
    config = _config()

    sender = RecordingSender()
    report = run_digest(sender=sender, config=config)

    assert report.sent_to == "me@example.com"
    assert report.recorded == 1
    assert len(sender.sent) == 1
    assert "Research Scientist" in sender.sent[0].html
    assert session.scalar(select(DigestAppearance)) is not None


def test_a_dry_run_sends_nothing_and_records_nothing(
    session: Session, company: Company, monkeypatch
) -> None:
    """A preview must not consume an appearance or previewing would suppress jobs."""
    monkeypatch.setenv("ACTION_TOKEN_SECRET", SECRET)
    _one_job(session, company, 90)
    config = _config()

    report = run_digest(dry_run=True, config=config)

    assert report.sent_to is None and report.recorded == 0
    assert "Research Scientist" in report.html
    assert session.scalar(select(DigestAppearance)) is None
