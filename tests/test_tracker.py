"""Tests for what an action link does and for what the dashboard tracks."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import FakeAdapter, make_posting
from job_hunters.actions import Action
from job_hunters.config import AppConfig, Secrets, SearchProfile, SystemConfig
from job_hunters.digest import build_digest
from job_hunters.ingest import ingest_company
from job_hunters.models import (
    Application,
    ApplicationEvent,
    ApplicationStatus,
    Company,
    DigestAppearance,
    DigestSection,
    EventKind,
    Job,
    JobSource,
    LocationFit,
    Score,
)
from job_hunters.sources import replay_posting
from job_hunters.tracker import (
    UnknownJob,
    build_dashboard,
    can_confirm,
    job_card,
    perform,
)

NOW = datetime(2026, 9, 9, 6, 0, tzinfo=UTC)
TODAY = date(2026, 9, 9)
SECRET = "an-action-token-secret"

PROFILE_DICT = {
    "titles": {"include": ["research scientist"], "exclude": ["sales"]},
    "keywords": {"strong": ["RLHF", "post-training"], "supporting": ["evaluation"]},
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


def _profile() -> SearchProfile:
    """The search profile these tests judge against. Built from literals."""
    return SearchProfile.model_validate(PROFILE_DICT)


def _config() -> AppConfig:
    """An AppConfig for the few tests that build a digest to compare against."""
    return AppConfig(
        search_profile=_profile(),
        system=SystemConfig.model_validate({"timezone": "Europe/Lisbon"}),
        watchlist=[],
        secrets=Secrets(_env_file=None),
    )


def _job(
    session: Session, company: Company, source_job_id: str = "1",
    title: str = "Research Scientist", *, score: int = 90, source: str = "greenhouse",
) -> Job:
    """One more ingested posting on one board (judged) as the job it deduplicated to.

    Whatever this board has already returned is handed back with it. A real
    fetch returns every open posting at once and ingest closes anything a
    successful fetch left out. So, building a second job by fetching only the
    second posting would quietly close the first.
    """
    already = [
        replay_posting(row.source, row.raw_json)
        for row in session.scalars(select(JobSource).where(JobSource.source == source))
    ]
    ingest_company(
        session, company,
        FakeAdapter.returning(
            source, *already,
            make_posting(source_job_id, title, source=source, location="Lisbon, Portugal"),
        ),
        NOW,
    )
    session.commit()
    posting = session.scalar(
        select(JobSource).where(JobSource.source_job_id == source_job_id)
    )
    session.add(
        Score(
            source_id=posting.id, score=score, summary="A post-training role. It fits.",
            rationale="Because.", matched_areas=["RLHF"], concerns=[],
            work_authorization="eligible", location_fit=LocationFit.PRIORITY,
            prompt_version=1, model="claude-haiku-4-5", content_hash=posting.content_hash,
        )
    )
    session.commit()
    return session.get(Job, posting.job_id)


def _applications(session: Session) -> list[Application]:
    """Every application row so a test can assert there is exactly one."""
    return list(session.scalars(select(Application)))


def _events(session: Session) -> list[ApplicationEvent]:
    """Every event row in the order they were written."""
    return list(session.scalars(select(ApplicationEvent).order_by(ApplicationEvent.id)))


def test_applying_records_the_application_and_opens_its_timeline(
    session: Session, company: Company
) -> None:
    """One click has to produce both the row and the first entry of its history."""
    job = _job(session, company)
    outcome = perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()

    assert outcome.changed
    application = _applications(session)[0]
    assert application.status == ApplicationStatus.APPLIED
    assert application.applied_at is not None
    assert [event.event for event in _events(session)] == [EventKind.APPLIED]


def test_applying_twice_is_a_no_op(session: Session, company: Company) -> None:
    """An action is a no-op if performed twice."""
    job = _job(session, company)
    perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    second = perform(session, Action.APPLIED, job.id, now=NOW + timedelta(days=7))
    session.commit()

    assert not second.changed
    assert len(_applications(session)) == 1, "one application per job, not two"
    assert len(_events(session)) == 1, "and one `applied` event, not two"


def test_the_recorded_date_is_not_moved_by_a_second_click(
    session: Session, company: Company
) -> None:
    """A link clicked again in December must not say you applied in December."""
    job = _job(session, company)
    perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    perform(session, Action.APPLIED, job.id, now=NOW + timedelta(days=90))
    session.commit()

    applied_at = _applications(session)[0].applied_at
    assert applied_at.replace(tzinfo=UTC) == NOW


def test_an_applied_job_does_not_come_back_in_the_next_digest(
    session: Session, company: Company
) -> None:
    """Once you have clicked the applied link, the digest does not ask you to click it again."""
    job = _job(session, company)
    assert build_digest(session, _config(), SECRET, today=TODAY, now=NOW).total == 1

    perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()

    assert build_digest(session, _config(), SECRET, today=TODAY, now=NOW).is_empty


def test_dismissing_takes_a_job_out_of_the_digest(
    session: Session, company: Company
) -> None:
    """Once you have clicked the dismiss link, the digest does not ask you to click it again."""
    job = _job(session, company)
    outcome = perform(session, Action.DISMISS, job.id, now=NOW)
    session.commit()

    assert outcome.changed
    assert _applications(session)[0].status == ApplicationStatus.DISMISSED
    assert build_digest(session, _config(), SECRET, today=TODAY, now=NOW).is_empty


def test_dismissing_writes_no_timeline_event(session: Session, company: Company) -> None:
    """`application_events` is the hiring timeline and a dismissal never entered one."""
    job = _job(session, company)
    perform(session, Action.DISMISS, job.id, now=NOW)
    session.commit()
    assert _events(session) == []


def test_dismissing_twice_is_a_no_op(session: Session, company: Company) -> None:
    """Every action link has to survive being clicked again, not only "Applied"."""
    job = _job(session, company)
    perform(session, Action.DISMISS, job.id, now=NOW)
    session.commit()
    second = perform(session, Action.DISMISS, job.id, now=NOW)
    session.commit()

    assert not second.changed
    assert len(_applications(session)) == 1


def test_dismissing_a_job_you_applied_to_is_refused(
    session: Session, company: Company
) -> None:
    """It is already out of the digest, so the only thing the write could do is lose the record."""
    job = _job(session, company)
    perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    outcome = perform(session, Action.DISMISS, job.id, now=NOW)
    session.commit()

    assert not outcome.changed
    application = _applications(session)[0]
    assert application.status == ApplicationStatus.APPLIED
    assert application.applied_at is not None, "the date survived the refusal"


def test_applying_to_a_job_you_dismissed_promotes_it(
    session: Session, company: Company
) -> None:
    """Applying to a job you dismissed loses nothing, so it is allowed."""
    job = _job(session, company)
    perform(session, Action.DISMISS, job.id, now=NOW)
    session.commit()
    outcome = perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()

    assert outcome.changed
    assert _applications(session)[0].status == ApplicationStatus.APPLIED
    assert len(_events(session)) == 1


def test_an_interested_row_is_promoted_rather_than_duplicated(
    session: Session, company: Company
) -> None:
    """`job_id` is unique, so a second row would be an IntegrityError rather than a bug you see."""
    job = _job(session, company)
    session.add(Application(job_id=job.id, status=ApplicationStatus.INTERESTED))
    session.commit()
    existing = _applications(session)[0].id

    perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()

    rows = _applications(session)
    assert len(rows) == 1 and rows[0].id == existing
    assert rows[0].status == ApplicationStatus.APPLIED


@pytest.mark.parametrize("action", [Action.DRAFT_CV, Action.DRAFT_COVER_LETTER])
def test_drafting_is_refused_and_writes_nothing(
    session: Session, company: Company, action: Action
) -> None:
    """The generator is Phase 6. A page claiming work was queued would be a lie."""
    job = _job(session, company)
    outcome = perform(session, action, job.id, now=NOW)
    session.commit()

    assert not outcome.changed
    assert _applications(session) == []


@pytest.mark.parametrize(
    ("action", "status", "expected"),
    [
        (Action.APPLIED, None, True),
        (Action.APPLIED, ApplicationStatus.INTERESTED, True),
        (Action.APPLIED, ApplicationStatus.DISMISSED, True),
        (Action.APPLIED, ApplicationStatus.APPLIED, False),
        (Action.APPLIED, ApplicationStatus.REJECTED, False),
        (Action.DISMISS, None, True),
        (Action.DISMISS, ApplicationStatus.INTERESTED, True),
        (Action.DISMISS, ApplicationStatus.DISMISSED, False),
        (Action.DISMISS, ApplicationStatus.APPLIED, False),
        (Action.DRAFT_CV, None, False),
        (Action.DRAFT_COVER_LETTER, None, False),
    ],
)
def test_the_page_offers_a_button_exactly_when_the_action_would_do_something(
    session: Session, company: Company, action: Action, status: str | None, expected: bool
) -> None:
    """A button that does nothing is worse than a sentence saying nothing is left to do."""
    job = _job(session, company)
    if status is not None:
        session.add(Application(job_id=job.id, status=status))
        session.commit()

    card = job_card(session, job.id, prompt_version=1)
    confirmable, explanation = can_confirm(action, card)
    assert confirmable is expected
    assert explanation, "either answer has to come with its reason"


def test_the_offered_button_and_the_action_agree(session: Session, company: Company) -> None:
    """A page that promised one thing and a POST that did another is worse than either."""
    job = _job(session, company)
    for action in (Action.APPLIED, Action.DISMISS, Action.DRAFT_CV):
        card = job_card(session, job.id, prompt_version=1)
        confirmable, _ = can_confirm(action, card)
        outcome = perform(session, action, job.id, now=NOW)
        session.commit()
        assert outcome.changed is confirmable, f"{action} said one thing and did another"


def test_a_card_for_a_job_that_is_gone_is_told_apart_from_a_bad_link(
    session: Session, company: Company
) -> None:
    """A valid link naming a deleted job is its own answer and not a forgery."""
    with pytest.raises(UnknownJob):
        job_card(session, 999999, prompt_version=1)


def test_the_card_carries_the_score_and_the_summary(
    session: Session, company: Company
) -> None:
    """The confirmation page shows what the email showed, so the two do not disagree."""
    job = _job(session, company, score=88)
    card = job_card(session, job.id, prompt_version=1)
    assert card.score == 88
    assert "post-training" in card.summary
    assert card.company == "Acme"
    assert card.status is None


def _advance(session: Session, job: Job, *events: str, base: datetime = NOW) -> None:
    """Adds timeline events to a job's application one day apart."""
    application = session.scalar(select(Application).where(Application.job_id == job.id))
    for offset, event in enumerate(events, start=1):
        session.add(
            ApplicationEvent(
                application_id=application.id, event=event,
                occurred_at=base + timedelta(days=offset),
            )
        )
    session.commit()


def test_an_untouched_database_gives_an_empty_dashboard(
    session: Session, company: Company
) -> None:
    """Before the first click there is nothing to count and it must say so rather than divide by zero."""
    state = build_dashboard(session, _config(), now=NOW)
    assert state.is_empty
    assert state.applied_total == 0
    assert state.response_rate == 0.0


def test_the_pipeline_counts_where_each_application_stands(
    session: Session, company: Company
) -> None:
    """`applications.status` is what is true now, which is the first question the page answers."""
    first = _job(session, company, "1", "Research Scientist, One")
    second = _job(session, company, "2", "Research Scientist, Two")
    perform(session, Action.APPLIED, first.id, now=NOW)
    perform(session, Action.APPLIED, second.id, now=NOW)
    session.commit()

    state = build_dashboard(session, _config(), now=NOW)
    assert state.pipeline == {ApplicationStatus.APPLIED: 2}
    assert state.applied_total == 2


def test_a_dismissed_job_is_counted_apart_from_the_pipeline(
    session: Session, company: Company
) -> None:
    """A dismissal is not an application and would flatter every rate it was counted in."""
    applied = _job(session, company, "1", "Research Scientist, One")
    dropped = _job(session, company, "2", "Research Scientist, Two")
    perform(session, Action.APPLIED, applied.id, now=NOW)
    perform(session, Action.DISMISS, dropped.id, now=NOW)
    session.commit()

    state = build_dashboard(session, _config(), now=NOW)
    assert state.dismissed == 1
    assert state.applied_total == 1
    assert [a.job_id for a in state.applications] == [applied.id]


def test_each_application_carries_its_own_timeline(
    session: Session, company: Company
) -> None:
    """A single status column throws away progression, which is the point of the table."""
    job = _job(session, company)
    perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    _advance(session, job, EventKind.RECRUITER_SCREEN, EventKind.TECHNICAL)

    application = build_dashboard(session, _config(), now=NOW).applications[0]
    assert [event.event for event in application.events] == [
        EventKind.APPLIED, EventKind.RECRUITER_SCREEN, EventKind.TECHNICAL,
    ]
    assert application.last_event.label == "technical"


def test_the_funnel_counts_stages_reached_and_not_where_things_stand(
    session: Session, company: Company
) -> None:
    """One rejected after an onsite still reached the onsite, which a status column forgets."""
    job = _job(session, company)
    perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    _advance(session, job, EventKind.RECRUITER_SCREEN, EventKind.ONSITE, EventKind.REJECTED)

    reached = {stage.name: stage.count for stage in build_dashboard(session, _config(), now=NOW).funnel}
    assert reached[EventKind.APPLIED] == 1
    assert reached[EventKind.ONSITE] == 1
    assert reached[EventKind.OFFER] == 0


def test_the_funnel_share_is_measured_against_what_was_applied_to(
    session: Session, company: Company
) -> None:
    """A stage count on its own says nothing without the number it came out of."""
    first = _job(session, company, "1", "Research Scientist, One")
    second = _job(session, company, "2", "Research Scientist, Two")
    for job in (first, second):
        perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    _advance(session, first, EventKind.RECRUITER_SCREEN)

    stages = {stage.name: stage for stage in build_dashboard(session, _config(), now=NOW).funnel}
    assert stages[EventKind.APPLIED].share_of_applied == 1.0
    assert stages[EventKind.RECRUITER_SCREEN].share_of_applied == 0.5


def test_a_rejection_counts_as_an_answer_and_being_ghosted_does_not(
    session: Session, company: Company
) -> None:
    """The rate is about whether anybody replied and not about whether they said yes."""
    rejected = _job(session, company, "1", "Research Scientist, One")
    ghosted = _job(session, company, "2", "Research Scientist, Two")
    for job in (rejected, ghosted):
        perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    _advance(session, rejected, EventKind.REJECTED)
    _advance(session, ghosted, EventKind.GHOSTED)

    state = build_dashboard(session, _config(), now=NOW)
    assert state.response_rate == 0.5


def test_the_response_rate_is_broken_down_by_board(
    session: Session, company: Company
) -> None:
    """Which board a role came from is worth knowing before spending more time on it."""
    lever = _job(session, company, "1", "Research Scientist, One", source="lever")
    ashby = _job(session, company, "2", "Research Scientist, Two", source="ashby")
    for job in (lever, ashby):
        perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    _advance(session, lever, EventKind.RECRUITER_SCREEN)

    rates = {row.label: row for row in build_dashboard(session, _config(), now=NOW).by_source}
    assert rates["lever"].rate == 1.0
    assert rates["ashby"].rate == 0.0


def test_tailored_and_untailored_applications_are_counted_apart(
    session: Session, company: Company
) -> None:
    """The comparison only means anything if both sides were counted the same way from the start."""
    tailored = _job(session, company, "1", "Research Scientist, One")
    plain = _job(session, company, "2", "Research Scientist, Two")
    for job in (tailored, plain):
        perform(session, Action.APPLIED, job.id, now=NOW)
    session.commit()
    row = session.scalar(select(Application).where(Application.job_id == tailored.id))
    row.cover_letter_path = "data/drafts/cover-letter.md"
    session.commit()
    _advance(session, tailored, EventKind.RECRUITER_SCREEN)

    rates = {r.label: r for r in build_dashboard(session, _config(), now=NOW).by_tailoring}
    assert rates["tailored"].applied == 1 and rates["tailored"].responded == 1
    assert rates["not tailored"].applied == 1 and rates["not tailored"].responded == 0


def test_a_job_the_digest_has_stopped_showing_is_still_here(
    session: Session, company: Company
) -> None:
    """This is what the "Still Open" line in every digest promises."""
    job = _job(session, company)
    for day in (5, 6, 7, 8):
        session.add(
            DigestAppearance(
                job_id=job.id, digest_date=date(2026, 9, day),
                section=DigestSection.PRIORITY, position=1,
                score_at_appearance=90, content_hash_at_appearance=job.content_hash,
            )
        )
    session.commit()

    state = build_dashboard(session, _config(), now=NOW)
    assert [role.job_id for role in state.open_roles] == [job.id]
    assert state.open_roles[0].appearances == 4
    assert state.open_roles[0].suppressed, "the email has stopped asking, so the page says so"
    assert state.open_suppressed == 1


def test_a_role_the_digest_is_still_showing_is_not_marked_suppressed(
    session: Session, company: Company
) -> None:
    """Most of what is listed is simply not dealt with yet and must not read as hidden."""
    job = _job(session, company)
    session.add(
        DigestAppearance(
            job_id=job.id, digest_date=date(2026, 9, 8),
            section=DigestSection.PRIORITY, position=1,
            score_at_appearance=90, content_hash_at_appearance=job.content_hash,
        )
    )
    session.commit()

    role = build_dashboard(session, _config(), now=NOW).open_roles[0]
    assert role.appearances == 1 and not role.suppressed


def test_the_count_beside_a_role_is_the_one_suppression_works_from(
    session: Session, company: Company
) -> None:
    """A rescore resets the digest's counter so a raw total would explain nothing."""
    job = _job(session, company, score=90)
    for day, shown_at in ((5, 60), (6, 90), (7, 90)):
        session.add(
            DigestAppearance(
                job_id=job.id, digest_date=date(2026, 9, day),
                section=DigestSection.PRIORITY, position=1,
                score_at_appearance=shown_at, content_hash_at_appearance=job.content_hash,
            )
        )
    session.commit()

    role = build_dashboard(session, _config(), now=NOW).open_roles[0]
    assert role.appearances == 2, "the 60 is 30 points away, which resets the counter"


def test_an_actioned_job_leaves_the_open_list(session: Session, company: Company) -> None:
    """Anything still listed here would be something you have already dealt with."""
    job = _job(session, company)
    perform(session, Action.DISMISS, job.id, now=NOW)
    session.commit()
    assert build_dashboard(session, _config(), now=NOW).open_roles == ()


def test_a_job_below_the_threshold_is_not_listed_as_open(
    session: Session, company: Company
) -> None:
    """The dashboard reads the same threshold the digest does or the two would disagree."""
    _job(session, company, score=69)
    assert build_dashboard(session, _config(), now=NOW).open_roles == ()


def test_the_open_list_is_capped_but_says_how_many_there_are(
    session: Session, company: Company
) -> None:
    """A page that silently truncated would make the "Still Open" count a lie."""
    # Titles far enough apart that the fuzzy pass in `dedup.py` keeps them separate.
    for index, area in enumerate(("RLHF", "Evaluation", "Pretraining", "Inference")):
        _job(session, company, str(index), f"Research Scientist, {area}")

    state = build_dashboard(session, _config(), now=NOW, open_limit=2)
    assert len(state.open_roles) == 2
    assert state.open_total == 4


def test_open_roles_are_ordered_by_score(session: Session, company: Company) -> None:
    """The best thing you have not dealt with should be the first thing you read."""
    _job(session, company, "1", "Research Scientist, Low", score=75)
    _job(session, company, "2", "Research Scientist, High", score=95)

    titles = [role.title for role in build_dashboard(session, _config(), now=NOW).open_roles]
    assert titles == ["Research Scientist, High", "Research Scientist, Low"]
