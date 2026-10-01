"""Tests for the web application.

The action tests are written through `TestClient` rather than by calling the
route functions. `ACTION_TOKEN_SECRET` is set per test rather than read from
`.env`, so the suite behaves the same on a machine that has one and a machine
that does not.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import FakeAdapter, make_posting
from job_hunters.actions import Action, CandidateAction, action_url, sign, sign_candidate
from job_hunters.config import ConfigError, load_all, load_search_profile
from job_hunters.ingest import ingest_company
from job_hunters.normalize import normalize_company
from job_hunters.tables import (
    Application,
    CandidateCompany,
    CandidateStatus,
    ApplicationStatus,
    Company,
    Job,
    JobSource,
    LocationFit,
    Score,
)
from job_hunters.web import app

SECRET = "an-action-token-secret"
NOW = datetime(2026, 9, 10, 8, 0, tzinfo=UTC)
TTL = 90


@pytest.fixture
def signed(monkeypatch) -> str:
    """Puts a known signing secret in the environment for the duration of one test."""
    monkeypatch.setenv("ACTION_TOKEN_SECRET", SECRET)
    return SECRET


def _job(session: Session, company: Company, title: str = "Research Scientist",
         *, score: int = 90) -> Job:
    """One ingested posting judged as the job it deduplicated to."""
    prompt_version = load_search_profile().scoring.prompt_version
    ingest_company(
        session, company,
        FakeAdapter.returning(
            "greenhouse", make_posting("1", title, location="Lisbon, Portugal")
        ),
        NOW,
    )
    session.commit()
    posting = session.scalar(select(JobSource))
    session.add(
        Score(
            source_id=posting.id, score=score, summary="A post-training role. It fits.",
            rationale="Because.", matched_areas=["RLHF"], concerns=[],
            work_authorization="eligible", location_fit=LocationFit.PRIORITY,
            prompt_version=prompt_version, model="claude-haiku-4-5",
            content_hash=posting.content_hash,
        )
    )
    session.commit()
    return session.get(Job, posting.job_id)


def _token(action: Action, job_id: int, **kwargs) -> str:
    """One signed token for these tests with the lifetime the config ships."""
    return sign(SECRET, action, job_id, ttl_days=kwargs.pop("ttl_days", TTL), **kwargs)


def _status(session: Session, job_id: int) -> str | None:
    """This job's application status or None when it has no application row."""
    session.expire_all()
    row = session.scalar(select(Application).where(Application.job_id == job_id))
    return row.status if row is not None else None


def test_health_returns_ok() -> None:
    """The `/health` endpoint responds 200 with a small ok body."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"Status": "Ok"}


def test_startup_validates_the_real_config() -> None:
    """Starting the app loads this repository's config without error."""
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_startup_refuses_a_broken_config(monkeypatch) -> None:
    """A broken config stops the server from starting with a silent misconfiguration."""

    def _raise() -> None:
        raise ConfigError("`system_config.yaml` is invalid:\n  timezone: nope")

    monkeypatch.setattr("job_hunters.web.load_all", _raise)

    with pytest.raises(ConfigError, match="timezone"), TestClient(app):
        pass  # pragma: no cover - startup raises before the body runs


def test_startup_creates_the_database_schema(tmp_path, monkeypatch) -> None:
    """Starting the app creates the schema since the container's volume starts empty."""
    from sqlalchemy import inspect

    from job_hunters import db as db_module

    db_module.reset_engine()
    monkeypatch.setattr("job_hunters.paths.DB_PATH", tmp_path / "fresh.db")
    monkeypatch.setattr("job_hunters.paths.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("job_hunters.paths.DRAFTS_DIR", tmp_path / "data" / "drafts")
    monkeypatch.setattr("job_hunters.paths.BACKUP_DIR", tmp_path / "backups")

    with TestClient(app):
        tables = set(inspect(db_module.get_engine()).get_table_names())
    db_module.reset_engine()
    assert {"jobs", "job_sources", "scores"} <= tables


def test_unknown_routes_404() -> None:
    """Only the routes we declared exist."""
    with TestClient(app) as client:
        assert client.get("/not-a-route").status_code == 404


def test_a_signed_link_opens_a_page_naming_the_job(
    session: Session, company: Company, signed: str
) -> None:
    """A link that opened a blank page would leave you confirming you know not what."""
    job = _job(session, company)
    with TestClient(app) as client:
        response = client.get(f"/a/{_token(Action.APPLIED, job.id)}")

    assert response.status_code == 200
    assert "Research Scientist" in response.text
    assert "Acme" in response.text
    assert "90" in response.text


def test_opening_the_link_changes_nothing(
    session: Session, company: Company, signed: str
) -> None:
    """A mail client prefetching links must not be able to mark a role applied."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.get(f"/a/{_token(Action.APPLIED, job.id)}")

    assert _status(session, job.id) is None, "a GET wrote a row"


def test_the_page_carries_the_form_that_does_the_work(
    session: Session, company: Company, signed: str
) -> None:
    """The button posts back to the same address, which is the whole GET/POST split."""
    job = _job(session, company)
    token = _token(Action.APPLIED, job.id)
    with TestClient(app) as client:
        text = client.get(f"/a/{token}").text

    assert f'action="/a/{token}"' in text
    assert 'method="post"' in text


def test_a_draft_link_says_it_is_not_built_and_offers_no_button(
    session: Session, company: Company, signed: str
) -> None:
    """Nothing is queued so the page has to say so and not offer to queue it."""
    job = _job(session, company)
    with TestClient(app) as client:
        text = client.get(f"/a/{_token(Action.DRAFT_COVER_LETTER, job.id)}").text

    assert "Phase 6" in text
    assert 'method="post"' not in text


def test_a_link_for_a_job_already_applied_to_offers_no_button(
    session: Session, company: Company, signed: str
) -> None:
    """A button that does nothing is worse than a sentence saying nothing is left to do."""
    job = _job(session, company)
    session.add(Application(job_id=job.id, status=ApplicationStatus.APPLIED))
    session.commit()

    with TestClient(app) as client:
        text = client.get(f"/a/{_token(Action.APPLIED, job.id)}").text

    assert "Already recorded" in text
    assert 'method="post"' not in text


def test_confirming_applied_records_it(
    session: Session, company: Company, signed: str
) -> None:
    """Clicking the button posts back to the same address and creates the application row."""
    job = _job(session, company)
    with TestClient(app) as client:
        response = client.post(f"/a/{_token(Action.APPLIED, job.id)}")

    assert response.status_code == 200
    assert "Marked as applied" in response.text
    assert _status(session, job.id) == ApplicationStatus.APPLIED


def test_confirming_the_same_link_twice_is_a_no_op(
    session: Session, company: Company, signed: str
) -> None:
    """Clicking the button twice posts back to the same address and does not create a second row."""
    job = _job(session, company)
    token = _token(Action.APPLIED, job.id)
    with TestClient(app) as client:
        first = client.post(f"/a/{token}")
        second = client.post(f"/a/{token}")

    assert first.status_code == second.status_code == 200
    assert "Already recorded" in second.text
    assert len(list(session.scalars(select(Application)))) == 1


def test_two_posts_landing_at_once_are_still_one_application(
    session: Session, company: Company, signed: str, monkeypatch
) -> None:
    """A double-click sends the form twice and `applications.job_id` is unique."""
    from job_hunters import web as web_module

    job = _job(session, company)
    session.add(Application(job_id=job.id, status=ApplicationStatus.INTERESTED))
    session.commit()

    real_perform = web_module.perform
    attempts: list[int] = []

    def racing(session_, action, job_id, **kwargs):
        """Loses the race once by inserting a row the constraint already forbids."""
        attempts.append(job_id)
        if len(attempts) == 1:
            session_.add(Application(job_id=job_id, status=ApplicationStatus.APPLIED))
            session_.flush()  # pragma: no cover - raises before returning
        return real_perform(session_, action, job_id, **kwargs)

    monkeypatch.setattr(web_module, "perform", racing)

    with TestClient(app) as client:
        response = client.post(f"/a/{_token(Action.APPLIED, job.id)}")

    assert response.status_code == 200
    assert len(attempts) == 2, "the losing attempt was retried"
    assert len(list(session.scalars(select(Application)))) == 1


def test_an_applied_job_leaves_the_dashboard_open_list(
    session: Session, company: Company, signed: str
) -> None:
    """What the digest stops showing is what the dashboard stops showing too."""
    job = _job(session, company)
    with TestClient(app) as client:
        assert "Research Scientist" in client.get("/").text
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        after = client.get("/").text

    assert "Nothing outstanding" in after


def test_confirming_dismiss_records_it(
    session: Session, company: Company, signed: str
) -> None:
    """Dismissal is the other action a link can carry out today."""
    job = _job(session, company)
    with TestClient(app) as client:
        response = client.post(f"/a/{_token(Action.DISMISS, job.id)}")

    assert "Dismissed" in response.text
    assert _status(session, job.id) == ApplicationStatus.DISMISSED


def test_posting_a_draft_link_queues_nothing(
    session: Session, company: Company, signed: str
) -> None:
    """The page offers no button but the address can still be posted to by hand."""
    job = _job(session, company)
    with TestClient(app) as client:
        response = client.post(f"/a/{_token(Action.DRAFT_CV, job.id)}")

    assert response.status_code == 200
    assert "Not built yet" in response.text
    assert _status(session, job.id) is None


def test_an_expired_link_redirects_to_the_dashboard(
    session: Session, company: Company, signed: str
) -> None:
    """A link that was genuinely ours and simply sat too long is not an error."""
    job = _job(session, company)
    stale = sign(SECRET, Action.APPLIED, job.id, ttl_days=1,
                 now=datetime.now(UTC) - timedelta(days=2))

    with TestClient(app) as client:
        response = client.get(f"/a/{stale}", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/?expired=1"

        landed = client.get(response.headers["location"])
    assert "expired" in landed.text.lower()


def test_an_expired_link_changes_nothing_even_when_posted(
    session: Session, company: Company, signed: str
) -> None:
    """The redirect must not be a front door that a POST walks around."""
    job = _job(session, company)
    stale = sign(SECRET, Action.APPLIED, job.id, ttl_days=1,
                 now=datetime.now(UTC) - timedelta(days=2))

    with TestClient(app) as client:
        client.post(f"/a/{stale}", follow_redirects=False)
    assert _status(session, job.id) is None


@pytest.mark.parametrize(
    "token", ["not-a-token", "", "AAAA.AAAA", "YXBwbGllZDo0Mg.\N{SNOWMAN}"]
)
def test_a_forged_or_damaged_link_is_refused_and_not_redirected(
    session: Session, company: Company, signed: str, token: str
) -> None:
    """A forgery must never look like it worked, which a redirect to the dashboard would."""
    _job(session, company)
    with TestClient(app) as client:
        response = client.post(f"/a/{token}", follow_redirects=False)

    # An empty token is not this route at all: `/a/` has no path parameter.
    assert response.status_code in (400, 404, 405)
    if response.status_code == 400:
        assert "cannot be trusted" in response.text


def test_a_link_signed_with_another_secret_is_refused(
    session: Session, company: Company, signed: str
) -> None:
    """The secret is what makes a link this installation's rather than anyone's."""
    job = _job(session, company)
    forged = sign("someone-elses-secret", Action.APPLIED, job.id, ttl_days=TTL)

    with TestClient(app) as client:
        response = client.post(f"/a/{forged}")

    assert response.status_code == 400
    assert _status(session, job.id) is None


def test_a_link_naming_a_job_that_is_gone_says_so(
    session: Session, company: Company, signed: str
) -> None:
    """A valid link to a deleted job is its own answer and not a forgery nor a crash."""
    _job(session, company)
    with TestClient(app) as client:
        response = client.post(f"/a/{_token(Action.APPLIED, 999999)}")

    assert response.status_code == 404
    assert "no longer here" in response.text
    assert list(session.scalars(select(Application))) == []


def test_without_a_signing_secret_no_link_is_honoured(
    session: Session, company: Company, monkeypatch
) -> None:
    """Nothing can be verified, so nothing may be carried out and the page names the variable."""
    job = _job(session, company)
    monkeypatch.setenv("ACTION_TOKEN_SECRET", "")
    token = sign(SECRET, Action.APPLIED, job.id, ttl_days=TTL)

    with TestClient(app) as client:
        response = client.post(f"/a/{token}")

    assert response.status_code == 500
    assert "ACTION_TOKEN_SECRET" in response.text
    assert _status(session, job.id) is None


def test_the_dashboard_lists_what_is_still_open(
    session: Session, company: Company, signed: str
) -> None:
    """The "Still Open" line in every digest points here and promises this."""
    _job(session, company)
    with TestClient(app) as client:
        text = client.get("/").text

    assert "Research Scientist" in text
    assert "Open Roles" in text


def test_the_dashboard_signs_the_links_beside_each_open_role(
    session: Session, company: Company, signed: str
) -> None:
    """The same confirm-then-execute flow reachable without going back to the email."""
    job = _job(session, company)
    expected = action_url("http://localhost:8000", SECRET, Action.APPLIED, job.id,
                          ttl_days=TTL)
    marker = "/a/"

    with TestClient(app) as client:
        text = client.get("/").text
        # Whatever token the page printed has to be one this installation accepts.
        token = text[text.index(marker) + len(marker):].split('"')[0]
        assert client.get(f"/a/{token}").status_code == 200

    assert "Applied" in text and "Dismiss" in text
    assert expected.rsplit("/", 1)[0] in text, "links are built from the declared base_url"


def test_the_dashboard_still_renders_without_a_signing_secret(
    session: Session, company: Company, monkeypatch
) -> None:
    """A missing secret must not take the whole page down. Only the links on it."""
    _job(session, company)
    monkeypatch.setenv("ACTION_TOKEN_SECRET", "")

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "ACTION_TOKEN_SECRET is not set" in response.text
    assert "/a/" not in response.text


def test_the_dashboard_shows_the_pipeline_and_the_timeline(
    session: Session, company: Company, signed: str
) -> None:
    """Where each application stands and how it got there."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        text = client.get("/").text

    assert "Pipeline" in text
    assert "applied" in text
    assert "Funnel" in text
    assert "Response Rate" in text


def test_a_dashboard_with_nothing_on_it_says_so(session: Session) -> None:
    """An empty database has to read as empty rather than as broken."""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Nothing outstanding" in response.text
    assert "No application yet" in response.text


def test_a_query_string_nobody_expected_does_not_break_the_dashboard(
    session: Session, signed: str
) -> None:
    """A dashboard is where you land from a redirect so it must not be fussy about the query."""
    with TestClient(app) as client:
        assert client.get("/?expired=whatever&other=1").status_code == 200


def test_a_job_title_cannot_become_markup(
    session: Session, company: Company, signed: str
) -> None:
    """A board's text reaches a browser here so it is escaped rather than trusted."""
    job = _job(session, company, title="Research Scientist <script>alert(1)</script>")
    with TestClient(app) as client:
        dashboard = client.get("/").text
        confirmation = client.get(f"/a/{_token(Action.APPLIED, job.id)}").text

    for page in (dashboard, confirmation):
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page


@pytest.fixture
def watchlist(tmp_path, monkeypatch) -> Path:
    """A watchlist of this test's own, so approving from a page never edits `config/`."""
    target = tmp_path / "companies_watchlist.yaml"
    target.write_text(
        "- { slug: anthropic, name: Anthropic, ats: greenhouse, token: anthropic, tier: lab }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("job_hunters.paths.WATCHLIST_PATH", target)
    return target


@pytest.fixture(autouse=True)
def no_tier_call(monkeypatch) -> None:
    """No test reaches the API. A suggestion is a bonus and every page works without one."""
    monkeypatch.setattr("job_hunters.web.suggest_tier", lambda *_args, **_kwargs: None)


def _candidate(session: Session, name: str, *, ats: str | None = "ashby",
               token: str | None = None, status: str = CandidateStatus.PENDING) -> CandidateCompany:
    """One queued company with a board unless `ats` is None."""
    row = CandidateCompany(
        name=name, name_key=normalize_company(name), status=status, sightings=2,
        ats_type=ats, ats_token=token or (name.lower().replace(" ", "-") if ats else None),
        board_url="https://ashby.test/x" if ats else None, board_jobs=24 if ats else None,
        careers_url="https://example.test/careers", roles=["Research Scientist"],
        evidence=[{"source": "hn", "source_job_id": "1", "title": f"{name} | Berlin",
                   "url": "https://news.ycombinator.com/item?id=1", "seen": "2026-09-14"}],
    )
    session.add(row)
    session.commit()
    return row


def _decided(session: Session, candidate_id: int) -> CandidateCompany:
    """The candidate as the database now has it. The app writes in its own session."""
    session.expire_all()
    return session.get(CandidateCompany, candidate_id)


def _candidate_token(action: CandidateAction, candidate_id: int) -> str:
    """One signed company token for these tests."""
    return sign_candidate(SECRET, action, candidate_id, ttl_days=TTL, now=NOW)


def test_the_dashboard_lists_the_queue_with_boards_apart_from_those_without(
    session: Session, signed: str, watchlist: Path
) -> None:
    """Approving needs a board, so the two are not offered the same decisions."""
    _candidate(session, "Prior Labs")
    _candidate(session, "Tufalabs", ats=None)

    with TestClient(app) as client:
        page = client.get("/").text

    assert "Discovered Companies (1)" in page and "No Board Found (1)" in page
    assert "Sightings" in page and "24 posting" not in page, "the board's size is not the count"
    assert page.count(">Approve</a>") == 1, "only the company with a board can be approved"
    assert page.count(">Reject</a>") == 2


def test_the_queue_says_when_an_approval_takes_effect(
    session: Session, signed: str, watchlist: Path
) -> None:
    """A company added now is not fetched now and a reader who is not told will wonder."""
    _candidate(session, "Prior Labs")
    with TestClient(app) as client:
        page = client.get("/").text
    assert "next ingest" in page and "every 2h" in page


def test_the_queue_hides_its_links_when_nothing_can_be_signed(
    session: Session, monkeypatch, watchlist: Path
) -> None:
    """Rows with dead buttons would read as broken, so the reason is printed instead."""
    monkeypatch.delenv("ACTION_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(
        "job_hunters.config.Secrets.optional",
        lambda self, name: None if name == "action_token_secret" else "x",
    )
    _candidate(session, "Prior Labs")

    with TestClient(app) as client:
        page = client.get("/").text

    assert "Prior Labs" in page and "ACTION_TOKEN_SECRET is not set" in page
    assert "/c/" not in page


def test_approving_from_the_page_appends_the_line_and_shows_it(
    session: Session, signed: str, watchlist: Path
) -> None:
    """The line is the only copy outside the file and an editor can overwrite the file."""
    row = _candidate(session, "Prior Labs")
    token = _candidate_token(CandidateAction.APPROVE, row.id)

    with TestClient(app) as client:
        assert "Prior Labs" in client.get(f"/c/{token}").text
        done = client.post(f"/c/{token}", data={"slug": "prior-labs", "name": "Prior Labs",
                                                "tier": "lab"})

    assert done.status_code == 200
    assert "added to the watchlist" in done.text.lower()
    assert "slug: prior-labs" in done.text and "tier: lab" in done.text
    assert "slug: prior-labs" in watchlist.read_text()
    decided = _decided(session, row.id)
    assert decided.status == CandidateStatus.APPROVED and decided.slug == "prior-labs"


def test_the_confirm_page_offers_the_fields_the_command_takes(
    session: Session, signed: str, watchlist: Path
) -> None:
    """Approving from a button alone would file every company under the extractor's spelling."""
    row = _candidate(session, "Artificial Intelligence Underwriting Company")
    with TestClient(app) as client:
        page = client.get(f"/c/{_candidate_token(CandidateAction.APPROVE, row.id)}").text

    assert 'name="slug"' in page and 'name="name"' in page and 'name="tier"' in page
    assert 'value="artificial-intelligence-underwriting-company"' in page
    for tier in ("lab", "bigtech", "infra", "discovered"):
        assert f'value="{tier}"' in page


def test_a_suggested_tier_is_the_one_already_chosen(
    session: Session, signed: str, watchlist: Path, monkeypatch
) -> None:
    """The suggestion is only useful if the reviewer can take it by pressing the button."""
    from job_hunters.tables import Tier

    row = _candidate(session, "Prior Labs")
    monkeypatch.setattr(
        "job_hunters.web.suggest_tier",
        lambda *_args, **_kwargs: (Tier.LAB, "it trains foundation models"),
    )
    with TestClient(app) as client:
        page = client.get(f"/c/{_candidate_token(CandidateAction.APPROVE, row.id)}").text

    assert '<option value="lab" selected>' in page
    assert "it trains foundation models" in page


def test_a_page_still_works_when_no_tier_can_be_suggested(
    session: Session, signed: str, watchlist: Path
) -> None:
    """No API key, no network or a refused call must not stop a company being approved."""
    row = _candidate(session, "Prior Labs")
    with TestClient(app) as client:
        page = client.get(f"/c/{_candidate_token(CandidateAction.APPROVE, row.id)}").text

    assert '<option value="discovered" selected>' in page
    assert "suggestion" not in page


def test_approving_twice_says_so_instead_of_writing_a_second_line(
    session: Session, signed: str, watchlist: Path
) -> None:
    """A link found again months later decides from the state it finds (like every other action)."""
    row = _candidate(session, "Prior Labs")
    token = _candidate_token(CandidateAction.APPROVE, row.id)

    with TestClient(app) as client:
        client.post(f"/c/{token}", data={"slug": "prior-labs", "name": "Prior Labs", "tier": "lab"})
        again = client.post(f"/c/{token}", data={"slug": "prior-labs", "name": "Prior Labs",
                                                 "tier": "lab"})

    assert "already in the watchlist as prior-labs" in again.text
    assert watchlist.read_text().count("slug: prior-labs") == 1


def test_a_slug_the_file_already_has_comes_back_as_a_fixable_form(
    session: Session, signed: str, watchlist: Path
) -> None:
    """The reviewer can pick another slug, so this is a question and not a dead end."""
    row = _candidate(session, "Anthropic Labs", ats="greenhouse", token="anthropic-labs")
    before = watchlist.read_text()

    with TestClient(app) as client:
        answer = client.post(
            f"/c/{_candidate_token(CandidateAction.APPROVE, row.id)}",
            data={"slug": "anthropic", "name": "Anthropic Labs", "tier": "lab"},
        )

    assert "anthropic" in answer.text and "already in" in answer.text
    assert 'name="slug"' in answer.text and 'value="anthropic"' in answer.text
    assert watchlist.read_text() == before
    assert _decided(session, row.id).status == CandidateStatus.PENDING


def test_a_company_with_no_board_cannot_be_approved_from_a_page_either(
    session: Session, signed: str, watchlist: Path
) -> None:
    """A signed link is not a way around the rule that there has to be something to watch."""
    row = _candidate(session, "Tufalabs", ats=None)
    token = _candidate_token(CandidateAction.APPROVE, row.id)

    with TestClient(app) as client:
        page = client.get(f"/c/{token}").text
        posted = client.post(f"/c/{token}", data={"slug": "tufalabs", "tier": "lab"})

    for body in (page, posted.text):
        assert "no greenhouse, lever or ashby board was found" in body.lower()
    assert "slug: tufalabs" not in watchlist.read_text()
    assert _decided(session, row.id).status == CandidateStatus.PENDING


def test_rejecting_from_the_page_takes_it_out_of_the_queue(
    session: Session, signed: str, watchlist: Path
) -> None:
    """The queue only shrinks if rejecting is as easy as approving."""
    row = _candidate(session, "Tufalabs", ats=None)
    token = _candidate_token(CandidateAction.REJECT, row.id)

    with TestClient(app) as client:
        assert "for good" in client.get(f"/c/{token}").text
        done = client.post(f"/c/{token}")
        page = client.get("/").text

    assert done.status_code == 200 and "rejected" in done.text.lower()
    assert _decided(session, row.id).status == CandidateStatus.REJECTED
    assert "No Board Found" not in page, "it has left the queue"
    assert "Rejected (1)" in page and ">Undo</a>" in page, "and is listed where it can come back"


def test_a_company_token_is_not_a_job_token(session: Session, company: Company, signed: str) -> None:
    """The two id spaces overlap, so the signed bytes say which kind they are."""
    job = _job(session, company)
    with TestClient(app) as client:
        crossed = client.get(f"/a/{_candidate_token(CandidateAction.APPROVE, job.id)}")
        other_way = client.get(f"/c/{_token(Action.APPLIED, job.id)}")

    assert crossed.status_code == 400 and other_way.status_code == 400
    for response in (crossed, other_way):
        assert "cannot be trusted" in response.text


def test_a_company_name_cannot_become_markup(
    session: Session, signed: str, watchlist: Path
) -> None:
    """Names are read out of prose by a model, so they are escaped like any board's text."""
    row = _candidate(session, "Prior <script>alert(1)</script> Labs")
    with TestClient(app) as client:
        page = client.get("/").text
        confirmation = client.get(f"/c/{_candidate_token(CandidateAction.APPROVE, row.id)}").text

    for body in (page, confirmation):
        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;" in body


def test_every_queued_company_with_a_board_is_a_link(
    session: Session, signed: str, watchlist: Path
) -> None:
    """The aggregators name a company without saying where it hires, so the board stands in."""
    named = _candidate(session, "Prior Labs")
    named.careers_url = None
    session.commit()

    with TestClient(app) as client:
        page = client.get("/").text

    assert 'href="https://jobs.ashbyhq.com/prior-labs"' in page


def test_a_rejection_is_undone_from_the_rejected_list(
    session: Session, signed: str, watchlist: Path
) -> None:
    """A misclick is put right from the dashboard."""
    row = _candidate(session, "Tufalabs", ats=None)

    with TestClient(app) as client:
        done = client.post(f"/c/{_candidate_token(CandidateAction.REJECT, row.id)}")
        link = re.search(r'href="([^"]*/c/[^"]+)">Undo</a>', client.get("/").text).group(1)
        button = re.search(r'action="(/c/[^"]+)"', client.get(link).text).group(1)
        back = client.post(button)
        page = client.get("/").text

    assert "rejected" in done.text.lower() and "Undo" not in done.text, "like a dismissal's page"
    assert 'href="/#discovered"' in done.text, "whose one control is the way back to the dashboard"
    assert "back in the queue" in back.text
    assert _decided(session, row.id).status == CandidateStatus.PENDING
    assert _decided(session, row.id).decided_at is None
    assert "No Board Found (1)" in page and "Rejected (" not in page, (
        "and it is waiting for a decision again"
    )


def test_undoing_what_was_never_rejected_says_so(
    session: Session, signed: str, watchlist: Path
) -> None:
    """A link found twice decides from the state it finds (like every other action)."""
    row = _candidate(session, "Tufalabs", ats=None)
    token = _candidate_token(CandidateAction.UNREJECT, row.id)

    with TestClient(app) as client:
        page = client.get(f"/c/{token}").text
        posted = client.post(f"/c/{token}")

    for body in (page, posted.text):
        assert "nothing to undo" in body
    assert _decided(session, row.id).status == CandidateStatus.PENDING


def test_an_approved_company_cannot_be_undone_into_the_queue(
    session: Session, signed: str, watchlist: Path
) -> None:
    """Leaving the watchlist is an edit to that file and not a decision taken here."""
    row = _candidate(session, "Prior Labs")
    with TestClient(app) as client:
        client.post(f"/c/{_candidate_token(CandidateAction.APPROVE, row.id)}",
                    data={"slug": "prior-labs", "name": "Prior Labs", "tier": "lab"})
        refused = client.post(f"/c/{_candidate_token(CandidateAction.UNREJECT, row.id)}")

    assert "is in the watchlist" in refused.text
    assert _decided(session, row.id).status == CandidateStatus.APPROVED


def test_the_queue_does_not_offer_undo_beside_every_company(
    session: Session, signed: str, watchlist: Path
) -> None:
    """Undo belongs to the rejected list and not to a row that has decided nothing."""
    _candidate(session, "Prior Labs")
    with TestClient(app) as client:
        page = client.get("/").text
    assert ">Undo</a>" not in page


def test_the_dashboard_lists_what_was_rejected_and_offers_only_the_way_back(
    session: Session, signed: str, watchlist: Path
) -> None:
    """The page that rejected offers no undo so this list is the way back."""
    row = _candidate(session, "Tufalabs", ats=None)
    with TestClient(app) as client:
        client.post(f"/c/{_candidate_token(CandidateAction.REJECT, row.id)}")
        page = client.get("/").text

    section = page[page.find("Rejected (1)"):]
    assert "Tufalabs" in section
    today = datetime.now(UTC).strftime("%d %b %Y")
    assert f"rejected {today}" in section, "the date is how a misclick is found again"
    assert ">Undo</a>" in section
    assert ">Approve</a>" not in section, "approving a rejected company is refused, so it is not offered"


def test_the_rejected_list_is_capped_and_says_so(
    session: Session, signed: str, watchlist: Path, monkeypatch
) -> None:
    """The rejected companies list only accumulates so it needs an end."""
    monkeypatch.setattr("job_hunters.web.REJECTED_SHOWN", 2)
    for name in ("One Co", "Two Co", "Three Co"):
        row = _candidate(session, name, ats=None)
        with TestClient(app) as client:
            client.post(f"/c/{_candidate_token(CandidateAction.REJECT, row.id)}")

    with TestClient(app) as client:
        page = client.get("/").text

    assert "Rejected (3)" in page
    assert "Showing the most recent 2 of 3" in page


def test_nothing_rejected_means_no_section_at_all(
    session: Session, signed: str, watchlist: Path
) -> None:
    """An empty fold is a row of furniture that says nothing."""
    _candidate(session, "Prior Labs")
    with TestClient(app) as client:
        page = client.get("/").text
    assert "Rejected (" not in page


def test_the_dashboard_offers_a_dismissed_job_the_two_ways_out(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """Changing your mind is either "I applied anyway" or "put it back" and nothing else."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.DISMISS, job.id)}")
        page = client.get("/").text

    section = page[page.find("Dismissed (1)"):]
    assert "Research Scientist" in section
    assert ">Applied</a>" in section and ">Undismiss</a>" in section
    assert ">Dismiss</a>" not in section, "it is already dismissed"


def test_undismissing_from_the_page_puts_the_job_back(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """The link asks first and then the job is open again."""
    job = _job(session, company)
    token = _token(Action.UNDISMISS, job.id)

    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.DISMISS, job.id)}")
        asked = client.get(f"/a/{token}").text
        assert "Undismiss" in asked and "open roles" in asked
        done = client.post(f"/a/{token}")
        page = client.get("/").text

    assert "Back among the open roles" in done.text
    assert "Still Open" in done.text, "and the email's counter is unchanged, which it says"
    assert _status(session, job.id) is None, "the row is gone, not reset"
    assert "Dismissed (" not in page
    assert "Research Scientist" in page[:page.find("Pipeline")], "it is an open role again"


def test_undismissing_never_deletes_an_application(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """A forged or stale link must not be a way to discard an application and its timeline."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        refused = client.post(f"/a/{_token(Action.UNDISMISS, job.id)}")

    assert "refused" in refused.text
    assert _status(session, job.id) == ApplicationStatus.APPLIED


def test_the_digest_links_do_not_include_undismissing(
    session: Session, company: Company, signed: str
) -> None:
    """A dismissed job is not in the email, so the email has nothing to undo."""
    from job_hunters.actions import action_links

    links = action_links("http://localhost:8000", SECRET, 1, ttl_days=TTL, now=NOW)
    assert [link.label for link in links] == [
        "Draft CV", "Draft Cover Letter", "Applied", "Dismiss",
    ]


def test_an_application_card_offers_to_add_an_event(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """The tracker measures a process that nothing else can record a step of."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        page = client.get("/").text

    assert ">Add an event</a>" in page


def test_the_event_form_offers_every_step_and_todays_date(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """A date field carries no time, so the day offered is the day where the reader is."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        page = client.get(f"/a/{_token(Action.ADD_EVENT, job.id)}").text

    for kind in ("recruiter_screen", "technical", "onsite", "offer", "rejected", "note"):
        assert f'value="{kind}"' in page
    today = datetime.now(ZoneInfo(load_all().system.timezone)).date().isoformat()
    assert f'name="occurred_on" value="{today}"' in page
    assert "applied" in page, "the timeline so far is on the page it is added to"


def test_recording_a_step_writes_it_and_says_where_the_application_stands(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """The point of the form is that the funnel and the pipeline have something to read."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        done = client.post(
            f"/a/{_token(Action.ADD_EVENT, job.id)}",
            data={"event": "recruiter_screen", "occurred_on": "2026-09-20",
                  "notes": "Twenty minutes, mostly logistics."},
        )
        page = client.get("/").text

    assert "Recorded: recruiter screen" in done.text and "in_process" in done.text
    assert _status(session, job.id) == ApplicationStatus.IN_PROCESS
    assert "Twenty minutes, mostly logistics." in page
    assert "20 Sep 2026" in page, "the date a reader picked is the date they are shown"


def _inside_fold(page: str, summary: str) -> str:
    """What one collapsed list holds (up to its own closing tag and past any nested in it)."""
    opening = re.search(rf"<details>\s*<summary>{re.escape(summary)}</summary>", page)
    assert opening, f"no collapsed list called {summary!r}"
    depth = 1
    for tag in re.finditer(r"<(/?)details\b", page[opening.end():]):
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            return page[opening.end():opening.end() + tag.start()]
    raise AssertionError(f"{summary!r} is never closed")


def test_an_application_that_reached_an_outcome_is_folded_away(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """An ended application is out of the way but still one click from its timeline."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        client.post(
            f"/a/{_token(Action.ADD_EVENT, job.id)}",
            data={"event": "rejected", "occurred_on": "2026-09-20"},
        )
        page = client.get("/").text

    folded = _inside_fold(page, "Ended (1)")
    cards = page[page.find("<h2>Pipeline</h2>"):page.find("Ended (1)")]
    assert "Research Scientist" not in cards, "it has left the cards"
    assert "rejected 1" in cards, "but the count above them still has it"
    assert "Research Scientist" in folded and "20 Sep 2026" in folded
    assert ">Add an event</a>" in folded, "a step that comes later can still be recorded"
    assert "Of applied (1)" in page, "and the funnel still counts it"


def test_an_application_still_waiting_is_not_folded(
    session: Session, company: Company, signed: str
) -> None:
    """Only an outcome moves a card so with none there is nothing to fold."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        page = client.get("/").text

    assert "Ended (" not in page
    assert "Research Scientist" in page[page.find("<h2>Pipeline</h2>"):page.find("<h2>Funnel</h2>")]


def test_a_card_reads_company_board_and_date_and_says_active_once_answered(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """No status bubble, no tailoring mark and no word until somebody answers."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        row = session.scalar(select(Application).where(Application.job_id == job.id))
        row.cv_path = "cv-tailored.pdf"
        session.commit()
        waiting = client.get("/").text
        client.post(
            f"/a/{_token(Action.ADD_EVENT, job.id)}",
            data={"event": "recruiter_screen", "occurred_on": "2026-09-20"},
        )
        answered = client.get("/").text

    assert "Acme &middot; via greenhouse" in waiting, "nothing between the company and the board"
    assert "tailored" not in waiting, "a tailored application is not marked as one"
    assert "class=\"good\">active" not in waiting and "class=\"bad\"" not in waiting
    assert "<span class=\"good\">active</span>" in answered


@pytest.mark.parametrize(
    ("ending", "colour"),
    [("offer", "good"), ("rejected", "bad"), ("withdrawn", "bad"), ("ghosted", "bad")],
)
def test_an_ended_card_says_how_it_ended_in_green_or_red(
    session: Session, company: Company, signed: str, watchlist: Path, ending: str, colour: str
) -> None:
    """An offer reads green and the three ways an application is lost read red."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        client.post(
            f"/a/{_token(Action.ADD_EVENT, job.id)}",
            data={"event": ending, "occurred_on": "2026-09-25"},
        )
        page = client.get("/").text

    assert f"<span class=\"{colour}\">{ending}</span>" in _inside_fold(page, "Ended (1)")


def test_a_refused_step_comes_back_as_the_same_form(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """Another date or another step would work, so this is a question and not a dead end."""
    job = _job(session, company)
    token = _token(Action.ADD_EVENT, job.id)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        client.post(f"/a/{token}", data={"event": "offer", "occurred_on": "2026-09-20"})
        again = client.post(f"/a/{token}", data={"event": "offer", "occurred_on": "2026-09-28"})

    assert "only happens once" in again.text
    assert 'name="event"' in again.text and 'value="2026-09-28"' in again.text


def test_a_date_that_is_not_a_date_changes_nothing(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """A browser sends a date field, but a hand-made post can send anything."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.APPLIED, job.id)}")
        answer = client.post(
            f"/a/{_token(Action.ADD_EVENT, job.id)}",
            data={"event": "technical", "occurred_on": "last tuesday"},
        )

    assert "not a date" in answer.text
    assert _status(session, job.id) == ApplicationStatus.APPLIED


def test_the_form_says_so_when_there_is_no_application_to_add_to(
    session: Session, company: Company, signed: str, watchlist: Path
) -> None:
    """A link kept from a job later dismissed opens a page that explains rather than a form."""
    job = _job(session, company)
    with TestClient(app) as client:
        client.post(f"/a/{_token(Action.DISMISS, job.id)}")
        page = client.get(f"/a/{_token(Action.ADD_EVENT, job.id)}").text

    assert "no application here to add to" in page
    assert 'name="event"' not in page
