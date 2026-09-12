"""Tests for the web application.

The action tests are written through `TestClient` rather than by calling the
route functions. `ACTION_TOKEN_SECRET` is set per test rather than read from
`.env`, so the suite behaves the same on a machine that has one and a machine
that does not.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import FakeAdapter, make_posting
from job_hunters.actions import Action, action_url, sign
from job_hunters.config import ConfigError
from job_hunters.ingest import ingest_company
from job_hunters.models import (
    Application,
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
            prompt_version=1, model="claude-haiku-4-5", content_hash=posting.content_hash,
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
    assert "Open roles" in text


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
    assert "Response rate" in text


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
