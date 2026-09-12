"""The web application with the signed action links and the dashboard behind them.

Four routes:

    GET  /health      liveness (touching nothing)
    GET  /            the dashboard
    GET  /a/{token}   what one signed link asks and a button
    POST /a/{token}   what that button does

There is no CSRF token. A form post from another site could reach these routes
because the browser is on the same machine, but it would have to name a valid
signed token to do anything - and the only place those exist is in the inbox the
digest was sent to. The token is doing the work a CSRF token would. This holds
only while the service is bound to loopback and single-user. Moving it to a
reverse proxy would need to revisit this along with the token lifetime (section 3.9).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy.exc import IntegrityError

from . import paths
from .actions import (
    ACTION_LABELS,
    Action,
    ActionLink,
    ExpiredToken,
    SignedAction,
    TokenError,
    action_links,
    verify,
)
from .config import AppConfig, ConfigError, load_all
from .db import SchemaError, init_db, session_scope
from .templating import render
from .tracker import (
    JobCard,
    Outcome,
    UnknownJob,
    build_dashboard,
    can_confirm,
    job_card,
    perform,
)

log = logging.getLogger("job_hunters.web")

# The two  actions the tracker can carry out today. Drafting is Phase 6 and its links
# reach an honest page from the email, but there is no reason to print one here.
DASHBOARD_ACTIONS: tuple[Action, ...] = (Action.APPLIED, Action.DISMISS)

# Redirect after a POST and the status that means "look over there instead".
SEE_OTHER = 303


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Validates configuration and creates the schema before serving any request."""
    try:
        load_all()
    except ConfigError as exc:
        log.error("Cannot start: %s", exc)
        raise
    # Whichever of `web` and `scheduler` starts first creates the schema in the
    # otherwise empty Docker volume. Both calls are idempotent and `create_all`
    # issues "create table if not exists", so the two racing is harmless.
    paths.ensure_runtime_dirs()
    try:
        init_db()
    except SchemaError as exc:
        # Refuse to start. A service reporting itself healthy over a database
        # it cannot read is the failure this check exists to prevent.
        log.error("Cannot start: %s", exc)
        raise
    yield


app = FastAPI(title="Job-Hunters", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Returns 200 with a small body: proof the process is alive and serving.

    Deliberately a liveness check and not a readiness check. It does not touch the
    database or any external service, only confirms the web process itself can
    accept a request and respond.
    """
    return {"Status": "Ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard(expired: str | None = None) -> HTMLResponse:
    """What is outstanding, where each application stands and how they go."""
    config = load_all()
    secret = config.secrets.optional("action_token_secret")
    with session_scope() as session:
        state = build_dashboard(session, config)
    links: dict[int, tuple[ActionLink, ...]] = {}
    if secret:
        links = {role.job_id: _links_for(config, secret, role.job_id)
                 for role in state.open_roles}
    return HTMLResponse(
        render("dashboard.html", dashboard=state, links=links,
               signed=bool(secret), expired=expired is not None)
    )


@app.get("/a/{token}", response_class=HTMLResponse)
def confirm(token: str) -> Response:
    """Shows what a signed link is asking and offers the button that carries it out."""
    config = load_all()
    resolved = _resolve(config, token)
    if not isinstance(resolved, SignedAction):
        return resolved

    with session_scope() as session:
        try:
            card = job_card(session, resolved.job_id, config.search_profile.scoring.prompt_version)
        except UnknownJob:
            return _gone(resolved.job_id)

    confirmable, explanation = can_confirm(resolved.action, card)
    return HTMLResponse(
        render(
            "confirm.html",
            card=card,
            token=token,
            action=resolved.action.value,
            label=ACTION_LABELS[resolved.action],
            confirmable=confirmable,
            explanation=explanation,
        )
    )


@app.post("/a/{token}", response_class=HTMLResponse)
def execute(token: str) -> Response:
    """Carries out one confirmed action. Posting the same link twice changes nothing."""
    config = load_all()
    resolved = _resolve(config, token)
    if not isinstance(resolved, SignedAction):
        return resolved

    try:
        card, outcome = _act(config, resolved)
    except UnknownJob:
        return _gone(resolved.job_id)
    except IntegrityError:
        # Two requests raced and both found no application row, so the loser hit
        # the unique constraint on `applications.job_id`. The row it wanted now
        # exists, so running the same action again takes the "already recorded"
        # branch. A double-click has to be the no-op and one retry is enough
        # because the second attempt can no longer be the one that inserts.
        log.info("action %s on job %s: retried after a concurrent write",
                 resolved.action.value, resolved.job_id)
        card, outcome = _act(config, resolved)

    log.info(
        "action %s on job %s: %s", resolved.action.value, resolved.job_id, outcome.headline
    )
    return HTMLResponse(render("outcome.html", card=card, outcome=outcome))


def _act(config: AppConfig, resolved: SignedAction) -> tuple[JobCard, Outcome]:
    """Read the job and then do what the token asked."""
    with session_scope() as session:
        # A link naming a job that has since been deleted must not create an
        # application row for it, which is why the card is read first.
        card = job_card(session, resolved.job_id, config.search_profile.scoring.prompt_version)
        return card, perform(session, resolved.action, resolved.job_id)


# ---------------------------------------------------------------------------
# Shared by both halves of /a/{token}
# ---------------------------------------------------------------------------


def _resolve(config: AppConfig, token: str) -> SignedAction | Response:
    """Verifies one token or the page to serve instead of honouring it."""
    secret = config.secrets.optional("action_token_secret")
    if not secret:
        return _notice(
            "This installation cannot check links.",
            "ACTION_TOKEN_SECRET is not set, so no link can be verified. Add it to "
            "`.env` (see `.env.example`) and restart. Note that generating a new "
            "secret invalidates every link already sitting in your inbox.",
            status=500,
        )
    try:
        return verify(secret, token)
    except ExpiredToken:
        # Genuinely ours and simply too old. The dashboard says so and shows the
        # role if it is still open.
        return RedirectResponse("/?expired=1", status_code=SEE_OTHER)
    except TokenError as exc:
        return _notice(
            "This link cannot be trusted.",
            f"{exc} Nothing was changed. If you typed or edited the address, open the "
            f"link from the email instead.",
            status=400,
        )


def _links_for(config: AppConfig, secret: str, job_id: int) -> tuple[ActionLink, ...]:
    """The signed links printed beside one role on the dashboard."""
    return action_links(
        config.system.base_url,
        secret,
        job_id,
        ttl_days=config.system.actions.token_ttl_days,
        only=DASHBOARD_ACTIONS,
    )


def _gone(job_id: int) -> HTMLResponse:
    """The page for a link naming a job this database no longer holds."""
    return _notice(
        "That job is no longer here.",
        f"The link is valid, but job {job_id} is not in the database any more. "
        f"Nothing was changed.",
        status=404,
    )


def _notice(headline: str, detail: str, *, status: int) -> HTMLResponse:
    """One short page saying what could not be done and why."""
    return HTMLResponse(
        render("notice.html", headline=headline, detail=detail), status_code=status
    )
