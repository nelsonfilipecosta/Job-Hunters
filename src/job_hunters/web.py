"""The web application with the signed action links and the dashboard behind them.

Four routes:

    GET  /health      liveness (touching nothing)
    GET  /            the dashboard
    GET  /a/{token}   what one signed link asks about a job and a button
    POST /a/{token}   what that button does
    GET  /c/{token}   the same for one discovered company with the watchlist line to be
    POST /c/{token}   what that button does

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
from dataclasses import dataclass
from urllib.parse import parse_qsl

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy.exc import IntegrityError

from . import paths
from .actions import (
    ACTION_LABELS,
    CANDIDATE_LABELS,
    QUEUE_DECISIONS,
    Action,
    ActionLink,
    CandidateAction,
    ExpiredToken,
    SignedAction,
    SignedCandidateAction,
    TokenError,
    action_links,
    candidate_links,
    candidate_url,
    verify,
    verify_candidate,
)
from .config import AppConfig, ConfigError, load_all
from .db import SchemaError, init_db, session_scope
from .promote import (
    PromoteError,
    Queued,
    approve,
    queued,
    reject,
    review_queue,
    unreject,
)
from .tables import CandidateCompany, CandidateStatus, Tier
from .templating import render
from .tiering import suggest_tier
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

# A company with no board found cannot be approved, so it is only offered the
# decision it can take.
REJECT_ONLY: tuple[CandidateAction, ...] = (CandidateAction.REJECT,)

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
    """What is outstanding, where each application stands, how they go and who is queued."""
    config = load_all()
    secret = config.secrets.optional("action_token_secret")
    with session_scope() as session:
        state = build_dashboard(session, config)
        # `review_queue` reconciles against `companies_watchlist.yaml` first, so a
        # company approved elsewhere has already left the queue by the time it is drawn.
        queue = tuple(queued(candidate) for candidate in review_queue(session))
    links: dict[int, tuple[ActionLink, ...]] = {}
    decisions: dict[int, tuple[ActionLink, ...]] = {}
    if secret:
        links = {role.job_id: _links_for(config, secret, role.job_id)
                 for role in state.open_roles}
        decisions = {entry.id: _decisions_for(config, secret, entry) for entry in queue}
    return HTMLResponse(
        render(
            "dashboard.html",
            dashboard=state,
            links=links,
            signed=bool(secret),
            expired=expired is not None,
            discovered=tuple(entry for entry in queue if entry.has_board),
            boardless=tuple(entry for entry in queue if not entry.has_board),
            decisions=decisions,
            ingest_schedule=config.system.schedules.ingest,
        )
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


@app.get("/c/{token}", response_class=HTMLResponse)
def confirm_candidate(token: str) -> Response:
    """Shows one queued company, the line the watchlist would gain and the button."""
    config = load_all()
    resolved = _resolve_candidate(config, token)
    if not isinstance(resolved, SignedCandidateAction):
        return resolved

    with session_scope() as session:
        candidate = session.get(CandidateCompany, resolved.candidate_id)
        if candidate is None:
            return _candidate_gone(resolved.candidate_id)
        entry, refusal = queued(candidate), _refusal(resolved.action, candidate)

    # Only for an approval that can still happen: a decided company needs no tier
    # and the reviewer should not wait on a model call to be told so.
    suggestion = None
    if resolved.action is CandidateAction.APPROVE and refusal is None:
        suggestion = suggest_tier(entry, config)
    return _candidate_page(entry, token, resolved.action, suggestion=suggestion, refusal=refusal)


@app.post("/c/{token}", response_class=HTMLResponse)
async def decide_candidate(request: Request, token: str) -> Response:
    """Approves or rejects one queued company. Posting the same link twice changes nothing."""
    config = load_all()
    resolved = _resolve_candidate(config, token)
    if not isinstance(resolved, SignedCandidateAction):
        return resolved
    # Read as a plain urlencoded body rather than through `Form(...)`, which would
    # pull in `python-multipart` for three text fields this page already controls.
    form = dict(parse_qsl((await request.body()).decode("utf-8")))

    with session_scope() as session:
        candidate = session.get(CandidateCompany, resolved.candidate_id)
        if candidate is None:
            return _candidate_gone(resolved.candidate_id)
        entry = queued(candidate)
        refusal = _refusal(resolved.action, candidate)
        if refusal is not None:
            return _candidate_page(entry, token, resolved.action, refusal=refusal)
        try:
            outcome = _decide(session, candidate, resolved.action, form, config)
        except PromoteError as exc:
            # Recoverable: the company is still pending and still has a board, so
            # the same page with another slug would work. The fields come back filled.
            log.info("candidate %s refused: %s", candidate.name, exc)
            return _candidate_page(
                entry, token, resolved.action, refusal=str(exc), chosen=form, confirmable=True
            )
        log.info("candidate %s: %s", entry.name, outcome.headline)

    return HTMLResponse(render("candidate_outcome.html", candidate=entry, outcome=outcome))


@dataclass(frozen=True)
class CandidateOutcome:
    """What a confirmed decision about a company did."""

    headline: str
    detail: str
    changed: bool
    line: str | None = None
    undo_url: str | None = None


def _decide(
    session,
    candidate: CandidateCompany,
    action: CandidateAction,
    form: dict[str, str],
    config: AppConfig,
) -> CandidateOutcome:
    """Carries out one decision, raising `PromoteError` with the reason it could not."""
    if action is CandidateAction.REJECT:
        reject(session, str(candidate.id))
        return CandidateOutcome(
            headline=f"{candidate.name} rejected.",
            detail="It leaves the queue for good. Later sightings still count against "
                   "its row, but it is never queued again.",
            changed=True,
            undo_url=_undo_url(config, candidate.id),
        )
    if action is CandidateAction.UNREJECT:
        unreject(session, str(candidate.id))
        return CandidateOutcome(
            headline=f"{candidate.name} is back in the queue.",
            detail="The rejection is undone and nothing else changed. It is waiting for "
                   "a decision again.",
            changed=True,
        )
    tier = form.get("tier", "")
    done = approve(
        session,
        str(candidate.id),
        slug=(form.get("slug") or "").strip() or None,
        name=(form.get("name") or "").strip() or None,
        # A hand-made POST could name a tier that is not one. The dropdown cannot.
        tier=tier if tier in set(Tier) else Tier.DISCOVERED,
    )
    return CandidateOutcome(
        headline=f"{done.candidate.name} added to the watchlist.",
        detail=f"{done.path.name} gained the line below. Its board is fetched by the "
               f"next ingest and its postings are then judged like any other.",
        changed=True,
        line=done.line,
    )


def _refusal(action: CandidateAction, candidate: CandidateCompany) -> str | None:
    """Why this decision cannot be taken. Read from the state rather than from the link."""
    decided = candidate.status
    if action is CandidateAction.APPROVE:
        if decided == CandidateStatus.APPROVED:
            return (
                f"{candidate.name} is already in the watchlist as "
                f"{candidate.slug or 'an entry'}."
            )
        if decided == CandidateStatus.REJECTED:
            return f"{candidate.name} was rejected, so it is no longer in the queue."
        if not candidate.ats_type or not candidate.ats_token:
            return (
                f"No Greenhouse, Lever or Ashby board was found for {candidate.name}, "
                f"so there is nothing to watch yet. It is probed again every week."
            )
        return None
    if action is CandidateAction.UNREJECT:
        if decided == CandidateStatus.APPROVED:
            return (
                f"{candidate.name} is in the watchlist. Remove its line from "
                f"`companies_watchlist.yaml` to stop watching it."
            )
        if decided != CandidateStatus.REJECTED:
            return f"{candidate.name} is already in the queue, so there is nothing to undo."
        return None
    if decided == CandidateStatus.APPROVED:
        return (
            f"{candidate.name} is in the watchlist. Remove its line from "
            f"`companies_watchlist.yaml` to stop watching it."
        )
    if decided == CandidateStatus.REJECTED:
        return f"{candidate.name} was already rejected."
    return None


def _candidate_page(
    entry: Queued,
    token: str,
    action: CandidateAction,
    *,
    suggestion: tuple[Tier, str] | None = None,
    refusal: str | None = None,
    chosen: dict[str, str] | None = None,
    confirmable: bool | None = None,
) -> HTMLResponse:
    """The confirm page for a company with the form filled in and any refusal above it."""
    chosen = chosen or {}
    tier, because = suggestion or (Tier.DISCOVERED, "")
    return HTMLResponse(
        render(
            "candidate_confirm.html",
            candidate=entry,
            token=token,
            action=action.value,
            label=CANDIDATE_LABELS[action],
            refusal=refusal,
            confirmable=refusal is None if confirmable is None else confirmable,
            tiers=[t.value for t in Tier],
            slug=chosen.get("slug") or entry.slug,
            name=chosen.get("name") or entry.name,
            tier=chosen.get("tier") or tier.value,
            because=because,
            suggested=suggestion is not None,
        )
    )


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


def _resolve_candidate(config: AppConfig, token: str) -> SignedCandidateAction | Response:
    """Verifies a company token or the page to serve instead of honouring it."""
    secret = config.secrets.optional("action_token_secret")
    if not secret:
        return _notice(
            "This installation cannot check links.",
            "ACTION_TOKEN_SECRET is not set, so no link can be verified. Add it to "
            "`.env` (see `.env.example`) and restart.",
            status=500,
        )
    try:
        return verify_candidate(secret, token)
    except ExpiredToken:
        return RedirectResponse("/?expired=1", status_code=SEE_OTHER)
    except TokenError as exc:
        return _notice(
            "This link cannot be trusted.",
            f"{exc} Nothing was changed. Open the review queue on the dashboard instead.",
            status=400,
        )


def _decisions_for(config: AppConfig, secret: str, entry: Queued) -> tuple[ActionLink, ...]:
    """The signed links printed beside one queued company."""
    return candidate_links(
        config.system.base_url,
        secret,
        entry.id,
        ttl_days=config.system.actions.token_ttl_days,
        only=QUEUE_DECISIONS if entry.has_board else REJECT_ONLY,
    )


def _undo_url(config: AppConfig, candidate_id: int) -> str | None:
    """The link that puts one rejection back signed like every other decision."""
    secret = config.secrets.optional("action_token_secret")
    if not secret:
        return None
    return candidate_url(
        config.system.base_url,
        secret,
        CandidateAction.UNREJECT,
        candidate_id,
        ttl_days=config.system.actions.token_ttl_days,
    )


def _candidate_gone(candidate_id: int) -> HTMLResponse:
    """The page for a link naming a company this database no longer holds."""
    return _notice(
        "That company is no longer here.",
        f"The link is valid, but company {candidate_id} is not in the queue any more. "
        f"Nothing was changed.",
        status=404,
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
