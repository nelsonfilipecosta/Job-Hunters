"""The action links the digest puts in an email.

Every entry in the email carries four actions: draft a CV, draft a cover
letter, mark the job applied and dismiss it. Clicking one changes the database,
so a link has to survive a trip through an inbox and come back unaltered."""

from __future__ import annotations

import base64
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256


class Action(StrEnum):
    DRAFT_CV = "draft_cv"
    DRAFT_COVER_LETTER = "draft_cover_letter"
    APPLIED = "applied"
    DISMISS = "dismiss"


ACTION_LABELS: dict[Action, str] = {
    Action.DRAFT_CV: "Draft CV",
    Action.DRAFT_COVER_LETTER: "Draft Cover Letter",
    Action.APPLIED: "Applied",
    Action.DISMISS: "Dismiss",
}


class TokenError(Exception):
    """A token that cannot be trusted (malformed or signed with another secret)."""


class ExpiredToken(TokenError):
    """A token whose signature is genuine but whose expiry has passed."""


@dataclass(frozen=True)
class SignedAction:
    """What a verified token was asking for."""

    action: Action
    job_id: int
    expires_at: datetime


def _b64(raw: bytes) -> str:
    """URL-safe base64 without the `=` padding, which is noise in a link."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    """Reverses `_b64` and restores the padding base64 needs to decode."""
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(secret: str, payload: bytes) -> str:
    """The HMAC-SHA256 of one payload under the secret."""
    return _b64(hmac.new(secret.encode("utf-8"), payload, sha256).digest())


def _matches(signature: str, expected: str) -> bool:
    """Constant-time comparison of a signature we were handed against the real one."""
    return hmac.compare_digest(signature.encode("utf-8"), expected.encode("ascii"))


def sign(
    secret: str,
    action: Action | str,
    job_id: int,
    *,
    ttl_days: int,
    now: datetime | None = None,
) -> str:
    """One token authorizing one action on one job for `ttl_days` days."""
    action = Action(action)
    expires_at = (now or datetime.now(UTC)) + timedelta(days=ttl_days)
    payload = f"{action.value}:{job_id}:{int(expires_at.timestamp())}".encode()
    return f"{_b64(payload)}.{_signature(secret, payload)}"


def verify(secret: str, token: str, *, now: datetime | None = None) -> SignedAction:
    """Reads a token back or raises if it was forged, damaged or has expired."""
    encoded, _, signature = token.partition(".")
    if not encoded or not signature:
        raise TokenError("Malformed action token.")
    try:
        # `binascii.Error` is a subclass of ValueError, so this covers both
        # invalid characters and a length base64 cannot decode.
        payload = _unb64(encoded)
    except ValueError as exc:
        raise TokenError("Malformed action token.") from exc
    if not _matches(signature, _signature(secret, payload)):
        raise TokenError("This link was not signed by this installation.")

    try:
        name, job_id, expires = payload.decode("utf-8").split(":")
        signed = SignedAction(
            action=Action(name),
            job_id=int(job_id),
            expires_at=datetime.fromtimestamp(int(expires), UTC),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        # The signature held, so this is our own token in a shape this version
        # no longer understands rather than anything an outsider produced.
        raise TokenError(f"Unreadable action token: {exc}") from exc

    if signed.expires_at <= (now or datetime.now(UTC)):
        raise ExpiredToken(f"This link expired on {signed.expires_at:%Y-%m-%d}.")
    return signed


def action_url(
    base_url: str, secret: str, action: Action | str, job_id: int, *, ttl_days: int, **kwargs
) -> str:
    """The full link to put in the email, built from the declared `base_url`."""
    return f"{base_url.rstrip('/')}/a/{sign(secret, action, job_id, ttl_days=ttl_days, **kwargs)}"
