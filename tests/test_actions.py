"""Tests for the action links the digest puts in an email."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from job_hunters.actions import (
    Action,
    ExpiredToken,
    TokenError,
    action_url,
    sign,
    verify,
)

SECRET = "a-secret-nobody-else-has"
NOW = datetime(2026, 9, 9, 8, 0, tzinfo=UTC)
TTL = 90


def test_a_token_survives_a_round_trip() -> None:
    """What was signed is what comes back."""
    signed = verify(SECRET, sign(SECRET, Action.APPLIED, 42, ttl_days=TTL, now=NOW), now=NOW)
    assert signed.action is Action.APPLIED
    assert signed.job_id == 42


def test_every_action_can_be_signed() -> None:
    """All four links in an entry round-trip and not only the one the tests reach for."""
    for action in Action:
        token = sign(SECRET, action, 7, ttl_days=TTL, now=NOW)
        assert verify(SECRET, token, now=NOW).action is action


def test_a_token_signed_with_another_secret_is_refused() -> None:
    """The secret is what makes a link ours rather than anyone's."""
    token = sign("someone-elses-secret", Action.DISMISS, 42, ttl_days=TTL, now=NOW)
    with pytest.raises(TokenError):
        verify(SECRET, token, now=NOW)


def test_an_edited_job_id_is_refused() -> None:
    """Retargeting a link at another job must not produce a token that verifies."""
    honest = sign(SECRET, Action.APPLIED, 42, ttl_days=TTL, now=NOW)
    payload, _, signature = honest.partition(".")
    forged = sign(SECRET, Action.APPLIED, 99, ttl_days=TTL, now=NOW).partition(".")[0]
    with pytest.raises(TokenError):
        verify(SECRET, f"{forged}.{signature}", now=NOW)
    assert payload != forged, "the two payloads really do differ"


@pytest.mark.parametrize(
    "token",
    [
        "", ".", "not-a-token", "YXBwbGllZDo0Mg", "!!!.!!!",
        # Non-ASCII, which is the one shape that reaches `compare_digest`.
        "YXBwbGllZDo0Mg.ü", "ü.ü", "YXBwbGllZDo0Mg.\N{SNOWMAN}",
        # A payload that verifies as bytes but is not a token this code wrote.
        "AAAA.AAAA", "YXBwbGllZDo0Mjo=.x",
    ],
)
def test_a_malformed_token_is_an_error_and_not_a_crash(token: str) -> None:
    """A damaged link is refused the same way a forged one is."""
    with pytest.raises(TokenError):
        verify(SECRET, token, now=NOW)


def test_a_token_whose_payload_is_ours_but_unreadable_is_refused() -> None:
    """A signature that holds over a payload this version cannot parse is still refused."""
    import base64

    payload = b"applied:not-a-number:0"
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    from job_hunters.actions import _signature

    with pytest.raises(TokenError):
        verify(SECRET, f"{encoded}.{_signature(SECRET, payload)}", now=NOW)


def test_an_expired_token_is_told_apart_from_a_forged_one() -> None:
    """A real link that sat too long is redirected while a forgery is refused."""
    token = sign(SECRET, Action.APPLIED, 42, now=NOW, ttl_days=1)
    with pytest.raises(ExpiredToken):
        verify(SECRET, token, now=NOW + timedelta(days=2))


def test_a_token_is_still_valid_the_day_before_it_expires() -> None:
    """A real link is valid during its whole lifetime."""
    token = sign(SECRET, Action.APPLIED, 42, ttl_days=TTL, now=NOW)
    assert verify(SECRET, token, now=NOW + timedelta(days=TTL - 1)).job_id == 42


def test_the_lifetime_is_whatever_the_caller_asked_for() -> None:
    """Nothing here holds a default, so a shortened config value really does shorten a link."""
    token = sign(SECRET, Action.APPLIED, 42, ttl_days=7, now=NOW)
    assert verify(SECRET, token, now=NOW + timedelta(days=6)).job_id == 42
    with pytest.raises(ExpiredToken):
        verify(SECRET, token, now=NOW + timedelta(days=8))


def test_a_token_is_url_safe() -> None:
    """A token goes in a path, so it may not contain a character that needs escaping."""
    token = sign(SECRET, Action.DRAFT_COVER_LETTER, 123456, ttl_days=TTL, now=NOW)
    assert set(token) <= set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    )


def test_the_link_is_built_from_the_declared_base_url() -> None:
    """Links are built from config and never from the container's own view of itself."""
    url = action_url("http://localhost:8000/", SECRET, Action.DISMISS, 42, ttl_days=TTL, now=NOW)
    assert url.startswith("http://localhost:8000/a/"), "one slash, from a base url with one too many"
    assert verify(SECRET, url.rsplit("/", 1)[1], now=NOW).job_id == 42
