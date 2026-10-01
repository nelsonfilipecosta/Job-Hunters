"""Tests for the signed links that carry a decision back to this installation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from job_hunters.actions import (
    CANDIDATE_LABELS,
    Action,
    CandidateAction,
    ExpiredToken,
    TokenError,
    action_url,
    candidate_links,
    candidate_url,
    sign,
    sign_candidate,
    verify,
    verify_candidate,
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
        # Non-ASCII in the signature half, which compares false rather than raising.
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


def _signed_payload(payload: bytes) -> str:
    """A token whose signature holds, whatever the payload inside it says."""
    from job_hunters.actions import _b64, _signature

    return f"{_b64(payload)}.{_signature(SECRET, payload)}"


def test_a_candidate_token_survives_a_round_trip() -> None:
    """What was signed is what comes back."""
    token = sign_candidate(SECRET, CandidateAction.APPROVE, 42, ttl_days=TTL, now=NOW)
    signed = verify_candidate(SECRET, token, now=NOW)
    assert signed.action is CandidateAction.APPROVE
    assert signed.candidate_id == 42


def test_every_candidate_decision_can_be_signed() -> None:
    """All three round-trip and not only the one the other tests reach for."""
    for action in CandidateAction:
        token = sign_candidate(SECRET, action, 7, ttl_days=TTL, now=NOW)
        assert verify_candidate(SECRET, token, now=NOW).action is action


def test_a_job_token_is_not_a_candidate_token() -> None:
    """Company 15 and job 15 are both id 15 and one secret signs both kinds of link."""
    job = sign(SECRET, Action.APPLIED, 15, ttl_days=TTL, now=NOW)
    candidate = sign_candidate(SECRET, CandidateAction.APPROVE, 15, ttl_days=TTL, now=NOW)
    with pytest.raises(TokenError):
        verify_candidate(SECRET, job, now=NOW)
    with pytest.raises(TokenError):
        verify(SECRET, candidate, now=NOW)


def test_a_candidate_token_signed_with_another_secret_is_refused() -> None:
    """The secret is what makes a link ours rather than anyone's."""
    token = sign_candidate("someone-elses-secret", CandidateAction.REJECT, 42, ttl_days=TTL, now=NOW)
    with pytest.raises(TokenError):
        verify_candidate(SECRET, token, now=NOW)


def test_an_edited_candidate_id_is_refused() -> None:
    """Retargeting a link at another company must not produce a token that verifies."""
    honest = sign_candidate(SECRET, CandidateAction.APPROVE, 42, ttl_days=TTL, now=NOW)
    payload, _, signature = honest.partition(".")
    forged = sign_candidate(SECRET, CandidateAction.APPROVE, 99, ttl_days=TTL, now=NOW)
    forged = forged.partition(".")[0]
    with pytest.raises(TokenError):
        verify_candidate(SECRET, f"{forged}.{signature}", now=NOW)
    assert payload != forged, "the two payloads really do differ"


@pytest.mark.parametrize(
    "token",
    [
        "", ".", "not-a-token", "!!!.!!!",
        # Base64 that decodes, with no signature half at all.
        "Y2FuZGlkYXRlOmFwcHJvdmU6NDI",
        # Non-ASCII in the signature half, which compares false rather than raising.
        "Y2FuZGlkYXRlOmFwcHJvdmU6NDI.ü", "ü.ü",
        # A payload that decodes as bytes but is not a token this code wrote.
        "AAAA.AAAA",
    ],
)
def test_a_malformed_candidate_token_is_an_error_and_not_a_crash(token: str) -> None:
    """A damaged link is refused the same way a forged one is."""
    with pytest.raises(TokenError):
        verify_candidate(SECRET, token, now=NOW)


@pytest.mark.parametrize(
    "payload",
    [
        b"candidate:approve:not-a-number:0",
        b"candidate:approve:42",
        b"candidate:destroy:42:99999999999",
        b"job:approve:42:99999999999",
    ],
)
def test_a_candidate_payload_that_is_ours_but_unreadable_is_refused(payload: bytes) -> None:
    """A signature that holds over a payload this version cannot read is still refused."""
    with pytest.raises(TokenError):
        verify_candidate(SECRET, _signed_payload(payload), now=NOW)


def test_a_candidate_token_lasts_exactly_as_long_as_it_was_given() -> None:
    """Undo is offered in a page that stays open, so a link outliving its welcome is the risk."""
    token = sign_candidate(SECRET, CandidateAction.UNREJECT, 42, ttl_days=7, now=NOW)
    assert verify_candidate(SECRET, token, now=NOW + timedelta(days=6)).candidate_id == 42
    with pytest.raises(ExpiredToken):
        verify_candidate(SECRET, token, now=NOW + timedelta(days=8))


def test_the_candidate_link_is_built_from_the_declared_base_url() -> None:
    """Links are built from config and never from the container's own view of itself."""
    url = candidate_url(
        "http://localhost:8000/", SECRET, CandidateAction.APPROVE, 42, ttl_days=TTL, now=NOW
    )
    assert url.startswith("http://localhost:8000/c/"), "one slash, from a base url with one too many"
    token = url.rsplit("/", 1)[1]
    assert verify_candidate(SECRET, token, now=NOW).candidate_id == 42
    assert set(token) <= set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    ), "a token goes in a path, so it may not contain a character that needs escaping"


def test_the_decisions_are_labelled_for_the_reviewer() -> None:
    """`unreject` is what the code calls it and `Undo` is what the reviewer is offered."""
    links = candidate_links(
        "http://localhost:8000", SECRET, 42, ttl_days=TTL, only=tuple(CandidateAction), now=NOW
    )
    assert [link.label for link in links] == ["Approve", "Reject", "Undo"]


def test_a_queued_company_is_not_offered_an_undo_it_has_no_use_for() -> None:
    """Nothing has been decided about a row in the queue, so the default leaves Undo out."""
    links = candidate_links("http://localhost:8000", SECRET, 42, ttl_days=TTL, now=NOW)
    assert [link.label for link in links] == ["Approve", "Reject"]


def test_every_decision_has_a_label_to_print() -> None:
    """`candidate_links` reads this table by key, so a decision added without one would raise."""
    assert set(CANDIDATE_LABELS) == set(CandidateAction)


def test_a_row_is_only_linked_to_the_decisions_it_was_asked_for() -> None:
    """A company with no board cannot be approved, so its row is not offered the link."""
    links = candidate_links(
        "http://localhost:8000", SECRET, 42, ttl_days=TTL, only=(CandidateAction.REJECT,), now=NOW
    )
    assert [link.label for link in links] == ["Reject"]
    signed = verify_candidate(SECRET, links[0].url.rsplit("/", 1)[1], now=NOW)
    assert signed.action is CandidateAction.REJECT and signed.candidate_id == 42
