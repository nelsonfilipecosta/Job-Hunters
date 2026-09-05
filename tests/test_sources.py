"""Tests for the three ATS adapters against recorded responses.

No test here touches the network. Each adapter is given an `httpx2.Client` whose
transport answers from `tests/fixtures/boards/*.json` (real payloads recorded
from live boards) or with the failure being tested.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx2
import pytest

from job_hunters.models import Company, FetchStatus
from job_hunters.sources import (
    ADAPTERS,
    AshbyAdapter,
    GreenhouseAdapter,
    LeverAdapter,
    get_adapter,
)

BOARDS = Path(__file__).parent / "fixtures" / "boards"


def _client_returning(payload, status: int = 200) -> httpx2.Client:
    """A client that answers every request with one canned JSON body."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Answers with the configured payload and status, ignoring the request."""
        return httpx2.Response(status, json=payload)

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def _client_raising(exc: Exception) -> httpx2.Client:
    """A client whose every request raises this exception instead of answering."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Raises the client's configured exception."""
        raise exc

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def _company(ats: str, token: str = "acme") -> Company:
    """An unsaved Company row configured for one ATS type and token."""
    return Company(slug="acme", name="Acme", ats_type=ats, ats_config={"token": token})


def test_registry_covers_every_ats_the_config_allows() -> None:
    """The adapter registry has exactly the three ATS types the config permits."""
    assert set(ADAPTERS) == {"greenhouse", "lever", "ashby"}
    assert isinstance(get_adapter("ashby"), AshbyAdapter)


def test_unknown_ats_type_is_a_readable_error() -> None:
    """Asking for an unregistered ATS type raises a KeyError naming it."""
    with pytest.raises(KeyError, match="(?i)no adapter for ats_type 'workday'"):
        get_adapter("workday")


def test_greenhouse_parses_a_real_board() -> None:
    """A recorded Greenhouse board parses into the expected postings and fields."""
    payload = json.loads((BOARDS / "greenhouse_imbue.json").read_text())
    result = GreenhouseAdapter(_client_returning(payload)).fetch(_company("greenhouse"))
    assert result.status == FetchStatus.OK
    assert len(result.items) == len(payload["jobs"]) == 3
    first = result.items[0]
    assert first.source == "greenhouse"
    assert first.source_job_id == str(payload["jobs"][0]["id"])
    assert first.title == payload["jobs"][0]["title"]
    assert first.url.startswith("https://")
    assert first.location_raw == "San Francisco"
    assert isinstance(first.posted_at, datetime)


def test_greenhouse_unescapes_the_double_encoded_content() -> None:
    """The raw `content` is `&lt;h2&gt;...`, but the posting must carry readable text."""
    payload = json.loads((BOARDS / "greenhouse_imbue.json").read_text())
    result = GreenhouseAdapter(_client_returning(payload)).fetch(_company("greenhouse"))
    text = result.items[0].description
    assert "&lt;" not in text and "<h2>" not in text and "<p>" not in text
    assert "Summary" in text


def test_lever_parses_a_real_board_with_its_structured_hints() -> None:
    """A recorded Lever board parses, including its country code and workplace type."""
    payload = json.loads((BOARDS / "lever_palantir.json").read_text())
    result = LeverAdapter(_client_returning(payload)).fetch(_company("lever"))
    assert result.status == FetchStatus.OK
    assert len(result.items) == 6
    by_loc = {p.location_raw: p for p in result.items}
    london = by_loc["London, United Kingdom"]
    assert london.country_code == "GB" and london.workplace_type == "hybrid"
    assert london.source_job_id == payload[0]["id"]
    assert london.title == payload[0]["text"]
    assert isinstance(london.posted_at, datetime)


def test_lever_joins_the_split_description_into_one_document() -> None:
    """Lever's three separate description fields are joined into one document."""
    payload = json.loads((BOARDS / "lever_palantir.json").read_text())
    result = LeverAdapter(_client_returning(payload)).fetch(_company("lever"))
    posting = result.items[0]
    raw = payload[0]
    assert raw["descriptionPlain"].strip()[:40] in posting.description
    assert raw["lists"][0]["text"] in posting.description
    assert "<li>" not in posting.description
    assert raw["additionalPlain"].strip()[:30] in posting.description


def test_lever_rejects_a_non_list_payload_as_a_failed_fetch() -> None:
    """A payload that isn't a JSON list is a failed fetch and not a crash."""
    result = LeverAdapter(_client_returning({"error": "nope"})).fetch(_company("lever"))
    assert result.status == FetchStatus.FAILED and "list" in result.error


def test_ashby_parses_a_real_board() -> None:
    """A recorded Ashby board parses into the expected postings and fields."""
    payload = json.loads((BOARDS / "ashby_reka.json").read_text())
    result = AshbyAdapter(_client_returning(payload)).fetch(_company("ashby"))
    assert result.status == FetchStatus.OK
    assert len(result.items) == 9
    first = result.items[0]
    assert first.source_job_id == payload["jobs"][0]["id"]
    assert first.location_raw == payload["jobs"][0]["location"]
    assert first.workplace_type == payload["jobs"][0]["workplaceType"]
    assert first.is_remote == payload["jobs"][0]["isRemote"]
    assert first.description and "<" not in first.description[:200]


def test_ashby_skips_unlisted_postings() -> None:
    """A posting marked `isListed: false` is dropped from the results."""
    payload = json.loads((BOARDS / "ashby_reka.json").read_text())
    payload["jobs"][0]["isListed"] = False
    result = AshbyAdapter(_client_returning(payload)).fetch(_company("ashby"))
    assert len(result.items) == 8


@pytest.mark.parametrize("adapter_cls", [GreenhouseAdapter, LeverAdapter, AshbyAdapter])
def test_a_404_is_a_failed_fetch_not_an_exception(adapter_cls) -> None:
    """A renamed board token must come back as `failed` so nothing gets closed."""
    result = adapter_cls(_client_returning({"error": "not found"}, status=404)).fetch(
        _company(adapter_cls.source)
    )
    assert result.status == FetchStatus.FAILED
    assert "404" in result.error
    assert result.items == ()


@pytest.mark.parametrize("adapter_cls", [GreenhouseAdapter, LeverAdapter, AshbyAdapter])
def test_a_network_error_is_a_failed_fetch(adapter_cls) -> None:
    """A connection error is a failed fetch for every adapter, not an exception."""
    client = _client_raising(httpx2.ConnectError("boom"))
    result = adapter_cls(client).fetch(_company(adapter_cls.source))
    assert result.status == FetchStatus.FAILED and "ConnectError" in result.error


@pytest.mark.parametrize("adapter_cls", [GreenhouseAdapter, LeverAdapter, AshbyAdapter])
def test_an_unexpected_shape_is_a_failed_fetch(adapter_cls) -> None:
    """A 200 with the wrong JSON must not be mistaken for an empty board."""
    result = adapter_cls(_client_returning({"unexpected": True})).fetch(
        _company(adapter_cls.source)
    )
    assert result.status == FetchStatus.FAILED


@pytest.mark.parametrize("adapter_cls", [GreenhouseAdapter, LeverAdapter, AshbyAdapter])
def test_a_missing_token_is_a_failed_fetch(adapter_cls) -> None:
    """A company with no ATS token in its config is a failed fetch."""
    company = Company(slug="acme", name="Acme", ats_type=adapter_cls.source, ats_config={})
    result = adapter_cls(_client_returning({})).fetch(company)
    assert result.status == FetchStatus.FAILED and "token" in result.error


def test_an_empty_board_is_ok_with_no_items_not_a_failure() -> None:
    """Empty and failed are different outcomes and ingest treats them differently."""
    result = GreenhouseAdapter(_client_returning({"jobs": []})).fetch(_company("greenhouse"))
    assert result.status == FetchStatus.OK and result.items == ()
