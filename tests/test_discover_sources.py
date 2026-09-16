"""Tests for the four `discover` sources against recorded responses.

No test here touches the network. Each source is given an `httpx2.Client`
whose transport answers from `tests/fixtures/discover/*.json` (real payloads
recorded from the live sites) or with the failure being tested.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx2
import pytest

from job_hunters.tables import FetchStatus
from job_hunters.sources import (
    ADAPTERS,
    DISCOVER_SOURCES,
    ArbeitnowSource,
    HackerNewsSource,
    RemoteOkSource,
    RemotiveSource,
    get_discover_source,
    replay_posting,
)
from job_hunters.sources.hn import ITEM_URL, SEARCH_URL

FIXTURES = Path(__file__).parent / "fixtures" / "discover"


def _load(name: str) -> dict | list:
    """One recorded payload by file name."""
    return json.loads((FIXTURES / name).read_text())


def _client_returning(payload, status: int = 200) -> httpx2.Client:
    """A client that answers every request with one canned JSON body."""
    return httpx2.Client(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(status, json=payload))
    )


def _hn_client(search=None, thread=None) -> httpx2.Client:
    """A client answering the search and the thread URLs from the fixtures unless told otherwise."""
    search = search if search is not None else _load("hn_search.json")
    thread = thread if thread is not None else _load("hn_thread.json")

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Routes the two Algolia URLs to their payloads and 404s anything else."""
        url = str(request.url)
        if url == SEARCH_URL:
            return httpx2.Response(200, json=search)
        if url == ITEM_URL.format(id=49522897):
            return httpx2.Response(200, json=thread)
        return httpx2.Response(404, json={"error": "not found"})

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_the_discover_registry_names_the_four_sources_and_not_the_boards() -> None:
    """The `discover` sources and the ATS adapters are two registries with nothing in common."""
    assert set(DISCOVER_SOURCES) == {"hn", "remoteok", "arbeitnow", "remotive"}
    assert not set(DISCOVER_SOURCES) & set(ADAPTERS)
    assert isinstance(get_discover_source("remotive"), RemotiveSource)


def test_an_unknown_discover_source_is_a_readable_error() -> None:
    """Asking for an unregistered source raises a KeyError naming it."""
    with pytest.raises(KeyError, match="(?i)no discover source named 'linkedin'"):
        get_discover_source("linkedin")


def test_hn_reads_this_months_hiring_thread_and_skips_its_siblings() -> None:
    """The "Who wants to be hired?" thread is listed first and must not be the one read."""
    result = HackerNewsSource(_hn_client()).fetch()
    assert result.status == FetchStatus.OK
    ids = {item.source_job_id for item in result.items}
    assert ids == {"49522903", "49524060", "49537584", "49559854"}


def test_hn_takes_the_first_line_as_the_title_and_the_whole_text_as_the_description() -> None:
    """The `Company | Role | Location` convention is the only title a comment has."""
    result = HackerNewsSource(_hn_client()).fetch()
    prior = next(item for item in result.items if item.source_job_id == "49537584")
    assert prior.title.startswith("Prior Labs | Berlin, Freiburg, NYC")
    assert "\n" not in prior.title
    assert "Research Scientist" in prior.description
    assert prior.url == "https://news.ycombinator.com/item?id=49537584"
    assert prior.company_name is None
    assert isinstance(prior.posted_at, datetime)
    # HTML entities are text again and paragraphs became line breaks.
    assert "&#x2F;" not in prior.description
    assert "<p>" not in prior.description


def test_hn_drops_replies_and_deleted_comments() -> None:
    """A reply is somebody's question and a deleted comment has nothing to read."""
    result = HackerNewsSource(_hn_client()).fetch()
    prior = next(item for item in result.items if item.source_job_id == "49537584")
    assert "children" not in prior.raw
    assert "49599999" not in {item.source_job_id for item in result.items}


def test_hn_replays_from_the_stored_payload() -> None:
    """What was stored rebuilds the same posting without the network."""
    result = HackerNewsSource(_hn_client()).fetch()
    first = result.items[0]
    assert replay_posting("hn", first.raw) == first


def test_hn_with_no_hiring_thread_listed_is_a_failure_not_a_crash() -> None:
    """A month with no thread yet must not be mistaken for a month with no postings."""
    search = {"hits": [{"objectID": "1", "title": "Ask HN: Who wants to be hired? (September 2026)"}]}
    result = HackerNewsSource(_hn_client(search=search)).fetch()
    assert result.status == FetchStatus.FAILED
    assert "Who is hiring" in result.error


def test_hn_search_failure_is_reported() -> None:
    """A 503 from Algolia is a failed fetch with the status in its message."""
    result = HackerNewsSource(_client_returning({"error": "down"}, status=503)).fetch()
    assert result.status == FetchStatus.FAILED
    assert "503" in result.error


def test_remoteok_skips_the_legal_notice_and_names_the_company() -> None:
    """The first list element is a notice (not a job) and every job carries its company."""
    payload = _load("remoteok.json")
    result = RemoteOkSource(_client_returning(payload)).fetch()
    assert result.status == FetchStatus.OK
    assert len(result.items) == len(payload) - 1
    mts = next(item for item in result.items if "Technical Staff" in item.title)
    assert mts.company_name == "Physical Superintelligence"
    assert mts.source == "remoteok"
    assert mts.is_remote is True
    assert mts.url.startswith("https://")
    assert "&lt;" not in mts.description and "<" not in mts.description
    assert isinstance(mts.posted_at, datetime)
    assert replay_posting("remoteok", mts.raw) == mts


def test_remoteok_that_is_not_a_list_is_a_failure() -> None:
    """A dict where a list was expected means the API changed shape."""
    result = RemoteOkSource(_client_returning({"jobs": []})).fetch()
    assert result.status == FetchStatus.FAILED


def test_arbeitnow_reads_the_data_page_and_its_escaped_html() -> None:
    """Jobs sit under `data`. The slug is the id and the description is escaped HTML."""
    payload = _load("arbeitnow.json")
    result = ArbeitnowSource(_client_returning(payload)).fetch()
    assert result.status == FetchStatus.OK
    assert len(result.items) == 3
    scientist = next(item for item in result.items if "Research Scientist" in item.title)
    assert scientist.source_job_id == payload["data"][0]["slug"]
    assert scientist.company_name == "Precisionmedicinegroup"
    assert "&lt;" not in scientist.description and "<div" not in scientist.description
    assert scientist.posted_at.year == 2026
    assert replay_posting("arbeitnow", scientist.raw) == scientist


def test_arbeitnow_follows_the_redirect_from_the_apex_domain() -> None:
    """The plan recorded a 301 from `arbeitnow.com` and the default client must follow it."""
    assert ArbeitnowSource()._client.follow_redirects is True


def test_remotive_reads_jobs_and_a_date_without_an_offset() -> None:
    """`publication_date` carries no offset and must still come out aware."""
    payload = _load("remotive.json")
    result = RemotiveSource(_client_returning(payload)).fetch()
    assert result.status == FetchStatus.OK
    assert len(result.items) == 2
    first = result.items[0]
    assert first.company_name == "iMerit Technology"
    assert first.location_raw == payload["jobs"][0]["candidate_required_location"]
    assert first.posted_at.tzinfo is not None
    assert "<p>" not in first.description
    assert replay_posting("remotive", first.raw) == first


@pytest.mark.parametrize(
    "source", [RemoteOkSource, ArbeitnowSource, RemotiveSource, HackerNewsSource]
)
def test_a_network_error_is_a_failed_result_not_an_exception(source) -> None:
    """Every source turns a connection error into `FetchResult.failed`."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Raises the way httpx does when the host is unreachable."""
        raise httpx2.ConnectError("no route", request=request)

    result = source(httpx2.Client(transport=httpx2.MockTransport(handler))).fetch()
    assert result.status == FetchStatus.FAILED
    assert "ConnectError" in result.error


@pytest.mark.parametrize(
    ("source", "broken"),
    [
        (RemoteOkSource, {"jobs": [{"id": "1", "position": "Dev"}]}),
        (ArbeitnowSource, {"data": [{"title": "Dev"}]}),
        (RemotiveSource, {"jobs": [{"title": "Dev"}]}),
    ],
)
def test_a_payload_of_the_wrong_shape_is_a_failure(source, broken) -> None:
    """A list where a dict was expected, or a job without its id, is an API change and is reported."""
    result = source(_client_returning(broken)).fetch()
    assert result.status == FetchStatus.FAILED
    assert "payload" in result.error.lower()
