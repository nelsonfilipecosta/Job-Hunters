"""What every job-board adapter produces and the one method it must implement.

An adapter turns one company's board into a list of `RawPostings`. It is the
only place that knows an ATS's URL scheme or JSON shape. Everything downstream
(normalize, dedup, ingest) works on `RawPosting` alone, so adding Workday later
means adding one module here and nothing anywhere else.

The contract that matters most is in `FetchResult`. A network error, a 404 from
a renamed board or a malformed JSON come back as `FetchResult.failed(...)`,
because the ingest loop must be able to tell "this fetch broke" apart from "this
board is empty".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx2

from ..models import Company, FetchStatus

# Sent with every request. Public boards are free to use, but identifying
# yourself is the polite default and makes rate-limit conversations possible.
USER_AGENT = "job-hunters/0.1 (+https://github.com/nelsonfilipecosta/Job-Hunters)"

DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class RawPosting:
    """One posting exactly as a board described it before any normalization.

    `description` is already plain text. Turning HTML into text is the adapter's
    job because only it knows whether the board sends HTML, escaped HTML (Greenhouse)
    or plain text (Ashby). Everything else is passed through as the board gave it.
    """

    source: str
    source_job_id: str
    title: str
    url: str | None
    location_raw: str | None
    description: str
    raw: dict[str, Any] = field(repr=False)
    country_code: str | None = None
    workplace_type: str | None = None
    is_remote: bool | None = None
    location_hints: tuple[str, ...] = ()
    posted_at: datetime | None = None


@dataclass(frozen=True)
class FetchResult:
    """The outcome of one fetch: ok with items or failed with a reason."""

    status: str
    items: tuple[RawPosting, ...] = ()
    error: str | None = None

    @classmethod
    def ok(cls, items: list[RawPosting]) -> FetchResult:
        """A board that answered, carrying the postings it listed."""
        return cls(status=FetchStatus.OK, items=tuple(items))

    @classmethod
    def failed(cls, error: str) -> FetchResult:
        """A board that could not be read, carrying why."""
        return cls(status=FetchStatus.FAILED, error=error)

    @property
    def succeeded(self) -> bool:
        """True when the board answered, whether or not it listed any postings."""
        return self.status == FetchStatus.OK


class JobSource(Protocol):
    """The interface an adapter implements. One board type and one class."""

    # A `SourceKind` value stored on every JobSource row this adapter creates.
    source: str

    def fetch(self, company: Company) -> FetchResult:
        """Fetches every listed posting on the company's board. Never raises."""
        ...


class SourceError(Exception):
    """A fetch that could not complete. Adapters convert this into `FetchResult.failed`."""


def default_client(timeout: float = DEFAULT_TIMEOUT) -> httpx2.Client:
    """A client with the project's User-Agent. Adapters accept a replacement so tests
    can inject `httpx2.MockTransport` and never touch the network."""
    return httpx2.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})


def get_json(client: httpx2.Client, url: str) -> Any:
    """GET a URL and return its JSON, raising SourceError on any failure.

    Every way this can go wrong is turned into one exception type with a readable
    message, so each adapter's `fetch` is a single try/except rather than four.
    """
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx2.HTTPStatusError as exc:
        raise SourceError(f"HTTP {exc.response.status_code} for {url}") from exc
    except httpx2.RequestError as exc:
        raise SourceError(f"{type(exc).__name__} for {url}: {exc}") from exc
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise SourceError(f"Non-JSON response from {url}") from exc


def parse_iso_datetime(value: Any) -> datetime | None:
    """Parses the ISO-8601 timestamps boards use or None when absent or unparseable.

    Always returns an aware datetime. A board that omits the offset gets UTC,
    because everything downstream stores and compares UTC and one naive value
    reaching a comparison raises `TypeError` and fails that company's ingest.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
