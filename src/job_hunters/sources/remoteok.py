"""RemoteOK's job board API.

    GET https://remoteok.com/api

Returns a bare JSON list whose first element is a legal notice rather than a
job. Each job has a numeric `id`, the title under `position`, a `company`
name, `tags`, a `location` and a `description` that is HTML which has itself
been HTML-escaped (the same as Greenhouse). Every listing is remote.
"""

from __future__ import annotations

import html

import httpx2

from ..tables import SourceKind
from ..normalize import html_to_text
from .base import (
    FetchResult,
    RawPosting,
    SourceError,
    default_client,
    get_json,
    parse_iso_datetime,
)

API_URL = "https://remoteok.com/api"


class RemoteOkSource:
    """Reads RemoteOK's listing through its public JSON endpoint."""

    source = SourceKind.REMOTEOK

    def __init__(self, client: httpx2.Client | None = None) -> None:
        """Takes an HTTP client or builds the shared default one."""
        self._client = client or default_client()

    def fetch(self) -> FetchResult:
        """Fetches every job currently listed. Never raises."""
        try:
            payload = get_json(self._client, API_URL)
            if not isinstance(payload, list):
                raise TypeError(f"Expected a list, got {type(payload).__name__}")
            # The legal notice at the head of the list has no id and no position.
            jobs = [j for j in payload if isinstance(j, dict) and j.get("id") and j.get("position")]
            items = [self._to_posting(job) for job in jobs]
        except SourceError as exc:
            return FetchResult.failed(str(exc))
        except (KeyError, TypeError, AttributeError) as exc:
            return FetchResult.failed(f"Unexpected RemoteOK payload: {exc!r}")
        return FetchResult.ok(items)

    @staticmethod
    def _to_posting(job: dict) -> RawPosting:
        """Converts one RemoteOK job into the shape every adapter returns."""
        return RawPosting(
            source=SourceKind.REMOTEOK,
            source_job_id=str(job["id"]),
            title=job["position"],
            url=job.get("url") or job.get("apply_url"),
            location_raw=job.get("location") or None,
            description=html_to_text(html.unescape(job.get("description") or "")),
            raw=job,
            is_remote=True,
            posted_at=parse_iso_datetime(job.get("date")),
            company_name=(job.get("company") or "").strip() or None,
        )
