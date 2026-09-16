"""Arbeitnow's job board API.

    GET https://www.arbeitnow.com/api/job-board-api

Returns `{"data": [...], "links": {...}, "meta": {...}}`, newest first and
paginated with `?page=`. Only the first page is read: the board is updated
hourly and a weekly read of its newest page is all `discover` needs. Each
job has a `slug` for an id, a `title`, a `company_name`, a `location`, a
`remote` flag and a `description` that is HTML-escaped HTML.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

import httpx2

from ..tables import SourceKind
from ..normalize import html_to_text
from .base import FetchResult, RawPosting, SourceError, default_client, get_json

API_URL = "https://www.arbeitnow.com/api/job-board-api"


class ArbeitnowSource:
    """Reads the newest page of Arbeitnow's board through its public JSON endpoint."""

    source = SourceKind.ARBEITNOW

    def __init__(self, client: httpx2.Client | None = None) -> None:
        """Takes an HTTP client or builds one that follows the board's redirect."""
        self._client = client or default_client(follow_redirects=True)

    def fetch(self) -> FetchResult:
        """Fetches the newest page of jobs. Never raises."""
        try:
            payload = get_json(self._client, API_URL)
            jobs = payload["data"]
            items = [self._to_posting(job) for job in jobs]
        except SourceError as exc:
            return FetchResult.failed(str(exc))
        except (KeyError, TypeError, AttributeError) as exc:
            return FetchResult.failed(f"Unexpected Arbeitnow payload: {exc!r}")
        return FetchResult.ok(items)

    @staticmethod
    def _to_posting(job: dict) -> RawPosting:
        """Converts one Arbeitnow job into the shape every adapter returns."""
        created = job.get("created_at")
        posted_at = (
            datetime.fromtimestamp(created, tz=UTC) if isinstance(created, (int, float)) else None
        )
        return RawPosting(
            source=SourceKind.ARBEITNOW,
            source_job_id=str(job["slug"]),
            title=job["title"],
            url=job.get("url"),
            location_raw=job.get("location") or None,
            description=html_to_text(html.unescape(job.get("description") or "")),
            raw=job,
            is_remote=job.get("remote"),
            posted_at=posted_at,
            company_name=(job.get("company_name") or "").strip() or None,
        )
