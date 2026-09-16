"""Remotive's job board API.

    GET https://remotive.com/api/remote-jobs

Returns `{"jobs": [...], "job-count": N, ...}` with two notice fields ahead of
the jobs. Each job has a numeric `id`, a `title`, a `company_name`, a
`category`, `candidate_required_location`, a `publication_date` without an
offset and a `description` in plain HTML. Every listing is remote.
"""

from __future__ import annotations

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

API_URL = "https://remotive.com/api/remote-jobs"


class RemotiveSource:
    """Reads Remotive's listing through its public JSON endpoint."""

    source = SourceKind.REMOTIVE

    def __init__(self, client: httpx2.Client | None = None) -> None:
        """Takes an HTTP client or builds the shared default one."""
        self._client = client or default_client()

    def fetch(self) -> FetchResult:
        """Fetches every job currently listed. Never raises."""
        try:
            payload = get_json(self._client, API_URL)
            jobs = payload["jobs"]
            items = [self._to_posting(job) for job in jobs]
        except SourceError as exc:
            return FetchResult.failed(str(exc))
        except (KeyError, TypeError, AttributeError) as exc:
            return FetchResult.failed(f"Unexpected Remotive payload: {exc!r}")
        return FetchResult.ok(items)

    @staticmethod
    def _to_posting(job: dict) -> RawPosting:
        """Converts one Remotive job into the shape every adapter returns."""
        return RawPosting(
            source=SourceKind.REMOTIVE,
            source_job_id=str(job["id"]),
            title=job["title"],
            url=job.get("url"),
            location_raw=job.get("candidate_required_location") or None,
            description=html_to_text(job.get("description") or ""),
            raw=job,
            is_remote=True,
            posted_at=parse_iso_datetime(job.get("publication_date")),
            company_name=(job.get("company_name") or "").strip() or None,
        )
