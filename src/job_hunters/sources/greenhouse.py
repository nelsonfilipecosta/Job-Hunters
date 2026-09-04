"""Greenhouse job boards.

    GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true

Returns `{"jobs": [...], "meta": {...}}`. Each job has an integer `id`, `title`,
`absolute_url`, a `location.name` string and `content`. `content` is a HTML that
has itself been HTML-escaped.
"""

from __future__ import annotations

import html

import httpx2

from ..models import Company, SourceKind
from ..normalize import html_to_text
from .base import (
    FetchResult,
    RawPosting,
    SourceError,
    default_client,
    get_json,
    parse_iso_datetime,
)

BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"


class GreenhouseAdapter:
    """Reads a Greenhouse job board through its public JSON endpoint.

    Greenhouse returns descriptions as HTML-escaped HTML, so the text is
    unescaped once and then flattened before it is stored.
    """

    source = SourceKind.GREENHOUSE

    def __init__(self, client: httpx2.Client | None = None) -> None:
        """Takes an HTTP client or builds the shared default one."""
        self._client = client or default_client()

    def fetch(self, company: Company) -> FetchResult:
        """Fetches every open posting on the company's board. Never raises."""
        token = company.ats_config.get("token")
        if not token:
            return FetchResult.failed(f"{company.slug}: no Greenhouse token in `ats_config`")
        try:
            payload = get_json(self._client, BOARD_URL.format(token=token))
            jobs = payload["jobs"]
            items = [self._to_posting(job) for job in jobs]
        except SourceError as exc:
            return FetchResult.failed(str(exc))
        except (KeyError, TypeError, AttributeError) as exc:
            return FetchResult.failed(f"Unexpected Greenhouse payload: {exc!r}")
        return FetchResult.ok(items)

    @staticmethod
    def _to_posting(job: dict) -> RawPosting:
        """Converts one Greenhouse job into the shape every adapter returns."""
        location = (job.get("location") or {}).get("name")
        # Offices are a second, sometimes cleaner, list of places.
        offices = tuple(o["name"] for o in job.get("offices") or [] if o.get("name"))
        return RawPosting(
            source=SourceKind.GREENHOUSE,
            source_job_id=str(job["id"]),
            title=job["title"],
            url=job.get("absolute_url"),
            location_raw=location,
            description=html_to_text(html.unescape(job.get("content") or "")),
            raw=job,
            location_hints=offices,
            posted_at=parse_iso_datetime(job.get("first_published") or job.get("updated_at")),
        )
