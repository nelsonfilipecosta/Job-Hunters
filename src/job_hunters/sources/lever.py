"""Lever job boards.

    GET https://api.lever.co/v0/postings/{token}?mode=json

Returns a bare JSON list. Each job has a UUID `id`, the title under `text`,
`hostedUrl` and a `country` field holding an ISO 3166-1 alpha-2 code and a
`workplaceType`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx2

from ..models import Company, SourceKind
from ..normalize import html_to_text
from .base import FetchResult, RawPosting, SourceError, default_client, get_json

BOARD_URL = "https://api.lever.co/v0/postings/{token}?mode=json"


class LeverAdapter:
    """Reads a Lever job board through its public JSON endpoint.

    Lever is the only board that supplies a country code outright and it splits
    a description across three fields that have to be joined back together.
    """

    source = SourceKind.LEVER

    def __init__(self, client: httpx2.Client | None = None) -> None:
        """Takes an HTTP client or builds the shared default one."""
        self._client = client or default_client()

    def fetch(self, company: Company) -> FetchResult:
        """Fetches every open posting on the company's board. Never raises."""
        token = company.ats_config.get("token")
        if not token:
            return FetchResult.failed(f"{company.slug}: no Lever token in `ats_config`")
        try:
            payload = get_json(self._client, BOARD_URL.format(token=token))
            if not isinstance(payload, list):
                raise TypeError(f"Expected a list, got {type(payload).__name__}")
            items = [self._to_posting(posting) for posting in payload]
        except SourceError as exc:
            return FetchResult.failed(str(exc))
        except (KeyError, TypeError, AttributeError) as exc:
            return FetchResult.failed(f"Unexpected Lever payload: {exc!r}")
        return FetchResult.ok(items)

    @staticmethod
    def _to_posting(posting: dict) -> RawPosting:
        """Converts one Greenhouse job into the shape every adapter returns."""
        categories = posting.get("categories") or {}
        all_locations = tuple(categories.get("allLocations") or ())
        location = categories.get("location")

        parts = [posting.get("descriptionPlain") or ""]
        for section in posting.get("lists") or []:
            heading = section.get("text") or ""
            body = html_to_text(section.get("content") or "")
            parts.append(f"{heading}\n{body}".strip())
        parts.append(posting.get("additionalPlain") or "")
        description = "\n\n".join(part for part in parts if part).strip()

        created = posting.get("createdAt")
        posted_at = (
            datetime.fromtimestamp(created / 1000, tz=UTC)
            if isinstance(created, (int, float))
            else None
        )
        return RawPosting(
            source=SourceKind.LEVER,
            source_job_id=str(posting["id"]),
            title=posting["text"],
            url=posting.get("hostedUrl"),
            location_raw=location,
            description=description,
            raw=posting,
            country_code=posting.get("country"),
            workplace_type=posting.get("workplaceType"),
            location_hints=tuple(loc for loc in all_locations if loc != location),
            posted_at=posted_at,
        )
