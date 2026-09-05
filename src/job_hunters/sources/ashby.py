"""Ashby job boards.

    GET https://api.ashbyhq.com/posting-api/job-board/{token}

Returns `{"jobs": [...], "apiVersion": ...}`. Each job has a UUID `id`,
`title`, `jobUrl`, `descriptionPlain` already as text, a `location` string,
`secondaryLocations` and two work-mode fields.

Those two work-mode fields often disagree. On a live survey of ~1,200 Ashby
postings, `isRemote` was True for every Hybrid role as well as every Remote
one. `workplaceType` (`Remote` / `Hybrid` / `OnSite` or absent) is the one
that actually distinguishes them, so it is passed as the primary hint and
`isRemote` only breaks ties when `workplaceType` is missing.
"""

from __future__ import annotations

import httpx2

from ..models import Company, SourceKind
from .base import (
    FetchResult,
    RawPosting,
    SourceError,
    default_client,
    get_json,
    parse_iso_datetime,
)

BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{token}"


class AshbyAdapter:
    """Reads an Ashby job board through its public posting API.

    Ashby is the only board that lists unpublished postings alongside live ones
    and the only one that states a workplace type, which is what the location
    parser trusts over the board's `isRemote` flag.
    """

    source = SourceKind.ASHBY

    def __init__(self, client: httpx2.Client | None = None) -> None:
        """Takes an HTTP client or builds the shared default one."""
        self._client = client or default_client()

    def fetch(self, company: Company) -> FetchResult:
        """Fetches every listed posting on the company's board. Never raises."""
        token = company.ats_config.get("token")
        if not token:
            return FetchResult.failed(f"{company.slug}: no Ashby token in `ats_config`")
        try:
            payload = get_json(self._client, BOARD_URL.format(token=token))
            jobs = payload["jobs"]
            # `isListed: false` means postings exist on the board but are not shown
            # publicly. They are not open roles from an applicant's point of view.
            items = [self._to_posting(job) for job in jobs if job.get("isListed", True)]
        except SourceError as exc:
            return FetchResult.failed(str(exc))
        except (KeyError, TypeError, AttributeError) as exc:
            return FetchResult.failed(f"Unexpected Ashby payload: {exc!r}")
        return FetchResult.ok(items)

    @staticmethod
    def _to_posting(job: dict) -> RawPosting:
        """Converts one Greenhouse job into the shape every adapter returns."""
        hints: list[str] = []
        for secondary in job.get("secondaryLocations") or []:
            if isinstance(secondary, dict) and secondary.get("location"):
                hints.append(secondary["location"])
        # Ashby sometimes puts the country only in the structured address.
        country = ((job.get("address") or {}).get("postalAddress") or {}).get("addressCountry")
        if country:
            hints.append(country)
        return RawPosting(
            source=SourceKind.ASHBY,
            source_job_id=str(job["id"]),
            title=job["title"],
            url=job.get("jobUrl") or job.get("applyUrl"),
            location_raw=job.get("location"),
            description=(job.get("descriptionPlain") or "").strip(),
            raw=job,
            workplace_type=job.get("workplaceType"),
            is_remote=job.get("isRemote"),
            location_hints=tuple(hints),
            posted_at=parse_iso_datetime(job.get("publishedAt")),
        )
