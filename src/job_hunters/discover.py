"""Finds which ATS, under which slug, hosts a company's job board.

The slug is rarely the company name. Rather than guess, `probe` tries a handful
of spellings against all three ATS URL patterns and reports every board that answers.
Its output is a ready-to-paste watchlist line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx2

from .sources.ashby import BOARD_URL as ASHBY_URL
from .sources.base import SourceError, default_client, get_json
from .sources.greenhouse import BOARD_URL as GREENHOUSE_URL
from .sources.lever import BOARD_URL as LEVER_URL


@dataclass(frozen=True)
class Discovery:
    """One board that answered a probe: which ATS, under which token and with how many jobs."""

    ats: str
    token: str
    job_count: int
    url: str

    def watchlist_line(self, slug: str, name: str) -> str:
        """A ready-to-paste `watchlist.yaml` entry for this board."""
        return f"- {{ slug: {slug}, name: {name}, ats: {self.ats}, token: {self.token}, tier: discovered }}"


def slug_variants(name: str) -> list[str]:
    """Plausible board slugs for a company name, most likely first, no duplicates."""
    base = name.strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", base)
    hyphenated = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    stems: list[str] = []
    for candidate in (compact, hyphenated):
        for suffix in ("-ai", "ai", "-labs", "labs", "-inc", "inc"):
            if candidate.endswith(suffix) and len(candidate) > len(suffix) + 1:
                stems.append(candidate[: -len(suffix)].rstrip("-"))
    variants: list[str] = []
    for v in [compact, hyphenated, *stems]:
        if v:
            variants += [v, f"{v}ai", f"{v}-ai"]
    seen: set[str] = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


def _probe_one(client: httpx2.Client, ats: str, url: str, token: str) -> Discovery | None:
    """Checks one URL for one board. None unless the answer has the right shape."""
    try:
        payload = get_json(client, url)
    except SourceError:
        return None
    if ats == "lever":
        return Discovery(ats, token, len(payload), url) if isinstance(payload, list) else None
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if jobs is None:
        return None
    return Discovery(ats, token, len(jobs), url)


def probe(name: str, client: httpx2.Client | None = None) -> list[Discovery]:
    """Tries every slug variant against every ATS and returns each board found."""
    client = client or default_client(timeout=15)
    found: list[Discovery] = []
    for token in slug_variants(name):
        for ats, pattern in (
            ("greenhouse", GREENHOUSE_URL.replace("?content=true", "")),
            ("lever", LEVER_URL),
            ("ashby", ASHBY_URL),
        ):
            hit = _probe_one(client, ats, pattern.format(token=token), token)
            if hit is not None:
                found.append(hit)
    return found
