"""Finds which ATS, under which slug, hosts a company's job board.

The slug is rarely the company name. Rather than guess, `probe` tries a handful
of spellings against all three ATS URL patterns and reports every board that answers.
Its output is a ready-to-paste watchlist line.

`job-hunters probe <name>` prints every board found and leaves the choice to
you. The discovery loop has nobody to ask, so `best_board` picks the board with
the most postings or the one a posting's own careers link named when that link
pointed straight at a board.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import httpx2

from .sources.ashby import BOARD_URL as ASHBY_URL
from .sources.base import SourceError, default_client, get_json
from .sources.greenhouse import BOARD_URL as GREENHOUSE_URL
from .sources.lever import BOARD_URL as LEVER_URL


@dataclass(frozen=True)
class Board:
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


# A careers link that points straight at a board names its ATS and token. The
# token is whatever follows the host, up to the next slash or query string.
_BOARD_LINKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("greenhouse", re.compile(r"(?:boards|job-boards)\.greenhouse\.io/(?:embed/job_board\?for=)?([A-Za-z0-9-]+)")),
    ("lever", re.compile(r"jobs\.lever\.co/([A-Za-z0-9-]+)")),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9-]+)")),
)


def token_from_url(url: str | None) -> tuple[str, str] | None:
    """The (ATS, token) a careers link names or None when it is not a board link."""
    if not url:
        return None
    for ats, pattern in _BOARD_LINKS:
        match = pattern.search(url)
        if match:
            return ats, match.group(1).lower()
    return None


def _probe_one(client: httpx2.Client, ats: str, url: str, token: str) -> Board | None:
    """Checks one URL for one board. None unless the answer has the right shape."""
    try:
        payload = get_json(client, url)
    except SourceError:
        return None
    if ats == "lever":
        return Board(ats, token, len(payload), url) if isinstance(payload, list) else None
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if jobs is None:
        return None
    return Board(ats, token, len(jobs), url)


def probe(
    name: str, client: httpx2.Client | None = None, *, hints: Iterable[str] = ()
) -> list[Board]:
    """Tries every slug variant against every ATS and returns each board found.

    `hints` are tokens to try before the variants, for when something (a
    careers link) already said what the slug is.
    """
    client = client or default_client(timeout=15)
    found: list[Board] = []
    tokens = [*hints, *slug_variants(name)]
    seen: set[str] = set()
    for token in [t for t in tokens if t and not (t in seen or seen.add(t))]:
        for ats, pattern in (
            ("greenhouse", GREENHOUSE_URL.replace("?content=true", "")),
            ("lever", LEVER_URL),
            ("ashby", ASHBY_URL),
        ):
            hit = _probe_one(client, ats, pattern.format(token=token), token)
            if hit is not None:
                found.append(hit)
    return found


def best_board(hits: Iterable[Board], hint: tuple[str, str] | None = None) -> Board | None:
    """The board to watch out of everything a probe found or None when none has postings."""
    with_postings = [hit for hit in hits if hit.job_count > 0]
    if not with_postings:
        return None
    if hint is not None:
        for hit in with_postings:
            if (hit.ats, hit.token) == hint:
                return hit
    return max(with_postings, key=lambda hit: hit.job_count)
