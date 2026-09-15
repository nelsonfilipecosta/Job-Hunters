"""The review queue and the two decisions that empty it.

`job-hunters promote --review` lists the companies `discover` has queued.
`--approve` appends one to `config/companies_watchlist.yaml` and `--reject`
silences one. Nothing is ever appended without a person asking for it.

The `companies_watchlist.yaml` file is the source of truth and this module
adds companies to it. The next `ingest` syncs the `companies` table from the
file, which is what makes an approved company's jobs appear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import paths
from .config import CompanyEntry, ConfigError, load_watchlist
from .discover import Watched, reconcile
from .models import CandidateCompany, CandidateStatus, Tier, utcnow
from .normalize import normalize_company


class PromoteError(Exception):
    """A decision that could not be carried out, with the reason in plain words."""


@dataclass(frozen=True)
class Approval:
    """What approving one candidate did."""

    candidate: CandidateCompany
    slug: str
    line: str
    path: Path


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


def review_queue(
    session: Session, watchlist: list[CompanyEntry] | None = None, now: datetime | None = None
) -> list[CandidateCompany]:
    """Every pending candidate. The ones with a board first and the most seen before the rest.

    Reconciles against the watchlist file first, so a company added by hand or on another
    machine drops out of the queue instead of being approved twice.
    """
    entries = watchlist if watchlist is not None else load_watchlist()
    reconcile(session, Watched.build(entries), now or utcnow())
    pending = session.scalars(
        select(CandidateCompany).where(CandidateCompany.status == CandidateStatus.PENDING)
    ).all()
    return sorted(
        pending,
        key=lambda c: (c.ats_type is None, -c.sightings, -(c.board_jobs or 0), c.name.lower()),
    )


def find_candidate(session: Session, ref: str) -> CandidateCompany:
    """A candidate by id or by name, or a PromoteError naming what was not found."""
    candidate = None
    if ref.strip().isdigit():
        candidate = session.get(CandidateCompany, int(ref))
    if candidate is None:
        candidate = session.scalar(
            select(CandidateCompany).where(CandidateCompany.name_key == normalize_company(ref))
        )
    if candidate is None:
        raise PromoteError(f"No candidate {ref!r}. `job-hunters promote --review` lists them by id.")
    return candidate


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------


def slug_for(name: str) -> str:
    """A watchlist slug derived from a company name: lowercase, hyphens and nothing else."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise PromoteError(f"Cannot derive a slug from {name!r}. Pass one with --slug.")
    return slug


def watchlist_line(slug: str, name: str, ats: str, token: str, tier: str | None) -> str:
    """One watchlist entry in the flow style the file uses and quoted only where YAML needs it.

    Without a tier the entry loads with the default one. `probe` leaves it out because a company
    you looked up yourself is not one the loop discovered.
    """
    tail = f", tier: {tier}" if tier else ""
    return f"- {{ slug: {slug}, name: {_scalar(name)}, ats: {ats}, token: {token}{tail} }}"


# Bare words YAML would read as something other than a string.
_YAML_KEYWORDS = frozenset({"true", "false", "null", "yes", "no", "on", "off", "y", "n"})


def _scalar(value: str) -> str:
    """A YAML flow scalar. Bare when it can be and double-quoted when YAML would read it as anything else."""
    if (
        re.fullmatch(r"[A-Za-z][A-Za-z0-9 .&'()/+-]*", value)
        and not value.endswith(" ")
        and value.lower() not in _YAML_KEYWORDS
    ):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def approve(
    session: Session,
    ref: str,
    *,
    slug: str | None = None,
    name: str | None = None,
    tier: str = Tier.DISCOVERED,
    watchlist_path: Path | None = None,
    now: datetime | None = None,
) -> Approval:
    """Appends one candidate to the watchlist and marks it approved.

    Everything is checked before anything is written: the candidate has a board,
    the slug is free and the line the file would gain parses as a valid entry.
    The file is written first and the row updated after, so a file that cannot
    be written leaves the candidate pending and the queue still showing it.
    """
    now = now or utcnow()
    path = watchlist_path or paths.WATCHLIST_PATH
    candidate = find_candidate(session, ref)
    if candidate.status == CandidateStatus.APPROVED:
        raise PromoteError(
            f"{candidate.name} is already in the watchlist as {candidate.slug or 'an entry'}."
        )
    if not candidate.ats_type or not candidate.ats_token:
        raise PromoteError(
            f"No Greenhouse, Lever or Ashby board was found for {candidate.name}, so there is "
            f"nothing to watch. If you know its board, add the line by hand "
            f"(`job-hunters probe {candidate.name!r}` probes the slug patterns)."
        )
    display = (name or candidate.name).strip()
    chosen = slug or slug_for(display)
    line = watchlist_line(chosen, display, candidate.ats_type, candidate.ats_token, tier)

    # Validate the entry alone first and then against the file it is about to join.
    try:
        CompanyEntry.model_validate(yaml.safe_load(line)[0])
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise PromoteError(f"The entry {line!r} is not a valid watchlist line: {exc}") from exc
    entries = load_watchlist(path)
    if any(entry.slug == chosen for entry in entries):
        raise PromoteError(
            f"The slug {chosen!r} is already in {path.name}. Pass another with --slug."
        )
    if any(
        entry.ats.value == candidate.ats_type
        and str(entry.ats_config.get("token", "")).lower() == candidate.ats_token
        for entry in entries
    ):
        raise PromoteError(
            f"{path.name} already watches the {candidate.ats_type} board "
            f"{candidate.ats_token!r}. Nothing was added."
        )

    append_line(path, line)
    try:
        load_watchlist(path)
    except ConfigError as exc:  # pragma: no cover - the line was validated above
        raise PromoteError(f"{path.name} no longer loads after the append: {exc}") from exc

    candidate.status = CandidateStatus.APPROVED
    candidate.decided_at = now
    candidate.slug = chosen
    session.flush()
    return Approval(candidate=candidate, slug=chosen, line=line, path=path)


DISCOVERED_HEADER = "# --- Discovered by the promotion loop ------------------------------------"


def append_line(path: Path, line: str) -> None:
    """Adds one line to the end of the file touching nothing above it."""
    try:
        existing = path.read_text(encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            if DISCOVERED_HEADER not in existing:
                handle.write(f"\n{DISCOVERED_HEADER}\n\n")
            handle.write(line + "\n")
    except OSError as exc:
        raise PromoteError(
            f"Cannot write to {path}: {exc.strerror or exc}. Inside the container "
            f"`config/` is mounted read-only, so approve on the host or add this "
            f"line to {path.name} by hand:\n  {line}"
        ) from exc


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------


def reject(session: Session, ref: str, now: datetime | None = None) -> CandidateCompany:
    """Marks one candidate rejected so the queue stops showing it. Its sightings keep counting."""
    candidate = find_candidate(session, ref)
    if candidate.status == CandidateStatus.APPROVED:
        raise PromoteError(
            f"{candidate.name} is in the watchlist. Remove its line from "
            f"`companies_watchlist.yaml` to stop watching it."
        )
    candidate.status = CandidateStatus.REJECTED
    candidate.decided_at = now or utcnow()
    session.flush()
    return candidate
