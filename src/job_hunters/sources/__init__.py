"""The job-board adapters and the registry that maps an ATS type to one.

    `get_adapter("greenhouse")` how ingest finds the right class for a company
    `replay_posting`            how scoring rebuilds a posting's text from its stored
                                payload, without fetching anything.

To add an ATS: write the module and add one line to `ADAPTERS`.
"""

from __future__ import annotations

from .ashby import AshbyAdapter
from .base import FetchResult, JobSource, RawPosting, SourceError
from .greenhouse import GreenhouseAdapter
from .lever import LeverAdapter

ADAPTERS: dict[str, type[JobSource]] = {
    "greenhouse": GreenhouseAdapter,
    "lever": LeverAdapter,
    "ashby": AshbyAdapter,
}


def get_adapter(ats_type: str) -> JobSource:
    """Instantiates the adapter for an ATS type or raises a readable KeyError."""
    try:
        return ADAPTERS[ats_type]()
    except KeyError:
        known = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"No adapter for ats_type {ats_type!r} (known: {known})") from None


def replay_posting(source: str, raw: dict) -> RawPosting:
    """Rebuilds a posting from the payload stored for it in `job_sources.raw_json`.

    Every adapter's `_to_posting` is a pure function of the dict the board
    returned, so the text the judge reads can be recomputed from the database
    without touching the network. That is what lets a normalization fix be
    replayed and what scoring uses to read each posting's own text rather
    than the one its job happens to display.
    """
    try:
        adapter = ADAPTERS[source]
    except KeyError:
        known = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"No adapter for source {source!r} (known: {known})") from None
    return adapter._to_posting(raw)


__all__ = [
    "ADAPTERS",
    "AshbyAdapter",
    "FetchResult",
    "GreenhouseAdapter",
    "JobSource",
    "LeverAdapter",
    "RawPosting",
    "SourceError",
    "get_adapter",
    "replay_posting",
]
