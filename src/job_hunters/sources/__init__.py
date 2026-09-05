"""The job-board adapters and the registry that maps an ATS type to one.

`get_adapter("greenhouse")` is how ingest finds the right class for a company.
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
]
