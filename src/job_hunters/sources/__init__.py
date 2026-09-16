"""The job-board adapters and the registries that map a source name to one.

    `get_adapter("greenhouse")`     how ingest finds the right class for a company
    `get_discover_source("hn")`     how `discover` finds the right class for a site
    `replay_posting`                how scoring rebuilds a posting's text from its
                                    stored payload, without fetching anything.

Two registries because the two kinds of source are asked different questions.
An ATS adapter is given a company and fetches its board. A `discover` source is
given nothing and fetches whatever the site lists, naming the company itself.

To add an ATS or a `discover` source: write the module and add one line here.
"""

from __future__ import annotations

from .arbeitnow import ArbeitnowSource
from .ashby import AshbyAdapter
from .base import DiscoverSource, FetchResult, JobSource, RawPosting, SourceError
from .greenhouse import GreenhouseAdapter
from .hn import HackerNewsSource
from .lever import LeverAdapter
from .remoteok import RemoteOkSource
from .remotive import RemotiveSource

ADAPTERS: dict[str, type[JobSource]] = {
    "greenhouse": GreenhouseAdapter,
    "lever": LeverAdapter,
    "ashby": AshbyAdapter,
}

DISCOVER_SOURCES: dict[str, type[DiscoverSource]] = {
    "hn": HackerNewsSource,
    "remoteok": RemoteOkSource,
    "arbeitnow": ArbeitnowSource,
    "remotive": RemotiveSource,
}


def get_adapter(ats_type: str) -> JobSource:
    """Instantiates the adapter with the given ATS type or raises a readable KeyError."""
    try:
        return ADAPTERS[ats_type]()
    except KeyError:
        known = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"No adapter for ats_type {ats_type!r} (known: {known})") from None


def get_discover_source(name: str) -> DiscoverSource:
    """Instantiates the `discover` source with this name or raises a readable KeyError."""
    try:
        return DISCOVER_SOURCES[name]()
    except KeyError:
        known = ", ".join(sorted(DISCOVER_SOURCES))
        raise KeyError(f"No discover source named {name!r} (known: {known})") from None


def replay_posting(source: str, raw: dict) -> RawPosting:
    """Rebuilds a posting from the payload stored for it in `job_sources.raw_json`.

    Every adapter's `_to_posting` is a pure function of the dict the board
    returned, so the text the judge reads can be recomputed from the database
    without touching the network. That is what lets a normalization fix be
    replayed and what scoring uses to read each posting's own text rather
    than the one its job happens to display. Sightings from `discover` replay the
    same way through their own source's `_to_posting`.
    """
    adapter = ADAPTERS.get(source) or DISCOVER_SOURCES.get(source)
    if adapter is None:
        known = ", ".join(sorted([*ADAPTERS, *DISCOVER_SOURCES]))
        raise KeyError(f"No adapter for source {source!r} (known: {known})")
    return adapter._to_posting(raw)


__all__ = [
    "ADAPTERS",
    "DISCOVER_SOURCES",
    "ArbeitnowSource",
    "AshbyAdapter",
    "DiscoverSource",
    "FetchResult",
    "GreenhouseAdapter",
    "HackerNewsSource",
    "JobSource",
    "LeverAdapter",
    "RawPosting",
    "RemoteOkSource",
    "RemotiveSource",
    "SourceError",
    "get_adapter",
    "get_discover_source",
    "replay_posting",
]
