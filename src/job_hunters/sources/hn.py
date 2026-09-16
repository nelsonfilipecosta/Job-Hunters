"""The Hacker News "Who is hiring?" thread through the Algolia API.

    GET https://hn.algolia.com/api/v1/search_by_date?query=...&tags=story,author_whoishiring
    GET https://hn.algolia.com/api/v1/items/{thread_id}

Two calls. The first finds the current month's thread, which is posted by the
`whoishiring` account alongside two sibling threads ("Who wants to be hired?"
and "Freelancer? Seeking freelancer?") that are skipped by title. The second
returns the thread with every comment nested under `children`. Only top-level
comments are postings.

A comment is free-form prose, but by convention its first line reads
`Company | Role(s) | Location | ...`. That line is taken as the posting's title
so the prefilter can read it. However, the company is not parsed here. The
convention is not reliable enough, so `company_name` stays empty and `extract.py`
pulls it out of the text with a model call.
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

SEARCH_URL = (
    "https://hn.algolia.com/api/v1/search_by_date"
    "?query=%22who%20is%20hiring%22&tags=story,author_whoishiring&hitsPerPage=10"
)
ITEM_URL = "https://hn.algolia.com/api/v1/items/{id}"
COMMENT_URL = "https://news.ycombinator.com/item?id={id}"
THREAD_TITLE = "Ask HN: Who is hiring?"

# The first line of a comment is its title. Longer than this and it is a paragraph.
MAX_TITLE_CHARS = 300


class HackerNewsSource:
    """Reads the current "Who is hiring?" thread through the Algolia API."""

    source = SourceKind.HN

    def __init__(self, client: httpx2.Client | None = None) -> None:
        """Takes an HTTP client or builds the shared default one."""
        self._client = client or default_client()

    def fetch(self) -> FetchResult:
        """Fetches every top-level comment on this month's thread. Never raises."""
        try:
            thread_id = self._current_thread()
            payload = get_json(self._client, ITEM_URL.format(id=thread_id))
            comments = payload["children"]
            # A deleted or flagged comment comes back with no text at all.
            items = [self._to_posting(c) for c in comments if c.get("text")]
        except SourceError as exc:
            return FetchResult.failed(str(exc))
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            return FetchResult.failed(f"Unexpected Hacker News payload: {exc!r}")
        return FetchResult.ok(items)

    def _current_thread(self) -> int:
        """The id of the newest "Who is hiring?" thread or a SourceError when none is listed."""
        payload = get_json(self._client, SEARCH_URL)
        for hit in payload["hits"]:
            if (hit.get("title") or "").startswith(THREAD_TITLE):
                return int(hit["objectID"])
        raise SourceError(f"No {THREAD_TITLE!r} thread in the Algolia search results")

    @staticmethod
    def _to_posting(comment: dict) -> RawPosting:
        """Converts one top-level comment into the shape every adapter returns.

        Replies are dropped from the stored payload. They are other people's questions
        and a thread's replies can outweigh its postings.
        """
        text = html_to_text(comment["text"])
        title = text.split("\n", 1)[0].strip()[:MAX_TITLE_CHARS]
        raw = {key: value for key, value in comment.items() if key != "children"}
        return RawPosting(
            source=SourceKind.HN,
            source_job_id=str(comment["id"]),
            title=title,
            url=COMMENT_URL.format(id=comment["id"]),
            location_raw=None,
            description=text,
            raw=raw,
            posted_at=parse_iso_datetime(comment.get("created_at")),
        )
