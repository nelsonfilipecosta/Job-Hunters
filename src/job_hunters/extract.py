"""The model call that reads a company out of free-form prose.

A Hacker News comment names its company somewhere in a paragraph of text. There
is no fixed place and no fixed form. This module asks a small model for the three
things the promotion loop needs from it: the company, the roles on offer and the
link the post gives for applying. The answer comes back as a structured output
validated against `Extraction`. Nothing here decides whether the company is worth
watching. That is the reviewer's call and `promote.py`'s job.

The structured sources (RemoteOK, Arbeitnow, Remotive) never come through here.
They already carry the company as a field.
"""

from __future__ import annotations

from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .judge import Usage

# A comment longer than this is cut before it reaches the model. Postings that
# run past it have said who they are long before this point.
MAX_TEXT_CHARS = 8_000
MAX_OUTPUT_TOKENS = 512


class Extraction(BaseModel):
    """What the model reads out of one posting. Sent to the API as the required output schema."""

    model_config = ConfigDict(extra="forbid")

    company: str | None = Field(
        description="The hiring company's name as the posting gives it and without "
        "suffixes such as a funding round, an accelerator batch or a legal form. "
        "Null when the posting names no company or it is a recruiter or agency "
        "hiring on behalf of unnamed clients.",
    )
    roles: list[str] = Field(
        description="The job titles the posting is hiring for, as written. Empty when "
        "it states none.",
    )
    careers_url: str | None = Field(
        description="The URL the posting gives for applying or for the company's "
        "jobs page, copied verbatim. Null when it gives none. Never a URL that is "
        "not in the text.",
    )


class ExtractError(Exception):
    """An extraction that could not be obtained for one posting.

    `fatal` marks the kinds that would fail the same way for every posting,
    such as a rejected key or an unknown model name, so the run stops instead
    of paying for the same error over and over.
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        """Keeps the message and whether the whole run should stop."""
        super().__init__(message)
        self.fatal = fatal


SYSTEM_PROMPT = (
    "You read job postings written as free-form text, one at a time, and report "
    "who is hiring. Answer in the required JSON shape. Take every value from the "
    "text itself. Never guess a company name, never complete a URL and never add "
    "a role the posting does not mention. A posting by a recruiting agency or a "
    "job board on behalf of unnamed clients has no company."
)


def render_posting(text: str) -> str:
    """The user turn: the posting's text, cut when it runs long."""
    body = text.strip()
    if len(body) > MAX_TEXT_CHARS:
        body = body[:MAX_TEXT_CHARS].rstrip() + "\n[text truncated]"
    return f"Posting:\n{body or '(empty)'}"


class Extractor:
    """Holds one client and one model for a whole run."""

    def __init__(self, client: Any, model: str) -> None:
        """Takes a ready client (real or fake) and the model name."""
        self._client = client
        self.model = model

    def extract(self, text: str) -> tuple[Extraction, Usage]:
        """Takes one posting's text and outputs one validated extraction with the token usage."""
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,  # the same posting should name the same company on a rerun
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": render_posting(text)}],
                output_format=Extraction,
            )
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as exc:
            raise ExtractError(f"{type(exc).__name__}: {exc}", fatal=True) from exc
        except anthropic.APIError as exc:
            raise ExtractError(f"{type(exc).__name__}: {exc}") from exc
        except ValidationError as exc:
            raise ExtractError(f"The answer did not fit the Extraction schema: {exc}") from exc
        extraction = response.parsed_output
        if extraction is None:
            raise ExtractError(f"No structured answer (stop_reason={response.stop_reason!r})")
        return extraction, Usage.from_response(response.usage)
