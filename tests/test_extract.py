"""Tests for the model call that reads a company out of free-form prose.

`FakeAnthropic` stands in for the API, so nothing here costs money.
"""

from __future__ import annotations

import anthropic
import pytest
from pydantic import ValidationError

from conftest import FakeAnthropic, api_error
from job_hunters.extract import (
    MAX_TEXT_CHARS,
    ExtractError,
    Extraction,
    Extractor,
    render_posting,
)


def extraction(**overrides) -> Extraction:
    """A complete Extraction with boring defaults."""
    fields = dict(company="Prior Labs", roles=["Research Scientist"], careers_url="https://jobs.ashbyhq.com/prior-labs")
    fields.update(overrides)
    return Extraction(**fields)


def test_the_schema_allows_an_unnamed_company_and_forbids_extras() -> None:
    """A recruiter's post has no company and a stray field would mean the prompt drifted."""
    assert Extraction(company=None, roles=[], careers_url=None).company is None
    with pytest.raises(ValidationError):
        Extraction(company="X", roles=[], careers_url=None, location="Lisbon")
    with pytest.raises(ValidationError):
        Extraction(company="X", roles=[])  # careers_url is not optional to omit


def test_the_posting_is_rendered_into_the_user_turn_and_long_text_is_cut() -> None:
    """The model reads the text and nothing else and never more than the cap."""
    assert render_posting("Acme | Research Scientist").endswith("Acme | Research Scientist")
    long = "x" * (MAX_TEXT_CHARS + 500)
    rendered = render_posting(long)
    assert len(rendered) < MAX_TEXT_CHARS + 100
    assert "truncated" in rendered


def test_the_extractor_returns_the_parsed_answer_and_the_usage() -> None:
    """One call, one validated answer and one usage. The model name comes from the caller."""
    client = FakeAnthropic(extraction())
    got, usage = Extractor(client, "claude-haiku-4-5").extract("Prior Labs | Research Scientist")
    assert got.company == "Prior Labs"
    assert usage.input_tokens == 1500
    request = client.requests[0]
    assert request["model"] == "claude-haiku-4-5"
    assert request["temperature"] == 0
    assert request["output_format"] is Extraction
    assert "Prior Labs" in request["messages"][0]["content"]


def test_a_rejected_key_is_fatal_and_a_rate_limit_is_not() -> None:
    """A bad key fails every posting the same way. A 429 is worth trying the next one."""
    fatal = Extractor(FakeAnthropic(api_error(anthropic.AuthenticationError, 401)), "m")
    with pytest.raises(ExtractError) as exc:
        fatal.extract("text")
    assert exc.value.fatal

    transient = Extractor(FakeAnthropic(api_error(anthropic.RateLimitError, 429)), "m")
    with pytest.raises(ExtractError) as exc:
        transient.extract("text")
    assert not exc.value.fatal


def test_an_answer_without_structured_output_is_an_error_not_a_crash() -> None:
    """A refusal or a cut-off answer has no parsed output and must be reported, not dereferenced."""
    with pytest.raises(ExtractError, match="No structured answer"):
        Extractor(FakeAnthropic(None), "m").extract("text")
