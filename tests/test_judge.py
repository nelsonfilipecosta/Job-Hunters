"""Tests for the LLM judge.

`FakeAnthropic` stands in for the API, so nothing here costs money.
"""

from __future__ import annotations

from dataclasses import replace

import anthropic
import httpx
import pytest
from pydantic import ValidationError

from conftest import FakeAnthropic, api_error, verdict
from job_hunters.config import ConfigError, load_search_profile
from job_hunters.judge import (
    MAX_DESCRIPTION_CHARS,
    Judge,
    JudgeError,
    PostingText,
    Usage,
    Verdict,
    build_system_prompt,
    cache_minimum_tokens,
    load_profile_text,
    render_posting,
)

TEXT = PostingText(
    company="Acme",
    title="Research Scientist, Post-Training",
    location_raw="Lisbon, Portugal",
    region="portugal",
    work_mode="onsite",
    description="We do RLHF.",
)


def test_the_verdict_schema_requires_every_field_and_forbids_extras() -> None:
    """An optional field is one the model may skip and a digest entry with no summary is useless."""
    schema = Verdict.model_json_schema()
    assert set(schema["required"]) == {
        "score", "summary", "rationale", "matched_areas", "concerns", "work_authorization",
    }
    assert schema["additionalProperties"] is False


def test_a_verdict_outside_the_score_range_is_rejected() -> None:
    """The API does not enforce the 0-100 bounds, so the model does on the way back."""
    with pytest.raises(ValidationError):
        verdict(101)
    with pytest.raises(ValidationError):
        verdict(50, work_authorization="maybe")


def test_the_system_prompt_is_stable_and_carries_profile_rubric_and_declared_facts() -> None:
    """The cached prefix is deterministic and holds everything the judge must know beyond the posting."""
    profile = load_search_profile()
    first = build_system_prompt(profile, "# CV\nPhD in NLP.")
    second = build_system_prompt(profile, "# CV\nPhD in NLP.")
    assert first == second
    assert "PhD in NLP." in first
    assert profile.scoring.rubric.strip() in first
    assert "eu (austria" in first, "region groups are spelled out for the model"
    assert "Would need sponsorship in: uk; us" in first
    assert "Do not reason about immigration" in first


def test_the_posting_is_rendered_into_the_user_turn_and_long_descriptions_are_cut() -> None:
    """Company, title, parsed location lead and a very long body is truncated with a marker."""
    rendered = render_posting(TEXT)
    assert rendered.startswith(
        "Company: Acme\nTitle: Research Scientist, Post-Training\n"
        "Location: Lisbon, Portugal (parsed as region portugal, onsite)"
    )
    assert rendered.endswith("Description:\nWe do RLHF.")

    long_text = replace(TEXT, description="x" * (MAX_DESCRIPTION_CHARS + 500))
    cut = render_posting(long_text)
    assert cut.endswith("[description truncated]")
    assert len(cut) < MAX_DESCRIPTION_CHARS + 300


def test_the_judge_sends_a_cached_prefix_and_returns_the_parsed_verdict() -> None:
    """The prefix is marked ephemeral, the schema is `Verdict` and the usage comes back."""
    client = FakeAnthropic(verdict(83, concerns=["no RL"]))
    judge = Judge(client, "claude-haiku-4-5", "prefix")

    result, usage = judge.judge(TEXT)

    assert result.score == 83 and result.concerns == ["no RL"]
    assert usage == Usage(input_tokens=1500, output_tokens=200,
                          cache_creation_input_tokens=4300, cache_read_input_tokens=0)
    (request,) = client.requests
    assert request["system"] == [
        {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}}
    ]
    assert request["output_format"] is Verdict
    assert request["max_tokens"] == 1024 and request["temperature"] == 0
    assert request["model"] == "claude-haiku-4-5"
    assert "Title: Research Scientist, Post-Training" in request["messages"][0]["content"]


def test_a_rejected_key_is_fatal_and_a_rate_limit_is_not() -> None:
    """Errors that repeat for every posting stop the run. Transient ones do not."""
    with pytest.raises(JudgeError) as fatal:
        Judge(FakeAnthropic(api_error(anthropic.AuthenticationError, 401)), "m", "p").judge(TEXT)
    assert fatal.value.fatal

    with pytest.raises(JudgeError) as transient:
        Judge(FakeAnthropic(api_error(anthropic.RateLimitError, 429)), "m", "p").judge(TEXT)
    assert not transient.value.fatal

    lost = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(JudgeError) as connection:
        Judge(FakeAnthropic(lost), "m", "p").judge(TEXT)
    assert not connection.value.fatal


def test_an_answer_without_structured_output_is_an_error_not_a_crash() -> None:
    """A response with no parsed output (a refusal, a cut-off) is reported and not dereferenced."""
    judge = Judge(FakeAnthropic(None), "m", "p")
    with pytest.raises(JudgeError, match="(?i)no structured answer"):
        judge.judge(TEXT)


def test_the_cache_minimum_is_looked_up_per_model() -> None:
    """The judge model comes from config, so its cache minimum can never be one model's constant."""
    assert cache_minimum_tokens("claude-haiku-4-5") == 4096
    assert cache_minimum_tokens("claude-sonnet-5") == 1024
    assert cache_minimum_tokens("claude-opus-5") == 512


def test_a_pinned_model_release_falls_back_to_its_family() -> None:
    """Naming an exact release in config still resolves a number."""
    assert cache_minimum_tokens("claude-haiku-4-5-20251001") == 4096


def test_an_unknown_model_reports_no_minimum_rather_than_a_guess() -> None:
    """A model the table has not caught up with must not be given another model's number."""
    assert cache_minimum_tokens("claude-something-7") is None


def test_usage_adds_field_by_field() -> None:
    """A run keeps one running total across calls."""
    assert Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40) == Usage(11, 22, 33, 44)


def test_profile_text_joins_every_markdown_file_in_name_order(tmp_path) -> None:
    """Achievement records added later join the CV in the prefix without any code change."""
    (tmp_path / "cv.md").write_text("# CV\n", encoding="utf-8")
    (tmp_path / "achievement-eval.md").write_text("Built evals.", encoding="utf-8")
    (tmp_path / "thesis.pdf").write_bytes(b"%PDF")

    text = load_profile_text(tmp_path)

    assert text.index("achievement-eval.md") < text.index("cv.md")
    assert "Built evals." in text and "PDF" not in text


def test_an_empty_profile_directory_is_a_config_error(tmp_path) -> None:
    """No CV means no judge: said plainly rather than as an empty prompt."""
    with pytest.raises(ConfigError, match="cv.md"):
        load_profile_text(tmp_path)
