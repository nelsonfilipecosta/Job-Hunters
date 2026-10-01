"""Tests for the tier a model suggests for a discovered company.

`FakeAnthropic` stands in for the API, so nothing here costs money.
"""

from __future__ import annotations

import anthropic
import pytest
from pydantic import ValidationError

from conftest import FakeAnthropic, api_error
from job_hunters.config import AppConfig, SearchProfile, Secrets, SystemConfig
from job_hunters.promote import Queued
from job_hunters.tables import Tier
from job_hunters.tiering import SYSTEM_PROMPT, TierSuggestion, describe, suggest_tier

PROFILE_DICT = {
    "titles": {"include": ["research scientist"], "exclude": ["sales"]},
    "keywords": {"strong": ["post-training"], "supporting": ["evaluation"]},
    "seniority": {"include": ["senior"], "exclude": ["intern"]},
    "location": {
        "base": "portugal",
        "priority": [{"work_modes": ["onsite"], "regions": ["portugal"]}],
        "acceptable": [{"work_modes": ["remote"], "regions": ["eu"]}],
        "work_authorization": {"have": ["eu"], "need_sponsorship": ["us"]},
    },
    "scoring": {
        "threshold": 70, "max_llm_scores_per_run": 300, "prompt_version": 1,
        "rubric": "Weight post-training.",
        "bands": [{"low": 70, "high": 100, "meaning": "A fit."},
                  {"low": 0, "high": 69, "meaning": "Not a fit."}],
    },
}


def _config(**secrets) -> AppConfig:
    """An AppConfig with no `.env` behind it, so a test says what is set."""
    return AppConfig(
        search_profile=SearchProfile.model_validate(PROFILE_DICT),
        system=SystemConfig.model_validate({"timezone": "Europe/Lisbon"}),
        watchlist=[],
        secrets=Secrets(_env_file=None, **secrets),
    )


def _queued(**overrides) -> Queued:
    """One queued company with a board, as the review page would hand it over."""
    fields = dict(
        id=1, name="Prior Labs", slug="prior-labs", ats="ashby", token="prior-labs",
        board_url="https://api.ashbyhq.com/posting-api/job-board/prior-labs",
        board_jobs=24, careers_url="https://jobs.ashbyhq.com/prior-labs",
        page_url="https://jobs.ashbyhq.com/prior-labs",
        roles=("Research Scientist, Foundation Model",), sightings=2, sources=("hn",),
        seen=(("Prior Labs | Berlin | ONSITE", "https://news.ycombinator.com/item?id=1"),),
    )
    fields.update(overrides)
    return Queued(**fields)


def suggestion(tier: str = "lab", because: str = "It trains foundation models.") -> TierSuggestion:
    """A complete TierSuggestion with boring defaults."""
    return TierSuggestion(tier=tier, because=because)


def test_the_schema_takes_only_the_four_tiers_and_forbids_extras() -> None:
    """A tier the watchlist cannot hold, or a stray field, would mean the prompt drifted."""
    assert TierSuggestion(tier="infra", because="Sells GPUs.").tier == "infra"
    with pytest.raises(ValidationError):
        TierSuggestion(tier="startup", because="Small.")
    with pytest.raises(ValidationError):
        TierSuggestion(tier="lab", because="Fine.", confidence=0.9)


def test_the_prompt_defines_every_tier_the_watchlist_can_hold() -> None:
    """A tier the model is never told about is one it can never answer."""
    for tier in Tier:
        assert tier.value in SYSTEM_PROMPT


def test_the_company_is_described_by_what_was_found_and_nothing_else() -> None:
    """The model judges the company, so it is given the company and not the search."""
    text = describe(_queued())
    assert "Prior Labs" in text
    assert "ashby 'prior-labs' with 24 open postings" in text
    assert "Research Scientist, Foundation Model" in text
    assert "Prior Labs | Berlin | ONSITE" in text
    assert "post-training" not in text, "the profile is not evidence about the company"


def test_a_company_with_nothing_but_a_name_is_still_describable() -> None:
    """Most of what a sighting carries is optional and a blank prompt would be a crash."""
    text = describe(_queued(ats=None, token=None, board_jobs=None, careers_url=None,
                            page_url=None, roles=(), seen=()))
    assert text == "Company: Prior Labs"


def test_the_suggestion_is_the_tier_and_the_reason_for_it() -> None:
    """The reviewer is choosing, so what the model thought is worth as much as its answer."""
    client = FakeAnthropic(suggestion(tier="lab", because="It trains foundation models."))
    assert suggest_tier(_queued(), _config(), client) == (
        Tier.LAB, "It trains foundation models."
    )


def test_the_call_is_cheap_deterministic_and_on_the_configured_model() -> None:
    """This runs on a page load, so it uses the small model and answers the same way twice."""
    client = FakeAnthropic(suggestion())
    suggest_tier(_queued(), _config(), client)
    request = client.requests[0]
    assert request["model"] == "claude-haiku-4-5", "the extract model from system_config.yaml"
    assert request["temperature"] == 0
    assert request["output_format"] is TierSuggestion


def test_no_api_key_means_no_suggestion_and_no_call() -> None:
    """An installation with no key still has to be able to approve a company."""
    assert suggest_tier(_queued(), _config()) is None


@pytest.mark.parametrize(
    "failure",
    [
        api_error(anthropic.AuthenticationError, 401),
        api_error(anthropic.RateLimitError, 429),
        api_error(anthropic.InternalServerError, 500),
        ValueError("something in the client"),
    ],
)
def test_a_failed_call_costs_the_page_nothing(failure: Exception) -> None:
    """A refused, throttled or broken call leaves the dropdown alone rather than the page."""
    assert suggest_tier(_queued(), _config(), FakeAnthropic(failure)) is None


def test_an_answer_without_a_structured_output_is_not_a_suggestion() -> None:
    """A model that stopped early returns nothing to read, which is not a tier."""
    assert suggest_tier(_queued(), _config(), FakeAnthropic(None)) is None
