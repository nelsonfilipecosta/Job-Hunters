"""The tier a model suggests for a discovered company

The watchlist files every company under a tier, so the review page asks a cheap
model what kind of company this looks like and preselects the answer. The reviewer
still chooses and nothing here writes anything.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import AppConfig
from .judge import make_client
from .promote import Queued
from .tables import Tier

log = logging.getLogger("job_hunters.tiering")

MAX_OUTPUT_TOKENS = 256

# What each tier means, in the words the reviewer would use. Sent as the prompt
# and kept here rather than in `search_profile.yaml`, because these are labels
# this codebase defines and not part of anybody's search.
TIER_MEANINGS: dict[str, str] = {
    Tier.LAB: (
        "an AI lab: training or researching frontier or foundation models is what "
        "the company is for, whatever its size"
    ),
    Tier.BIGTECH: (
        "a large established technology company that is not primarily an AI lab, "
        "however much AI work it does"
    ),
    Tier.INFRA: (
        "tooling, platform, data or compute sold to the people who build models: "
        "serving, training infrastructure, evaluation products, GPUs, developer tools"
    ),
    Tier.DISCOVERED: (
        "anything else, including a company whose business the evidence does not settle"
    ),
}

SYSTEM_PROMPT = (
    "You file companies into one of four tiers for a job seeker's watchlist. You are "
    "given what a job board and a hiring post revealed about one company and nothing "
    "else. Answer in the required JSON shape.\n\n"
    + "\n".join(f"- {tier}: {meaning}" for tier, meaning in TIER_MEANINGS.items())
    + "\n\nJudge the company, never the roles it happens to be hiring for: a lab "
    "hiring a recruiter is still a lab and a bank hiring researchers is not one. "
    "Answer `discovered` whenever the evidence does not settle it. Guessing is worse "
    "than leaving it to the reviewer."
)


class TierSuggestion(BaseModel):
    """What the model thinks this company is. Sent to the API as the required output schema."""

    model_config = ConfigDict(extra="forbid")

    tier: Literal["lab", "bigtech", "infra", "discovered"] = Field(
        description="The tier this company belongs in, or `discovered` when unsure."
    )
    because: str = Field(
        description="One short sentence naming the evidence that decided it.",
    )


def describe(candidate: Queued) -> str:
    """The user turn: everything known about one queued company and nothing else."""
    lines = [f"Company: {candidate.name}"]
    if candidate.careers_url:
        lines.append(f"Careers page: {candidate.careers_url}")
    if candidate.ats:
        lines.append(
            f"Job board: {candidate.ats} '{candidate.token}' "
            f"with {candidate.board_jobs or 0} open postings"
        )
    if candidate.roles:
        lines.append(f"Roles seen: {'; '.join(candidate.roles[:12])}")
    for title, _ in candidate.seen:
        lines.append(f"Seen hiring: {title[:160]}")
    return "\n".join(lines)


def suggest_tier(
    candidate: Queued, config: AppConfig, client: Any | None = None
) -> tuple[Tier, str] | None:
    """The tier a model would file this company under or None when it could not say.

    Never raises. The review page is worth showing without a suggestion and an
    installation with no API key must still be able to approve a company.
    """
    if client is None:
        api_key = config.secrets.optional("anthropic_api_key")
        if not api_key:
            return None
        client = make_client(api_key)
    try:
        response = client.messages.parse(
            model=config.system.models.extract,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,  # the same company should land in the same tier on a reload
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": describe(candidate)}],
            output_format=TierSuggestion,
        )
        suggestion = response.parsed_output
        if suggestion is None:
            return None
        return Tier(suggestion.tier), suggestion.because
    except (anthropic.APIError, ValidationError, ValueError) as exc:
        log.info("no tier suggestion for %s: %s", candidate.name, exc)
        return None
