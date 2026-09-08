"""The LLM judge.

This module makes the API call to judge one posting at a time. It takes one
posting's text and outputs a `Verdict` with the token usage the API reported
so the caller can watch the cache. The system prompt is built once per run
and marked for caching with `cache_control: ephemeral`. The answer comes back
as a structured output validated against `Verdict`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import paths, regions
from .config import ConfigError, SearchProfile

log = logging.getLogger("job_hunters.judge")

# A description longer than this is cut before it reaches the model, which
# bounds the cost of one call at roughly 3k tokens of posting.
MAX_DESCRIPTION_CHARS = 12_000
MAX_OUTPUT_TOKENS = 1024

# Published limits as of September 2026. A model missing from this table is
# not an error. Callers report the symptom without quoting a number they
# cannot vouch for, which is what keeps this table's staleness visible
# instead of turning it into a confident wrong answer.
CACHE_MINIMUM_TOKENS: dict[str, int] = {
    "claude-haiku-4-5": 4096,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-5": 1024,
    "claude-opus-4-8": 1024,
    "claude-opus-5": 512,
}


def cache_minimum_tokens(model: str) -> int | None:
    """The shortest prefix this model caches or None when it is not in the table."""

    if model in CACHE_MINIMUM_TOKENS:
        return CACHE_MINIMUM_TOKENS[model]
    families = [name for name in CACHE_MINIMUM_TOKENS if model.startswith(name)]
    return CACHE_MINIMUM_TOKENS[max(families, key=len)] if families else None


class Verdict(BaseModel):
    """What the judge says about one posting. Sent to the API as the required output schema."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(
        ge=0, le=100,
        description="Fit with the rubric and the candidate's background (from 0 to 100).",
    )
    summary: str = Field(
        description="Two short plain-text sentences for a daily digest: what the role "
        "is and why it does or does not fit the candidate.",
    )
    rationale: str = Field(
        description="Two to five sentences justifying the score, citing specific "
        "responsibilities or requirements from the posting.",
    )
    matched_areas: list[str] = Field(
        description="Short phrases naming what in the posting matches the "
        "candidate's background. Empty when nothing does.",
    )
    concerns: list[str] = Field(
        description="Short phrases naming what argues against the fit, including "
        "missing requirements. Empty when nothing does.",
    )
    work_authorization: Literal["eligible", "unclear", "blocked"] = Field(
        description="Whether the posting text itself contradicts the candidate's "
        "declared work authorization, as defined in the instructions.",
    )


@dataclass(frozen=True)
class PostingText:
    """Everything the judge is shown about one posting."""

    company: str
    title: str
    location_raw: str | None
    region: str
    work_mode: str
    description: str


@dataclass(frozen=True)
class Usage:
    """Token counts from one call or the sum of several."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def from_response(cls, usage: Any) -> Usage:
        """Reads the SDK's usage object, treating an absent cache count as zero."""
        return cls(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        )

    def __add__(self, other: Usage) -> Usage:
        """Adds two usages field by field, so a run can keep a running total."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


class JudgeError(Exception):
    """A judgement that could not be obtained for one posting.

    `fatal` marks the kinds that would fail the same way for every posting,
    such as a rejected key or an unknown model name, so the run stops instead
    of burning through the cap collecting identical errors.
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        """Keeps the message and whether the whole run should stop."""
        super().__init__(message)
        self.fatal = fatal


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def load_profile_text(profile_dir: Path | None = None) -> str:
    """Join every markdown file in `profile/` in name order for the cached prefix."""

    directory = profile_dir or paths.PROFILE_DIR
    files = sorted(directory.glob("*.md")) if directory.is_dir() else []
    if not files:
        raise ConfigError(
            f"No markdown files in {directory}. The judge needs your CV there (`profile/cv.md`)"
        )
    parts = []
    for file in files:
        parts.append(f"<!-- {file.name} -->\n{file.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(parts)


def _describe_regions(tokens: list[str]) -> str:
    """Region tokens as the model should read them, with each group spelled out."""
    parts = []
    for token in tokens:
        members = regions.REGION_GROUPS.get(token)
        parts.append(f"{token} ({', '.join(sorted(members))})" if members else token)
    return "; ".join(parts)


def build_system_prompt(profile: SearchProfile, profile_text: str) -> str:
    """The stable prefix: candidate profile, rubric, declared facts and the scoring guide."""

    location = profile.location
    auth = location.work_authorization
    seniority = profile.seniority
    lines = [
        "You screen job postings for one specific candidate. For each posting you are "
        "given, judge how well the role fits this candidate and the search rubric "
        "below, and answer in the required JSON shape. Judge only from the posting "
        "text and the candidate's profile. Never invent requirements the posting "
        "does not state.",
        "",
        "# The candidate",
        "",
        profile_text.strip(),
        "",
        "# What the candidate is looking for",
        "",
        profile.scoring.rubric.strip() or "Roles matching the profile above.",
        "",
        f"Seniority sought: {', '.join(seniority.include) or 'any'}. "
        f"Not sought: {', '.join(seniority.exclude) or 'none'}.",
        f"Title patterns the search treats as a strong signal: "
        f"{', '.join(profile.titles.include)}.",
        f"Domain vocabulary that marks the target work: "
        f"{', '.join(profile.keywords.strong)}."
        f"Weaker signals: "
        f"{', '.join(profile.keywords.supporting) or 'none'}.",
        "",
        "# Declared facts you must not second-guess",
        "",
        f"The candidate is based in {location.base}. The following is declared and "
        "is not for you to infer:",
        f"- Can work without visa sponsorship in: {_describe_regions(auth.have)}.",
        f"- Would need sponsorship in: {_describe_regions(auth.need_sponsorship) or 'nowhere listed'}.",
        "",
        "Do not reason about immigration law and do not guess whether a company "
        "would sponsor. Location preferences are applied by code, not by you.",
        "",
        "For `work_authorization`, answer one question only: does the posting text "
        "itself state a requirement that contradicts the declared status?",
        "- blocked: the text explicitly requires something the candidate does not "
        "hold, such as citizenship of a specific country, a security clearance, or "
        "an existing right to work in a country listed under 'would need "
        "sponsorship' together with a statement that sponsorship is not offered.",
        "- unclear: the text hints at such a restriction without stating it plainly, "
        "or the requirement depends on which of several listed locations applies.",
        "- eligible: the text states nothing that conflicts with the declared "
        "status. This is the default when the posting is silent.",
        "",
        "# How to score",
        "",
        "Give a score from 0 to 100 for how well the role fits the rubric and the "
        "candidate's background:",
        "- 85 to 100: the core of the search. Post-training or evaluation research "
        "on language models (RLHF, RLAIF, DPO and other preference optimization, "
        "reward modelling, supervised fine-tuning, RL environments, benchmark and "
        "evaluation design) where the candidate's background is a direct match.",
        "- 70 to 84: research adjacent to the core, with real experiments: "
        "alignment, interpretability, red-teaming, training-data quality, model "
        "behaviour, or post-training in another modality. The candidate could "
        "credibly do the work.",
        "- 40 to 69: a technical role that touches the domain but is centred on "
        "something else: training or inference infrastructure, platform or "
        "product engineering around fine-tuning and evals, applied deployment, "
        "pretraining. Or a research role where the candidate's background is a "
        "stretch.",
        "- 10 to 39: engineering or research clearly outside the domain, or a role "
        "the candidate is not looking for: internships, roles that are principally "
        "people management, director level and above.",
        "- 0 to 9: not a technical research or engineering role at all: sales, "
        "marketing, legal, finance, recruiting, support, operations.",
        "",
        "Judge the work the role actually consists of, not the team it sits in or "
        "the words in the title: a post-training team can be hiring a Kubernetes "
        "engineer. Do not reward or penalize company size or prestige. A four-person "
        "lab doing post-training research outranks a large company's generic "
        "machine-learning opening. When the posting covers several possible teams "
        "or tracks, score the best fit among them and say which in the rationale.",
        "",
        "# Calibration examples",
        "",
        "- 'Research Scientist, Preference Learning' at a twelve-person lab: designs "
        "reward models and DPO variants and runs the ablations. Score 93: the core "
        "of the search and a direct match for the candidate's work on learning from "
        "disagreeing human labels.",
        "- 'Research Engineer, Red Teaming': builds adversarial evaluations and runs "
        "robustness experiments on production models. Score 76: adjacent research "
        "with real experiments.",
        "- 'Machine Learning Engineer, Training Platform': maintains the distributed "
        "training stack that post-training teams use, with no experiments of its "
        "own. Score 45: infrastructure next to the domain, not research in it.",
        "- 'Research Program Manager': plans compute allocation and coordinates "
        "research programs. Score 15: not a research or engineering role.",
        "- 'Enterprise Account Executive' at an AI lab: sells to enterprise accounts. "
        "Score 2: sales, however technical the product.",
        "",
        "# The answer",
        "",
        "- summary: two short plain-text sentences for a daily email digest, in "
        "this order: what the role is, then why it does or does not fit the "
        "candidate. No markdown and no company boilerplate.",
        "- rationale: two to five sentences justifying the score, citing specific "
        "responsibilities or requirements from the posting.",
        "- matched_areas: short phrases naming what in the posting matches the "
        "candidate's background. Empty when nothing does.",
        "- concerns: short phrases naming what argues against the fit, including "
        "missing requirements. Empty when nothing does.",
        "- work_authorization: as defined above.",
    ]
    return "\n".join(lines)


def render_posting(text: PostingText) -> str:
    """The user turn: one posting and nothing that stays the same between calls."""
    
    description = text.description.strip()
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS].rstrip() + "\n[description truncated]"
    location = text.location_raw or "not stated"
    return (
        f"Company: {text.company}\n"
        f"Title: {text.title}\n"
        f"Location: {location} (parsed as region {text.region}, {text.work_mode})\n"
        f"\n"
        f"Description:\n{description or '(empty)'}"
    )


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


def make_client(api_key: str) -> anthropic.Anthropic:
    """The SDK client. Transient failures (429, 5xx, timeouts) are retried three times with backoff."""
    return anthropic.Anthropic(api_key=api_key, max_retries=3)


class Judge:
    """Holds one client, one model and one cached prefix for a whole run."""

    def __init__(self, client: Any, model: str, system_prompt: str) -> None:
        """Takes a ready client (real or fake), the model name and the prefix to cache."""
        self._client = client
        self.model = model
        self.system_prompt = system_prompt

    def judge(self, text: PostingText) -> tuple[Verdict, Usage]:
        """Takes one posting and outputs one validated verdict with the token usage."""
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0, # the same posting should get the same verdict on a rerun
                system=[
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": render_posting(text)}],
                output_format=Verdict,
            )
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as exc:
            raise JudgeError(f"{type(exc).__name__}: {exc}", fatal=True) from exc
        except anthropic.APIError as exc:
            raise JudgeError(f"{type(exc).__name__}: {exc}") from exc
        except ValidationError as exc:
            raise JudgeError(f"The answer did not fit the Verdict schema: {exc}") from exc
        verdict = response.parsed_output
        if verdict is None:
            raise JudgeError(f"No structured answer (stop_reason={response.stop_reason!r})")
        return verdict, Usage.from_response(response.usage)
