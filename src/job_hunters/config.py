"""Loads and validates the three config files this project runs on.

    search_profile.yaml       what you are looking for
    system_config.yaml        how the system runs
    companies_watchlist.yaml  which companies to watch

Defines a pydantic model for each file's shape, plus a `load_*` function per
file that reads it, checks it against that model and returns either a fully
typed object or a `ConfigError` naming exactly what is wrong and where. Every
model forbids unknown keys, so a typo in the file is a startup error rather
than a setting that silently never took effect.

Nothing secret lives here. SMTP passwords and API keys come from the
environment (`.env`), which is gitignored and checked at startup.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from . import paths, regions
from .models import AtsType, Tier, WorkMode


class ConfigError(Exception):
    """Raised for any problem while loading a configuration (with a readable message)."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# search_profile.yaml
# ---------------------------------------------------------------------------


NonEmptyStr = Annotated[str, Field(min_length=1)]


class TitlesConfig(StrictModel):
    include: list[NonEmptyStr] = Field(min_length=1)
    exclude: list[NonEmptyStr] = []


class KeywordsConfig(StrictModel):
    strong: list[NonEmptyStr] = Field(min_length=1)
    supporting: list[NonEmptyStr] = []


class SeniorityConfig(StrictModel):
    include: list[NonEmptyStr] = []
    exclude: list[NonEmptyStr] = []


class LocationRule(StrictModel):
    work_modes: list[WorkMode] = Field(min_length=1)
    regions: list[NonEmptyStr] = Field(min_length=1)

    @field_validator("regions")
    @classmethod
    def _known_regions(cls, value: list[str]) -> list[str]:
        """Rejects any region token not found in the region vocabulary."""
        if unknown := regions.unknown_tokens(value):
            raise ValueError(
                f"Unknown region token(s): {', '.join(unknown)}. "
                f"Valid tokens are countries or groups listed in job_hunters/regions.py "
                f"Note: 'eu' does not include switzerland or uk."
            )
        return value

    def matches(self, region: str, work_mode: str) -> bool:
        """Checks whether a job's region and work mode satisfy this rule."""
        return region in regions.expand(self.regions) and work_mode in {
            m.value for m in self.work_modes
        }


class WorkAuthorizationConfig(StrictModel):
    have: list[NonEmptyStr] = Field(min_length=1)
    need_sponsorship: list[NonEmptyStr] = []

    @field_validator("have", "need_sponsorship")
    @classmethod
    def _known_regions(cls, value: list[str]) -> list[str]:
        """Rejects any region token not found in the region vocabulary."""
        if unknown := regions.unknown_tokens(value):
            raise ValueError(f"Unknown region token(s): {', '.join(unknown)}")
        return value


class LocationConfig(StrictModel):
    base: NonEmptyStr
    priority: list[LocationRule] = Field(min_length=1)
    acceptable: list[LocationRule] = []
    work_authorization: WorkAuthorizationConfig

    @model_validator(mode="after")
    def _no_overlap_with_sponsorship(self) -> LocationConfig:
        """Rejects a region listed as both authorized and needing sponsorship."""
        both = set(self.work_authorization.have) & set(
            self.work_authorization.need_sponsorship
        )
        if both:
            raise ValueError(
                f"Region(s) listed as both authorized and needing sponsorship: "
                f"{', '.join(sorted(both))}"
            )
        return self


class ScoringConfig(StrictModel):
    threshold: int = Field(ge=0, le=100)
    max_llm_scores_per_run: int = Field(gt=0)
    prompt_version: int = Field(ge=1, default=1)
    rubric: str = ""


class SearchProfile(StrictModel):
    titles: TitlesConfig
    keywords: KeywordsConfig
    seniority: SeniorityConfig = SeniorityConfig()
    location: LocationConfig
    scoring: ScoringConfig

    def eligible_regions(self) -> frozenset[str]:
        """Returns every country reachable without sponsorship (groups expanded)."""
        return regions.expand(self.location.work_authorization.have)


# ---------------------------------------------------------------------------
# system_config.yaml
# ---------------------------------------------------------------------------


# "every 2h" | "daily 08:00" | "weekly mon 06:00"
_SCHEDULE_RE = re.compile(
    r"^(?:every \d+[mhd]"
    r"|daily \d{2}:\d{2}"
    r"|weekly (?:mon|tue|wed|thu|fri|sat|sun) \d{2}:\d{2})$"
)

ScheduleSpec = Annotated[str, Field(pattern=_SCHEDULE_RE.pattern)]


class SchedulesConfig(StrictModel):
    ingest: ScheduleSpec = "every 2h"
    score: ScheduleSpec = "every 2h"
    digest: ScheduleSpec = "daily 08:00"
    discovery: ScheduleSpec = "weekly mon 06:00"
    backup: ScheduleSpec = "weekly sun 03:00"


class ModelsConfig(StrictModel):
    judge: NonEmptyStr = "claude-haiku-4-5"
    extract: NonEmptyStr = "claude-haiku-4-5"
    tailor: NonEmptyStr = "claude-opus-5"


class EmailConfig(StrictModel):
    to: NonEmptyStr
    from_address: str | None = None
    smtp_host: NonEmptyStr = "smtp.gmail.com"
    smtp_port: int = Field(default=587, gt=0, lt=65536)
    smtp_username: str | None = None  # password comes from SMTP_PASSWORD in `.env`

    @field_validator("to", "from_address")
    @classmethod
    def _looks_like_email(cls, value: str | None) -> str | None:
        """Rejects a value that does not look like an email address."""
        if value is not None and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError(f"{value!r} does not look like an email address")
        return value


class RepeatSuppressionConfig(StrictModel):
    demote_after: int = Field(default=2, ge=1)
    suppress_after: int = Field(default=3, ge=1)
    reset_on_score_delta: int = Field(default=10, ge=0, le=100)
    enabled: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> RepeatSuppressionConfig:
        """Rejects a suppress_after threshold lower than demote_after."""
        if self.suppress_after < self.demote_after:
            raise ValueError(
                f"The suppress_after ({self.suppress_after}) value must be >= than "
                f"the demote_after ({self.demote_after}). Otherwise jobs would be "
                f"hidden before they are ever demoted."
            )
        return self


class DigestConfig(StrictModel):
    followup_after_days: int = Field(default=14, ge=1)
    repeat_suppression: RepeatSuppressionConfig = RepeatSuppressionConfig()


class SystemConfig(StrictModel):
    timezone: NonEmptyStr = "Europe/Lisbon"
    schedules: SchedulesConfig = SchedulesConfig()
    models: ModelsConfig = ModelsConfig()
    email: EmailConfig
    digest: DigestConfig = DigestConfig()

    @field_validator("timezone")
    @classmethod
    def _real_timezone(cls, value: str) -> str:
        """Rejects a timezone name that is not a real IANA timezone."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value


# ---------------------------------------------------------------------------
# companies_watchlist.yaml
# ---------------------------------------------------------------------------


# Which ats_config keys each adapter requires
REQUIRED_ATS_KEYS: dict[str, tuple[str, ...]] = {
    AtsType.GREENHOUSE: ("token",),
    AtsType.LEVER: ("token",),
    AtsType.ASHBY: ("token",),
    AtsType.WORKDAY: ("tenant", "wd", "site"),
}


class CompanyEntry(StrictModel):
    slug: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$")]
    name: NonEmptyStr
    ats: AtsType
    tier: Tier = Tier.DISCOVERED
    active: bool = True
    token: str | None = None
    ats_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_ats_config(self) -> CompanyEntry:
        """Merges token into ats_config and checks every key the ATS requires is present."""
        merged = dict(self.ats_config)
        if self.token:
            merged.setdefault("token", self.token)
        missing = [k for k in REQUIRED_ATS_KEYS[self.ats] if k not in merged]
        if missing:
            raise ValueError(
                f"{self.slug}: ats '{self.ats}' requires {', '.join(missing)} "
                f"(set `token:` for greenhouse/lever/ashby or `ats_config:` for workday)"
            )
        self.ats_config = merged
        return self

    def to_company_kwargs(self) -> dict[str, Any]:
        """Column values for a `companies` row."""
        return {
            "slug": self.slug,
            "name": self.name,
            "ats_type": self.ats.value,
            "ats_config": self.ats_config,
            "tier": self.tier.value,
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    search_profile: SearchProfile
    system: SystemConfig
    watchlist: list[CompanyEntry]


def _read_yaml(path: Path) -> Any:
    """Reads and parses a YAML file, raising ConfigError if missing or malformed."""
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML:\n{exc}") from exc


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    """Turns a pydantic ValidationError into a short message naming each bad field."""
    lines = [f"{path.name} is invalid:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)


def load_search_profile(path: Path | None = None) -> SearchProfile:
    """Loads and validates `search_profile.yaml`, defaulting to its path in `config/`."""
    target = path or paths.SEARCH_PROFILE_PATH
    try:
        return SearchProfile.model_validate(_read_yaml(target))
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(target, exc)) from exc


def load_system_config(path: Path | None = None) -> SystemConfig:
    """Loads and validates `system_config.yaml`, defaulting to its path in `config/`."""
    target = path or paths.SYSTEM_CONFIG_PATH
    try:
        return SystemConfig.model_validate(_read_yaml(target))
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(target, exc)) from exc


def load_watchlist(path: Path | None = None) -> list[CompanyEntry]:
    """Loads `companies_watchlist.yaml` and rejects any duplicate company slug."""
    target = path or paths.WATCHLIST_PATH
    raw = _read_yaml(target)
    if not isinstance(raw, list):
        raise ConfigError(f"{target.name} must be a YAML list of companies")
    try:
        entries = TypeAdapter(list[CompanyEntry]).validate_python(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(target, exc)) from exc

    counts = Counter(entry.slug for entry in entries)
    duplicates = sorted(slug for slug, count in counts.items() if count > 1)
    if duplicates:
        raise ConfigError(f"{target.name}: duplicate slug(s): {', '.join(duplicates)}")
    return entries


def load_all(config_dir: Path | None = None) -> AppConfig:
    """Load and validate all three files. Raises ConfigError with a readable message."""
    base = config_dir or paths.CONFIG_DIR
    return AppConfig(
        search_profile=load_search_profile(base / "search_profile.yaml"),
        system=load_system_config(base / "system_config.yaml"),
        watchlist=load_watchlist(base / "companies_watchlist.yaml"),
    )
