"""Loads and validates the three config files this project runs on.

    search_profile.yaml       what you are looking for
    system_config.yaml        how the system runs
    companies_watchlist.yaml  which companies to watch

Defines a pydantic model for each file's shape, plus a `load_*` function per
file that reads it, checks it against that model and returns either a fully
typed object or a `ConfigError` naming exactly what is wrong and where. Every
model forbids unknown keys, so a typo in the file is a startup error rather
than a setting that silently never took effect.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class ScoreBand(StrictModel):
    """One stretch of the 0-100 scale and what the judge should put in it."""

    low: int = Field(ge=0, le=100)
    high: int = Field(ge=0, le=100)
    meaning: NonEmptyStr

    @model_validator(mode="after")
    def _bounds_in_order(self) -> ScoreBand:
        """Rejects a band written back to front, which would silently match nothing."""
        if self.low > self.high:
            raise ValueError(f"band {self.low}-{self.high} has its bounds reversed")
        return self


class ScoreExample(StrictModel):
    """One worked example anchoring a point on the scale."""

    posting: NonEmptyStr
    score: int = Field(ge=0, le=100)
    reason: NonEmptyStr


class ScoringConfig(StrictModel):
    threshold: int = Field(ge=0, le=100)
    max_llm_scores_per_run: int = Field(gt=0)
    prompt_version: int = Field(ge=1, default=1)
    rubric: NonEmptyStr
    bands: list[ScoreBand] = Field(min_length=1)
    guidance: str = ""
    examples: list[ScoreExample] = []

    @field_validator("rubric")
    @classmethod
    def _rubric_says_something(cls, value: str) -> str:
        """Rejects a rubric of only whitespace, which a length check alone would let through."""
        if not value.strip():
            raise ValueError("Rubric must not be blank: the judge needs criteria to score against.")
        return value

    @model_validator(mode="after")
    def _bands_cover_the_scale(self) -> ScoringConfig:
        """Rejects bands that leave a gap or overlap, which would leave scores unguided."""
        ordered = sorted(self.bands, key=lambda band: band.low)
        if ordered[0].low != 0 or ordered[-1].high != 100:
            raise ValueError(
                f"Bands must cover 0 to 100, but they run "
                f"{ordered[0].low} to {ordered[-1].high}"
            )
        for lower, upper in zip(ordered, ordered[1:]):
            if upper.low != lower.high + 1:
                raise ValueError(
                    f"Bands must be contiguous, but {lower.low}-{lower.high} is "
                    f"followed by {upper.low}-{upper.high}"
                )
        return self


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


_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class EmailConfig(StrictModel):
    """How the digest is sent, but never to or from whom."""

    smtp_host: NonEmptyStr = "smtp.gmail.com"
    smtp_port: int = Field(default=587, gt=0, lt=65536)


class RepeatSuppressionConfig(StrictModel):
    demote_after: int = Field(default=2, ge=1)
    suppress_after: int = Field(default=3, ge=1)
    reset_on_score_delta: int = Field(default=10, ge=0, le=100)
    enabled: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> RepeatSuppressionConfig:
        """Rejects a `suppress_after` threshold lower than `demote_after`."""
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
    email: EmailConfig = EmailConfig()
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
# .env
# ---------------------------------------------------------------------------


class Secrets(BaseSettings):
    """The private secrets read from the environment first and from `.env` second."""

    model_config = SettingsConfigDict(
        env_file=paths.PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = None  # Phase 2: the judge
    smtp_password: SecretStr | None = None  # Phase 3: the digest email
    action_token_secret: SecretStr | None = None  # Phase 4: signed action links

    # Phase 3: who the digest goes to, what it is sent as and the account it authenticates with.
    digest_to: SecretStr | None = None
    digest_from: SecretStr | None = None
    smtp_username: SecretStr | None = None

    @field_validator("digest_to", "digest_from", "smtp_username")
    @classmethod
    def _looks_like_email(cls, value: SecretStr | None) -> SecretStr | None:
        """Rejects an address that could never deliver, naming no value."""
        if value is not None:
            plain = value.get_secret_value().strip()
            if plain and not _EMAIL_RE.fullmatch(plain):
                raise ValueError("Does not look like an email address.")
        return value

    def require(self, name: str) -> str:
        """The plain value of one secret or a ConfigError naming the variable."""
        value: SecretStr | None = getattr(self, name)
        if value is None or not value.get_secret_value().strip():
            raise ConfigError(
                f"{name.upper()} is not set. Add it to .env (copy .env.example) "
                f"or export it in the environment."
            )
        return value.get_secret_value().strip()

    DISPLAYABLE: ClassVar[frozenset[str]] = frozenset(
        {"digest_to", "digest_from", "smtp_username"}
    )

    def present(self) -> dict[str, bool]:
        """Which secrets are set by variable name. Never their values."""
        return {
            name.upper(): bool(self._plain(name)) for name in type(self).model_fields
        }

    def summary(self) -> dict[str, str]:
        """One line per secret for `show-config`: email addresses by value and credentials masked."""
        lines = {}
        for name in type(self).model_fields:
            plain = self._plain(name)
            if not plain:
                lines[name.upper()] = "missing"
            else:
                lines[name.upper()] = plain if name in self.DISPLAYABLE else "present"
        return lines

    def _plain(self, name: str) -> str:
        """The stripped value of one field or an empty string when it is unset."""
        value: SecretStr | None = getattr(self, name)
        return value.get_secret_value().strip() if value is not None else ""


def load_secrets(env_file: Path | None = None) -> Secrets:
    """Reads the secrets from a specific `.env` when given (tests) or the project's."""
    target = env_file if env_file is not None else paths.PROJECT_ROOT / ".env"
    try:
        return Secrets() if env_file is None else Secrets(_env_file=env_file)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(target, exc)) from exc


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    search_profile: SearchProfile
    system: SystemConfig
    watchlist: list[CompanyEntry]
    secrets: Secrets


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
    """Load and validate the three files plus the secrets. Raises ConfigError with a readable message."""
    base = config_dir or paths.CONFIG_DIR
    return AppConfig(
        search_profile=load_search_profile(base / "search_profile.yaml"),
        system=load_system_config(base / "system_config.yaml"),
        watchlist=load_watchlist(base / "companies_watchlist.yaml"),
        secrets=load_secrets()
    )
