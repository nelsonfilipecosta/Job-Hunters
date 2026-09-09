"""Tests that invalid configuration is refused loudly and by name.

`config.py` reads three YAML files and turns them into typed Python objects. Its
real job is not loading but rejecting. A YAML-driven system that silently
ignores a misspelled key is the worst kind of broken: it starts, it looks
healthy and it quietly does the wrong thing. Write `stong:` for `strong:` and
the keyword list is simply empty, with nothing to connect the missing jobs back
to the typo.

Most tests here take the real config, break exactly one thing and assert that
loading it raises. `match=` checks the error message too, since a test that
only checked "it raised" would pass even when the message was useless.

A few tests do the opposite and confirm valid input still loads. Validation can
fail in two directions and only one of them is loud: a rule tightened too far
rejects a config you legitimately need.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from job_hunters import paths, regions
from job_hunters.config import (
    ConfigError,
    Secrets,
    load_all,
    load_search_profile,
    load_secrets,
    load_system_config,
    load_watchlist,
)

SECRET_NAMES = (
    "ANTHROPIC_API_KEY", "SMTP_PASSWORD", "ACTION_TOKEN_SECRET",
    "DIGEST_TO", "DIGEST_FROM", "SMTP_USERNAME",
)


@pytest.fixture
def no_secret_env(monkeypatch) -> None:
    """Clears the three secret variables so a test only sees the `.env` file it wrote."""
    for name in SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_repo_config_is_valid() -> None:
    """The three YAML files in this repository load and validate together."""
    config = load_all()
    assert config.watchlist, "watchlist should not be empty"
    assert config.system.timezone == "Europe/Lisbon"


def test_watchlist_entries_all_resolve_a_token() -> None:
    """Every watched company has a board token without which it fetches nothing."""
    for entry in load_watchlist():
        assert entry.ats_config.get("token"), f"{entry.slug} has no board token"


def test_declared_work_authorization_matches_the_plan() -> None:
    """Portugal, Switzerland and Canada need no sponsorship. The UK and the US do."""
    eligible = load_search_profile().eligible_regions()
    assert "portugal" in eligible
    assert "switzerland" in eligible
    assert "canada" in eligible
    assert "uk" not in eligible
    assert "us" not in eligible


def test_eu_excludes_switzerland_and_uk() -> None:
    """`eu` covers the EU-27 only, so Switzerland and the UK need listing separately."""
    eu = regions.expand(["eu"])
    assert "portugal" in eu
    assert "germany" in eu
    assert "switzerland" not in eu
    assert "uk" not in eu


def test_country_tokens_pass_through_expansion() -> None:
    """A country token expands to itself, so config can mix countries and groups."""
    assert regions.expand(["switzerland", "canada"]) == frozenset(
        {"switzerland", "canada"}
    )


def test_every_group_member_is_a_known_country() -> None:
    """No region group names a country missing from the master list."""
    for group, members in regions.REGION_GROUPS.items():
        assert members <= regions.COUNTRIES, f"{group} names an unknown country"


def _write(tmp_path: Path, name: str, data: object) -> Path:
    """Dump a modified config into a temporary file and return its path."""
    target = tmp_path / name
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    return target


def _valid_profile() -> dict:
    """Load the real `search_profile.yaml` as a plain dict, ready to be broken."""
    return yaml.safe_load(paths.SEARCH_PROFILE_PATH.read_text(encoding="utf-8"))


def _valid_system() -> dict:
    """Load the real `system_config.yaml` as a plain dict, ready to be broken."""
    return yaml.safe_load(paths.SYSTEM_CONFIG_PATH.read_text(encoding="utf-8"))


def test_misspelled_key_is_rejected(tmp_path: Path) -> None:
    """A misspelled key is refused by name instead of being silently ignored."""
    profile = _valid_profile()
    profile["keywords"]["stong"] = profile["keywords"].pop("strong")
    with pytest.raises(ConfigError, match="stong"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_unknown_region_token_is_rejected(tmp_path: Path) -> None:
    """A misspelled region is refused instead of quietly matching nothing."""
    profile = _valid_profile()
    profile["location"]["priority"][0]["regions"] = ["portugal", "swizerland"]
    with pytest.raises(ConfigError, match="swizerland"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_a_region_cannot_be_both_authorized_and_sponsored(tmp_path: Path) -> None:
    """A region cannot be declared both authorized and in need of sponsorship."""
    profile = _valid_profile()
    profile["location"]["work_authorization"]["need_sponsorship"] = ["us", "canada"]
    with pytest.raises(ConfigError, match="canada"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_a_missing_rubric_is_rejected(tmp_path: Path) -> None:
    """The rubric states what the search is for, so an absent one is a startup error."""
    profile = _valid_profile()
    del profile["scoring"]["rubric"]
    with pytest.raises(ConfigError, match="rubric"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_a_blank_rubric_is_rejected(tmp_path: Path) -> None:
    """Whitespace passes a length check but tells the judge nothing, so it is refused too."""
    profile = _valid_profile()
    profile["scoring"]["rubric"] = "   \n  "
    with pytest.raises(ConfigError, match="blank"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_a_gap_between_score_bands_is_rejected(tmp_path: Path) -> None:
    """A score landing in a gap would have no meaning attached, so the model would invent one."""
    profile = _valid_profile()
    profile["scoring"]["bands"] = [
        {"low": 0, "high": 40, "meaning": "no"},
        {"low": 60, "high": 100, "meaning": "yes"},
    ]
    with pytest.raises(ConfigError, match="contiguous"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_overlapping_score_bands_are_rejected(tmp_path: Path) -> None:
    """Two meanings for one score is a contradiction the judge would resolve arbitrarily."""
    profile = _valid_profile()
    profile["scoring"]["bands"] = [
        {"low": 0, "high": 60, "meaning": "no"},
        {"low": 50, "high": 100, "meaning": "yes"},
    ]
    with pytest.raises(ConfigError, match="contiguous"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_bands_that_do_not_reach_the_ends_of_the_scale_are_rejected(tmp_path: Path) -> None:
    """The judge is asked for 0 to 100, so the bands must describe all of it."""
    profile = _valid_profile()
    profile["scoring"]["bands"] = [{"low": 10, "high": 90, "meaning": "middling"}]
    with pytest.raises(ConfigError, match="0 to 100"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_a_band_written_back_to_front_is_rejected(tmp_path: Path) -> None:
    """`low` above `high` would match no score at all."""
    profile = _valid_profile()
    profile["scoring"]["bands"] = [{"low": 100, "high": 0, "meaning": "everything"}]
    with pytest.raises(ConfigError, match="reversed"):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_one_band_covering_the_whole_scale_is_accepted(tmp_path: Path) -> None:
    """A blunt scale is a legitimate choice, so the check is not too strict."""
    profile = _valid_profile()
    profile["scoring"]["bands"] = [{"low": 0, "high": 100, "meaning": "how well it fits"}]
    profile["scoring"]["examples"] = []
    assert load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_an_example_scored_off_the_scale_is_rejected(tmp_path: Path) -> None:
    """A calibration example must name a score the judge could actually give."""
    profile = _valid_profile()
    profile["scoring"]["examples"] = [{"posting": "A job.", "score": 150, "reason": "why"}]
    with pytest.raises(ConfigError):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_out_of_range_threshold_is_rejected(tmp_path: Path) -> None:
    """A score threshold outside 0-100 is refused since nothing could reach it."""
    profile = _valid_profile()
    profile["scoring"]["threshold"] = 150
    with pytest.raises(ConfigError):
        load_search_profile(_write(tmp_path, "search_profile.yaml", profile))


def test_unknown_timezone_is_rejected(tmp_path: Path) -> None:
    """A misspelled timezone is refused and not left to fire the digest an hour off."""
    system = _valid_system()
    system["timezone"] = "Europe/Lisboa"
    with pytest.raises(ConfigError, match="Lisboa"):
        load_system_config(_write(tmp_path, "system_config.yaml", system))


def test_malformed_schedule_is_rejected(tmp_path: Path) -> None:
    """A schedule no scheduler could parse is refused at load time."""
    system = _valid_system()
    system["schedules"]["digest"] = "every morning"
    with pytest.raises(ConfigError):
        load_system_config(_write(tmp_path, "system_config.yaml", system))


@pytest.mark.parametrize(
    "spec", ["every 2h", "every 30m", "every 1d", "daily 08:00", "weekly sun 03:00"]
)
def test_accepted_schedule_forms(tmp_path: Path, spec: str) -> None:
    """All five legal schedule formats still load, so the check is not too strict."""
    system = _valid_system()
    system["schedules"]["digest"] = spec
    assert load_system_config(_write(tmp_path, "system_config.yaml", system))


def test_suppress_after_below_demote_after_is_rejected(tmp_path: Path) -> None:
    """Thresholds that would hide a job before it was ever demoted are refused."""
    system = _valid_system()
    system["digest"]["repeat_suppression"]["suppress_after"] = 1
    system["digest"]["repeat_suppression"]["demote_after"] = 3
    with pytest.raises(ConfigError, match="demote_after"):
        load_system_config(_write(tmp_path, "system_config.yaml", system))


def test_greenhouse_entry_without_a_token_is_rejected(tmp_path: Path) -> None:
    """A Greenhouse company with no board token is refused."""
    entries = [{"slug": "acme", "name": "Acme", "ats": "greenhouse", "tier": "lab"}]
    with pytest.raises(ConfigError, match="token"):
        load_watchlist(_write(tmp_path, "companies_watchlist.yaml", entries))


def test_workday_entry_requires_tenant_and_site(tmp_path: Path) -> None:
    """A Workday company is refused unless tenant, wd and site are all present."""
    entries = [
        {"slug": "nvidia", "name": "NVIDIA", "ats": "workday", "tier": "bigtech",
         "ats_config": {"tenant": "nvidia"}}
    ]
    with pytest.raises(ConfigError, match="site"):
        load_watchlist(_write(tmp_path, "companies_watchlist.yaml", entries))


def test_workday_entry_with_full_config_is_accepted(tmp_path: Path) -> None:
    """A fully configured Workday company loads, so the check is not too strict."""
    entries = [
        {"slug": "nvidia", "name": "NVIDIA", "ats": "workday", "tier": "bigtech",
         "ats_config": {"tenant": "nvidia", "wd": "wd5",
                        "site": "NVIDIAExternalCareerSite"}}
    ]
    (entry,) = load_watchlist(_write(tmp_path, "companies_watchlist.yaml", entries))
    assert entry.ats_config["site"] == "NVIDIAExternalCareerSite"


def test_duplicate_slug_is_rejected(tmp_path: Path) -> None:
    """The same company slug listed twice is refused."""
    entries = [
        {"slug": "acme", "name": "Acme", "ats": "ashby", "token": "acme"},
        {"slug": "acme", "name": "Acme Again", "ats": "lever", "token": "acme"},
    ]
    with pytest.raises(ConfigError, match="duplicate"):
        load_watchlist(_write(tmp_path, "companies_watchlist.yaml", entries))


def test_bad_entry_error_names_its_index(tmp_path: Path) -> None:
    """A bad entry is reported by its position, which matters in a 28-line file."""
    entries = [
        {"slug": "anthropic", "name": "Anthropic", "ats": "greenhouse",
         "token": "anthropic"},
        {"slug": "openai", "name": "OpenAI", "ats": "ashbyy", "token": "openai"},
    ]
    with pytest.raises(ConfigError, match=r"\b1\b"):
        load_watchlist(_write(tmp_path, "companies_watchlist.yaml", entries))


def test_missing_file_reports_its_path(tmp_path: Path) -> None:
    """A missing config file gives a readable error naming the path and not a crash."""
    with pytest.raises(ConfigError, match="missing config file"):
        load_search_profile(tmp_path / "nope.yaml")


def test_invalid_yaml_reports_the_file(tmp_path: Path) -> None:
    """Malformed YAML gives an error naming the file that failed to parse."""
    target = tmp_path / "search_profile.yaml"
    target.write_text("titles: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_search_profile(target)


def test_secrets_are_read_from_a_dotenv_file_in_the_example_format(tmp_path: Path, no_secret_env) -> None:
    """`KEY = value` with spaces, an empty value and a comment all parse as `.env.example` writes them."""
    env = tmp_path / ".env"
    env.write_text("# keys\nANTHROPIC_API_KEY = sk-ant-test\nSMTP_PASSWORD =\n", encoding="utf-8")

    secrets = load_secrets(env)

    assert secrets.require("anthropic_api_key") == "sk-ant-test"
    assert secrets.present() == {
        "ANTHROPIC_API_KEY": True, "SMTP_PASSWORD": False, "ACTION_TOKEN_SECRET": False,
        "DIGEST_TO": False, "DIGEST_FROM": False, "SMTP_USERNAME": False,
    }


def test_a_missing_secret_is_refused_by_name(tmp_path: Path, no_secret_env) -> None:
    """The error names the variable to set, which a 401 from the API never would."""
    secrets = load_secrets(tmp_path / "absent.env")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        secrets.require("anthropic_api_key")


def test_the_environment_wins_over_the_dotenv_file(tmp_path: Path, monkeypatch) -> None:
    """Inside a container Compose exports the variables, and those must beat any stray file."""
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert load_secrets(env).require("anthropic_api_key") == "from-env"


def test_addresses_are_read_from_the_dotenv_file_and_not_from_yaml(
    tmp_path: Path, no_secret_env, monkeypatch
) -> None:
    """Who the digest goes to identifies a person, so it never sits in a tracked file."""
    monkeypatch.delenv("DIGEST_TO", raising=False)
    env = tmp_path / ".env"
    env.write_text("DIGEST_TO = me@example.com\nSMTP_USERNAME = me@example.com\n", encoding="utf-8")

    secrets = load_secrets(env)

    assert secrets.require("digest_to") == "me@example.com"
    assert "to" not in load_system_config().email.model_fields_set
    assert not hasattr(load_system_config().email, "to"), "the address moved out of YAML"


def test_a_malformed_address_is_rejected_without_echoing_it(
    tmp_path: Path, no_secret_env, monkeypatch
) -> None:
    """A typo that could never deliver is refused, and the error repeats no address."""
    monkeypatch.delenv("DIGEST_TO", raising=False)
    env = tmp_path / ".env"
    env.write_text("DIGEST_TO = not-an-address\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_secrets(env)
    assert "digest_to" in str(exc.value).lower(), "the error names the variable"
    assert "not-an-address" not in str(exc.value), "but never the address itself"


def test_secret_values_never_appear_when_printed(tmp_path: Path, no_secret_env) -> None:
    """Printing the settings, or the presence summary, cannot leak a key into a log."""
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-hidden\n", encoding="utf-8")

    secrets = load_secrets(env)

    assert "sk-ant-hidden" not in repr(secrets)
    assert "sk-ant-hidden" not in str(secrets)
    assert "sk-ant-hidden" not in str(secrets.present())
    assert "sk-ant-hidden" not in str(secrets.summary())


def test_the_summary_shows_addresses_but_masks_credentials(tmp_path: Path, no_secret_env) -> None:
    """An address is shown so a typo in it can be spotted. A key has no such use."""
    env = tmp_path / ".env"
    env.write_text(
        "ANTHROPIC_API_KEY=sk-ant-hidden\nDIGEST_TO=me@example.com\n", encoding="utf-8"
    )

    summary = load_secrets(env).summary()

    assert summary["DIGEST_TO"] == "me@example.com"
    assert summary["ANTHROPIC_API_KEY"] == "present"
    assert summary["SMTP_PASSWORD"] == "missing"


def test_load_all_includes_the_secrets() -> None:
    """One call validates the YAML files and reads the secrets together."""
    assert isinstance(load_all().secrets, Secrets)
