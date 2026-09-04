"""Tests for turning raw board data into clean and comparable fields.

The location cases come from `tests/fixtures/locations.yaml`, a survey of real
strings seen on watched boards. Adding a string that the parser gets wrong is
a one-entry change there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from job_hunters import regions
from job_hunters.normalize import (
    content_hash,
    html_to_text,
    normalize_company,
    normalize_title,
    parse_location,
    raw_hash,
)

FIXTURE = Path(__file__).parent / "fixtures" / "locations.yaml"
CASES = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def _case_id(case: dict) -> str:
    """A readable pytest id for one location fixture case."""
    hints = case.get("hints") or {}
    suffix = f" +{hints}" if hints else ""
    return f"{case['raw']!r}{suffix}"


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_real_location_strings_resolve_as_expected(case: dict) -> None:
    """Every surveyed string resolves to the regions and work mode a human would."""
    hints = case.get("hints") or {}
    parsed = parse_location(
        case["raw"],
        country_code=hints.get("country_code"),
        workplace_type=hints.get("workplace_type"),
        is_remote=hints.get("is_remote"),
        hints=tuple(hints.get("extra") or ()),
    )
    assert list(parsed.regions) == case["regions"], parsed
    assert parsed.region == case["region"], parsed
    assert parsed.work_mode == case["work_mode"], parsed


def test_every_region_the_parser_emits_is_in_the_vocabulary() -> None:
    """A region token nothing else understands would silently break the digest."""
    for case in CASES:
        for token in case["regions"]:
            assert token in regions.COUNTRIES, token
        assert case["region"] in regions.COUNTRIES | {regions.OTHER, regions.UNKNOWN}


def test_eu_is_never_a_region_on_a_job() -> None:
    """`eu` is a config group, but a job always resolves to a country."""
    assert parse_location("Europe").region == regions.UNKNOWN
    assert "eu" not in parse_location("Berlin, Germany | Lisbon").regions


def test_unknown_is_the_answer_to_nonsense() -> None:
    """Text that resolves to nothing is `unknown`."""
    parsed = parse_location("asdfgh qwerty")
    assert parsed == parse_location("asdfgh qwerty")
    assert parsed.region == regions.UNKNOWN
    assert parsed.regions == ()


# --- html --------------------------------------------------------------------


def test_html_to_text_strips_tags_and_keeps_structure() -> None:
    """Tags disappear, entities decode and block order is preserved."""
    markup = "<h2>Summary</h2><p>We&#39;re hiring &amp; growing.</p><ul><li>One</li><li>Two</li></ul>"
    text = html_to_text(markup)
    assert "<" not in text and "&" not in text.replace("& growing", "")
    assert "We're hiring & growing." in text
    assert "- One" in text and "- Two" in text
    assert text.index("Summary") < text.index("We're")


def test_html_to_text_drops_scripts_and_collapses_whitespace() -> None:
    """Script contents are dropped and repeated whitespace collapses to one."""
    markup = "<p>a</p>\n\n\n<script>alert(1)</script><p>   b   c </p>"
    assert html_to_text(markup) == "a\n\nb c"


def test_html_to_text_handles_greenhouse_double_encoding_after_unescape() -> None:
    """Greenhouse sends escaped HTML, so the adapter unescapes once and then this runs."""
    import html as html_module

    escaped = "&lt;p&gt;Alignment &amp;amp; evals&lt;/p&gt;"
    assert html_to_text(html_module.unescape(escaped)) == "Alignment & evals"


def test_html_to_text_of_nothing_is_empty() -> None:
    """None and the empty string both produce an empty string."""
    assert html_to_text(None) == "" and html_to_text("") == ""


# --- titles ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Research Scientist, Post-Training", "Research Scientist - Post Training"),
        ("Member of Technical Staff (ML)", "Member of Technical Staff - ML"),
        ("Research Scientist (Remote)", "Research Scientist"),
        ("Research Scientist - New York", "Research Scientist"),
        ("Research Scientist | Acme", "Research Scientist"),
        ("Research Scientist (Req 12345)", "Research Scientist"),
        ("Research Scientist #48210", "Research Scientist"),
        ("Research Scientist (f/m/d)", "Research Scientist"),
        ("  Research   Scientist  ", "Research Scientist"),
    ],
)
def test_titles_that_are_the_same_role_normalize_identically(a: str, b: str) -> None:
    """Cosmetic differences in a title normalize to the same string."""
    assert normalize_title(a, "Acme") == normalize_title(b, "Acme")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Senior Research Scientist", "Research Scientist"),
        ("Staff Research Scientist", "Research Scientist"),
        ("Research Scientist, Post-Training", "Research Scientist, Pre-Training"),
        ("Research Scientist", "Research Engineer"),
        ("Software Engineer, Inference", "Software Engineer, Training"),
        ("Research Scientist - Post Training", "Research Scientist"),
    ],
)
def test_titles_that_are_different_roles_stay_different(a: str, b: str) -> None:
    """The noise stripping must never eat the words that make a role distinct."""
    assert normalize_title(a) != normalize_title(b)


def test_company_prefix_is_stripped_only_when_it_is_the_company() -> None:
    """A leading company-name prefix is dropped only when it matches this company."""
    assert normalize_title("Acme | Research Scientist", "Acme") == "research scientist"
    assert normalize_title("Acme | Research Scientist", "Other") == "acme research scientist"


def test_normalize_company() -> None:
    """A company name loses punctuation and legal suffixes."""
    assert normalize_company("Liquid AI, Inc.") == "liquid ai inc"


# --- hashes ------------------------------------------------------------------


def test_content_hash_ignores_whitespace_but_not_words() -> None:
    """Whitespace differences don't change the hash, but a retitle does."""
    a = content_hash("Acme", "Research Scientist", "Lisbon", "We   do\n\nRLHF.")
    b = content_hash("Acme", "Research Scientist", "Lisbon", "We do RLHF.")
    c = content_hash("Acme", "Senior Research Scientist", "Lisbon", "We do RLHF.")
    assert a == b
    assert a != c, "a retitled role must be judged again"


def test_raw_hash_is_order_independent() -> None:
    """Two payloads with the same keys in a different order hash the same."""
    assert raw_hash({"a": 1, "b": [1, 2]}) == raw_hash({"b": [1, 2], "a": 1})
    assert raw_hash({"a": 1}) != raw_hash({"a": 2})
