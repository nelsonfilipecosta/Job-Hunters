"""The region vocabulary.

A job's parsed location is stored as a single *country* token. Configuration,
however, is more convenient to write in *groups* - "anywhere in the EU" beats
listing 27 countries. So `search_profile.yaml` may name either and matching
expands groups to their members before comparing.

To support a country not listed here, add it to ``COUNTRIES``. Config
validation rejects unknown tokens, so a typo fails loudly at startup instead
of silently matching nothing.
"""

from __future__ import annotations

# Country-level tokens. `jobs.region` always holds one of these.
COUNTRIES: frozenset[str] = frozenset(
    {
        # EU
        "austria", "belgium", "bulgaria", "croatia", "cyprus", "czechia",
        "denmark", "estonia", "finland", "france", "germany", "greece",
        "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg",
        "malta", "netherlands", "poland", "portugal", "romania", "slovakia",
        "slovenia", "spain", "sweden",
        # Europe (non-EU)
        "switzerland", "uk", "norway", "iceland",
        # Elsewhere
        "us", "canada", "australia", "new zealand",
    }
)

# Group tokens usable in config. Values must all be members of COUNTRIES.
REGION_GROUPS: dict[str, frozenset[str]] = {
    "eu": frozenset(
        {
            "austria", "belgium", "bulgaria", "croatia", "cyprus", "czechia",
            "denmark", "estonia", "finland", "france", "germany", "greece",
            "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg",
            "malta", "netherlands", "poland", "portugal", "romania", "slovakia",
            "slovenia", "spain", "sweden",
        }
    ),
    # The wider free-movement area a Portuguese citizen may work in without a
    # permit process.
    "eea_efta": frozenset({"switzerland", "norway", "iceland"}),
}

# Emitted when a location parsed confidently to a real country that is not in
# COUNTRIES (Tokyo, Singapore, Dubai).
OTHER = "other"

# Emitted when a location string cannot be parsed confidently. Routes to the
# "Worth checking" digest section rather than being silently dropped.
UNKNOWN = "unknown"

# Every token config may legally contain.
VALID_REGION_TOKENS: frozenset[str] = (
    COUNTRIES | frozenset(REGION_GROUPS) | frozenset({UNKNOWN})
)


def expand(tokens: list[str] | frozenset[str]) -> frozenset[str]:
    """Expand group tokens to country tokens and pass country tokens through.

    >>> "portugal" in expand(["eu"])
    True
    >>> "switzerland" in expand(["eu"])
    False
    """
    result: set[str] = set()
    for token in tokens:
        result |= REGION_GROUPS.get(token, frozenset({token}))
    return frozenset(result)


def unknown_tokens(tokens: list[str] | frozenset[str]) -> list[str]:
    """Return the tokens that are not part of the vocabulary (for error messages)."""
    return sorted(set(tokens) - VALID_REGION_TOKENS)
