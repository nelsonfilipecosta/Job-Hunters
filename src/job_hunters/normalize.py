"""Turns raw postings into the clean and comparable form the rest of the system reads.

    html_to_text     Greenhouse and Lever return HTML. The judge and the content
                     hash need text.
    parse_location   Free text into country tokens plus a work mode. The digest's
                     Priority / Remote / Worth-checking split depends on it being
                     right, so when it is unsure it says `unknown` rather than
                     guessing and hiding a good job.
    normalize_title  The text deduplication compares. Strips the noise that makes
                     one role look like two.

The lookup tables below are deliberately small and explicit rather than a
geocoding library. Every entry is a string that actually appeared on a board
in the watchlist and adding one is a one-line change with a test.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from . import regions
from .models import WorkMode

# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

_BLOCK_TAGS = frozenset(
    "p div br li ul ol h1 h2 h3 h4 h5 h6 tr table thead tbody section article "
    "header footer blockquote pre hr dd dt".split()
)
_SKIP_TAGS = frozenset({"script", "style"})


class _TextExtractor(HTMLParser):
    """Collects text, turning block elements into line breaks and <li> into bullets."""

    def __init__(self) -> None:
        """Starts with an empty buffer and nothing being skipped."""
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Opens a skipped region or turns a block tag into a break or a bullet."""
        if tag in _SKIP_TAGS:
            self._skipping += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        """Closes a skipped region or ends a block with a line break."""
        if tag in _SKIP_TAGS:
            self._skipping = max(0, self._skipping - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Keeps the text between tags, unless it sits inside a skipped tag."""
        if not self._skipping:
            self.parts.append(data)


def html_to_text(markup: str | None) -> str:
    """Strips tags and entities and keeps paragraph and list structure as newlines.

    Expects real HTML. Greenhouse's escaped HTML must be `html.unescape`d by its adapter first.
    """
    if not markup:
        return ""
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):  # at most one blank line in a row
            out.append(line)
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# Location -> region + work mode
# ---------------------------------------------------------------------------

# ISO 3166-1 alpha-2 -> region token. Lever supplies these directly.
_ISO2: dict[str, str] = {
    "US": "us", "GB": "uk", "CH": "switzerland", "CA": "canada", "PT": "portugal",
    "DE": "germany", "FR": "france", "NL": "netherlands", "IE": "ireland",
    "ES": "spain", "IT": "italy", "SE": "sweden", "DK": "denmark", "NO": "norway",
    "PL": "poland", "LT": "lithuania", "AT": "austria", "BE": "belgium",
    "FI": "finland", "CZ": "czechia", "GR": "greece", "HU": "hungary",
    "LU": "luxembourg", "RO": "romania", "SK": "slovakia", "SI": "slovenia",
    "BG": "bulgaria", "HR": "croatia", "CY": "cyprus", "EE": "estonia",
    "LV": "latvia", "MT": "malta", "IS": "iceland", "AU": "australia",
    "NZ": "new zealand",
}

_COUNTRY_NAMES: dict[str, str] = {
    "united states": "us", "united states of america": "us", "usa": "us",
    "u.s.": "us", "u.s.a.": "us", "us": "us", "america": "us",
    "united kingdom": "uk", "uk": "uk", "u.k.": "uk", "great britain": "uk",
    "britain": "uk", "england": "uk", "scotland": "uk", "wales": "uk",
    "northern ireland": "uk",
    "switzerland": "switzerland", "schweiz": "switzerland", "suisse": "switzerland",
    "svizzera": "switzerland", "ch": "switzerland",
    "canada": "canada", "can": "canada",
    "portugal": "portugal", "pt": "portugal",
    "germany": "germany", "deutschland": "germany", "de": "germany",
    "france": "france", "fr": "france",
    "netherlands": "netherlands", "the netherlands": "netherlands", "holland": "netherlands",
    "nl": "netherlands",
    "ireland": "ireland", "ie": "ireland",
    "spain": "spain", "españa": "spain", "es": "spain",
    "italy": "italy", "italia": "italy", "it": "italy",
    "sweden": "sweden", "se": "sweden",
    "denmark": "denmark", "dk": "denmark",
    "norway": "norway", "no": "norway",
    "poland": "poland", "pl": "poland",
    "lithuania": "lithuania", "austria": "austria", "belgium": "belgium",
    "finland": "finland", "czechia": "czechia", "czech republic": "czechia",
    "greece": "greece", "hungary": "hungary", "luxembourg": "luxembourg",
    "romania": "romania", "slovakia": "slovakia", "slovenia": "slovenia",
    "bulgaria": "bulgaria", "croatia": "croatia", "cyprus": "cyprus",
    "estonia": "estonia", "latvia": "latvia", "malta": "malta", "iceland": "iceland",
    "australia": "australia", "au": "australia",
    "new zealand": "new zealand", "nz": "new zealand",
    # Countries outside the search
    "japan": regions.OTHER, "korea": regions.OTHER, "south korea": regions.OTHER,
    "india": regions.OTHER, "singapore": regions.OTHER, "israel": regions.OTHER,
    "uae": regions.OTHER, "united arab emirates": regions.OTHER,
    "saudi arabia": regions.OTHER, "mexico": regions.OTHER, "brazil": regions.OTHER,
    "colombia": regions.OTHER, "china": regions.OTHER, "hong kong": regions.OTHER,
    "taiwan": regions.OTHER, "argentina": regions.OTHER, "chile": regions.OTHER,
    "south africa": regions.OTHER, "nigeria": regions.OTHER, "kenya": regions.OTHER,
    "philippines": regions.OTHER, "indonesia": regions.OTHER, "vietnam": regions.OTHER,
    "thailand": regions.OTHER, "malaysia": regions.OTHER, "pakistan": regions.OTHER,
    "turkey": regions.OTHER, "egypt": regions.OTHER, "qatar": regions.OTHER,
    "serbia": regions.OTHER, "ukraine": regions.OTHER,
}

_US_STATES: dict[str, str] = {
    code.lower(): "us"
    for code in (
        "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
        "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
    ).split()
}
_US_STATES.update(
    {
        name: "us"
        for name in (
            "alabama alaska arizona arkansas california colorado connecticut delaware "
            "florida georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana "
            "maine maryland massachusetts michigan minnesota mississippi missouri montana "
            "nebraska nevada new hampshire|new jersey|new mexico|new york|north carolina|"
            "north dakota|ohio oklahoma oregon pennsylvania rhode island|south carolina|"
            "south dakota|tennessee texas utah vermont virginia washington west virginia|"
            "wisconsin wyoming d.c.|washington d.c.|washington, d.c."
        ).replace("|", " ").split()
        if " " not in name
    }
)
_US_STATES.update(
    {n: "us" for n in ("new hampshire", "new jersey", "new mexico", "new york",
                       "north carolina", "north dakota", "rhode island", "south carolina",
                       "south dakota", "west virginia", "d.c.", "washington d.c.",
                       "washington, d.c.", "dc")}
)

_CA_PROVINCES: dict[str, str] = {
    n: "canada" for n in ("on", "qc", "bc", "ab", "mb", "sk", "ns", "nb", "nl", "pe",
                          "ontario", "quebec", "québec", "british columbia", "alberta",
                          "manitoba", "saskatchewan", "nova scotia", "new brunswick")
}

_CITIES: dict[str, str] = {
    # Portugal
    "lisbon": "portugal", "lisboa": "portugal", "porto": "portugal", "coimbra": "portugal",
    "braga": "portugal", "aveiro": "portugal",
    # Switzerland
    "zurich": "switzerland", "zürich": "switzerland", "zuerich": "switzerland",
    "geneva": "switzerland", "genève": "switzerland", "geneve": "switzerland",
    "lausanne": "switzerland", "bern": "switzerland", "basel": "switzerland",
    "lugano": "switzerland",
    # US
    "san francisco": "us", "sf": "us", "san francisco bay area": "us", "bay area": "us",
    "new york": "us", "new york city": "us", "nyc": "us", "seattle": "us",
    "palo alto": "us", "mountain view": "us", "menlo park": "us", "sunnyvale": "us",
    "san jose": "us", "boston": "us", "cambridge, ma": "us", "austin": "us",
    "washington": "us", "washington dc": "us", "chicago": "us", "denver": "us", "miami": "us",
    "houston": "us", "los angeles": "us", "la": "us", "memphis": "us",
    "southaven": "us", "bastrop": "us", "honolulu": "us", "san diego": "us",
    "colorado springs": "us", "fayetteville": "us", "huntsville": "us",
    "pittsburgh": "us", "atlanta": "us", "philadelphia": "us", "portland": "us",
    # UK / Ireland
    "london": "uk", "cambridge": "uk", "oxford": "uk", "edinburgh": "uk",
    "manchester": "uk", "dublin": "ireland",
    # EU
    "paris": "france", "berlin": "germany", "munich": "germany", "münchen": "germany",
    "hamburg": "germany", "frankfurt": "germany", "amsterdam": "netherlands",
    "madrid": "spain", "barcelona": "spain", "milan": "italy", "milano": "italy",
    "rome": "italy", "stockholm": "sweden", "copenhagen": "denmark", "oslo": "norway",
    "warsaw": "poland", "vilnius": "lithuania", "helsinki": "finland",
    "brussels": "belgium", "vienna": "austria", "prague": "czechia", "athens": "greece",
    "budapest": "hungary", "tallinn": "estonia", "riga": "latvia",
    # Canada
    "toronto": "canada", "ottawa": "canada", "montreal": "canada", "montréal": "canada",
    "vancouver": "canada", "waterloo": "canada", "calgary": "canada", "edmonton": "canada",
    # Australia
    "sydney": "australia", "melbourne": "australia",
    # Cities outside the search
    "tokyo": regions.OTHER, "seoul": regions.OTHER, "singapore": regions.OTHER,
    "bangalore": regions.OTHER, "bengaluru": regions.OTHER, "delhi": regions.OTHER,
    "new delhi": regions.OTHER, "mumbai": regions.OTHER, "pune": regions.OTHER,
    "hyderabad": regions.OTHER, "dubai": regions.OTHER, "abu dhabi": regions.OTHER,
    "tel aviv": regions.OTHER, "são paulo": regions.OTHER, "sao paulo": regions.OTHER,
    "mexico city": regions.OTHER, "beijing": regions.OTHER, "shanghai": regions.OTHER,
    "shenzhen": regions.OTHER, "taipei": regions.OTHER, "riyadh": regions.OTHER,
    "bogota": regions.OTHER, "bogotá": regions.OTHER, "buenos aires": regions.OTHER,
    "belgrade": regions.OTHER, "kyiv": regions.OTHER,
}

# Words that describe how and not where. Dropped before resolving places.
_NOISE_WORDS = frozenset(
    "remote remote-friendly remotefriendly hybrid onsite on-site in-office travel "
    "required travel-required friendly international global worldwide anywhere "
    "flexible distributed multiple locations location various tbd or and".split()
)
# Regions that are real but not countries. Cannot resolve, so `unknown`.
_SUPRANATIONAL = frozenset(
    "europe emea apac americas north america latam latin america asia middle east "
    "nordics benelux dach".split()
)

_SPLIT_RE = re.compile(r"\s*(?:\||;|/|&|,|\bor\b|\band\b|\s-\s|–|—)\s*")
_GENDER_MARKER_RE = re.compile(r"^(?:[a-z]\s*/\s*){1,3}[a-z]$|^all genders$")
_REQ_ID_RE = re.compile(r"\b(?:req(?:uisition)?|job|id|jr|r)[\s#.-]*\d{3,}\b|#\d{3,}\b|\b\d{5,}\b", re.I)


@dataclass(frozen=True)
class ParsedLocation:
    """What a location string resolved to.

    `regions` holds every vocabulary country found in order of appearance.
    `region` is the summary the digest sorts on:
        - the first of `regions`;
        - `other` when the place was real but outside the search;
        - `unknown` otherwise.
    """

    regions: tuple[str, ...]
    region: str
    work_mode: str


def _resolve_place(token: str) -> str | None:
    """One cleaned token: region token, OTHER or None if unrecognised."""
    t = token.strip().strip(".").strip()
    if not t:
        return None
    for table in (_COUNTRY_NAMES, _CITIES, _US_STATES, _CA_PROVINCES):
        if t in table:
            return table[t]
    upper = t.upper()
    if len(upper) == 2 and upper.isalpha() and upper in _ISO2:
        return _ISO2[upper]
    return None


def _work_mode_from_hints(workplace_type: str | None, is_remote: bool | None) -> str | None:
    """The work mode a board stated outright or None if it stated nothing usable.

    `workplace_type` is trusted over `is_remote` because Ashby sets `isRemote`
    true on hybrid roles too, so the boolean alone would call them remote.
    """
    if workplace_type:
        w = workplace_type.strip().lower().replace("_", "").replace("-", "")
        if w == "remote":
            return WorkMode.REMOTE
        if w == "hybrid":
            return WorkMode.HYBRID
        if w in {"onsite", "office", "inoffice"}:
            return WorkMode.ONSITE
    if is_remote is True:
        return WorkMode.REMOTE
    if is_remote is False:
        return WorkMode.ONSITE
    return None


def parse_location(
    raw: str | None,
    *,
    country_code: str | None = None,
    workplace_type: str | None = None,
    is_remote: bool | None = None,
    hints: tuple[str, ...] | list[str] = (),
) -> ParsedLocation:
    """Resolves a board's location text, plus any structured hints, to regions.

    Most trusted first: an ISO country code (Lever), the board's own work-mode
    field (Lever, Ashby), the free-text string and any extra location strings.
    Text that resolves to nothing is reported as `unknown` rather than guessed.
    """
    found: list[str] = []
    saw_other = False
    saw_remote_word = False
    saw_hybrid_word = False
    resolved_anything = False

    def add(token: str) -> None:
        """Records one resolved place, keeping `other` out of the searched list."""
        nonlocal saw_other, resolved_anything
        if token == regions.OTHER:
            saw_other = True
            resolved_anything = True
        elif token not in found:
            found.append(token)
            resolved_anything = True

    if country_code:
        code = _ISO2.get(country_code.strip().upper())
        if code:
            add(code)
        elif len(country_code.strip()) == 2 and country_code.strip().isalpha():
            add(regions.OTHER)

    texts = [raw or "", *hints]
    for text in texts:
        cleaned = text.lower().replace("(", ",").replace(")", ",")
        for piece in _SPLIT_RE.split(cleaned):
            piece = piece.strip().strip(".").strip()
            if not piece:
                continue
            if "remote" in piece:
                saw_remote_word = True
            if "hybrid" in piece:
                saw_hybrid_word = True
            if piece in _NOISE_WORDS or piece in _SUPRANATIONAL:
                continue
            # "remote-friendly, united states" -> "united states"
            words = [w for w in piece.replace("-", " ").split() if w not in _NOISE_WORDS]
            candidate = " ".join(words).strip()
            token = _resolve_place(piece) or (_resolve_place(candidate) if candidate else None)
            if token:
                add(token)

    work_mode = _work_mode_from_hints(workplace_type, is_remote)
    if work_mode is None:
        if saw_remote_word:
            work_mode = WorkMode.REMOTE
        elif saw_hybrid_word:
            work_mode = WorkMode.HYBRID
        elif resolved_anything:
            work_mode = WorkMode.ONSITE
        else:
            work_mode = WorkMode.UNKNOWN

    if found:
        region = found[0]
    elif saw_other:
        region = regions.OTHER
    else:
        region = regions.UNKNOWN
    return ParsedLocation(regions=tuple(found), region=region, work_mode=work_mode)


# ---------------------------------------------------------------------------
# Titles, companies, hashes
# ---------------------------------------------------------------------------

_NOISE_FRAGMENT_WORDS = _NOISE_WORDS | frozenset(
    "contract contractor fulltime full-time full time parttime part-time temporary "
    "intern internship f/m/d m/f/d w/m/d all genders".split()
)


def _is_noise_fragment(fragment: str) -> bool:
    """True if a parenthesised or trailing fragment carries no role information."""
    f = fragment.strip().lower()
    if not f:
        return True
    if _REQ_ID_RE.search(f) or _GENDER_MARKER_RE.match(f):
        return True
    words = re.split(r"[\s,/&|-]+", f)
    if all(w in _NOISE_FRAGMENT_WORDS for w in words if w):
        return True
    if _resolve_place(f) is not None or all(_resolve_place(w) for w in words if w):
        return True
    return False


def normalize_title(title: str, company_name: str | None = None) -> str:
    """The comparable form of a title: lowercase, punctuation-free and noise stripped.

    Removes:
        - parenthesised fragments that only say where or how ("(Remote)" or "(Zurich)");
        - trailing " - X" / " | X" segments when X is such noise or the company's own name;
        - requisition ids.

    Keeps anything that describes the role itself. For instance, "Research Scientist - Post Training"
    must stay distinct from "Research Scientist - Pre Training".
    """
    t = title.strip()
    company = (company_name or "").strip().lower()

    # Parenthesised noise: "(Remote)", "(US)", "(Req 1234)".
    def _paren(match: re.Match) -> str:
        """Drops a bracketed fragment if it is noise, otherwise keeps its contents."""
        inner = match.group(1)
        return " " if _is_noise_fragment(inner) else f" {inner} "

    t = re.sub(r"\(([^)]*)\)", _paren, t)
    t = re.sub(r"\[([^\]]*)\]", _paren, t)

    # Trailing segments after " - ", " | ", " – ", ":" that are pure noise.
    changed = True
    while changed:
        changed = False
        m = re.search(r"\s+(?:-|–|—|\||:)\s+([^-–—|:]+)$", t)
        if m:
            tail = m.group(1)
            if _is_noise_fragment(tail) or (company and tail.strip().lower() == company):
                t = t[: m.start()]
                changed = True
    # Leading company prefix "Acme | Research Scientist".
    if company:
        t = re.sub(rf"^{re.escape(company)}\s*(?:-|–|—|\||:)\s*", "", t, flags=re.I)

    t = _REQ_ID_RE.sub(" ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_company(name: str) -> str:
    """Lowercase and punctuation-free company name for keys."""
    n = re.sub(r"[^a-z0-9]+", " ", name.lower())
    return re.sub(r"\s+", " ", n).strip()


def normalize_whitespace(text: str | None) -> str:
    """Collapses runs of whitespace. The stable form used inside content hashes."""
    return re.sub(r"\s+", " ", text or "").strip()


def sha256(*parts: str) -> str:
    """Hashes several strings as one, but separated so "ab"+"c" cannot collide with "a"+"bc"."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def raw_hash(raw: dict[str, Any]) -> str:
    """Hash of a board's untouched payload. It answers "did the source change?"."""
    return hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def content_hash(company_name: str, title: str, location_raw: str | None, description: str) -> str:
    """Hash of everything the judge reads over normalized text.

    Title, company and location are included on purpose: a role retitled from
    "Research Scientist" to "Senior Research Scientist, Post-Training" must be
    judged again even if the body did not change.
    """
    return sha256(
        normalize_company(company_name),
        normalize_whitespace(title).lower(),
        normalize_whitespace(location_raw).lower(),
        normalize_whitespace(description).lower(),
    )


# Guard against the vocabulary drifting away from these tables. Any token a
# table can emit must be a real region token. Otherwise the digest would sort on a
# value nothing else understands.
_emittable = set(_ISO2.values()) | set(_COUNTRY_NAMES.values()) | set(_CITIES.values()) \
    | set(_US_STATES.values()) | set(_CA_PROVINCES.values())
_bad = _emittable - regions.COUNTRIES - {regions.OTHER}
if _bad:  # pragma: no cover - a programming error, caught at import
    raise RuntimeError(f"`normalize.py` emits region tokens missing from regions.COUNTRIES: {sorted(_bad)}")
