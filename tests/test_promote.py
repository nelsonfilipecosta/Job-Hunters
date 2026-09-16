"""Tests for the review queue and the two decisions that empty it.

`approve` writes `companies_watchlist.yaml`, so every test here works on a
copy of the real file in a temporary directory and checks that the copy still
loads afterwards.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from job_hunters import paths
from job_hunters.config import CompanyEntry, load_watchlist
from job_hunters.tables import CandidateCompany, CandidateStatus
from job_hunters.promote import (
    DISCOVERED_HEADER,
    PromoteError,
    approve,
    find_candidate,
    reject,
    review_queue,
    slug_for,
    watchlist_line,
)

NOW = datetime(2026, 9, 14, 9, 0, tzinfo=UTC)


@pytest.fixture
def watchlist(tmp_path: Path) -> Path:
    """A copy of the repository's watchlist that a test may append to."""
    target = tmp_path / "companies_watchlist.yaml"
    shutil.copy(paths.WATCHLIST_PATH, target)
    return target


def _candidate(session: Session, name: str, *, ats: str | None = "ashby", token: str | None = None,
               jobs: int = 5, sightings: int = 1, status: str = CandidateStatus.PENDING) -> CandidateCompany:
    """A queued company with a board unless `ats` is None."""
    from job_hunters.normalize import normalize_company

    row = CandidateCompany(
        name=name, name_key=normalize_company(name), status=status, sightings=sightings,
        ats_type=ats, ats_token=(token or slug_for(name)) if ats else None,
        board_url=f"https://{ats}.test" if ats else None, board_jobs=jobs if ats else None,
        roles=["Research Scientist"], evidence=[{"source": "hn", "source_job_id": "1",
                                                 "title": name, "url": "u", "seen": NOW.isoformat()}],
        first_seen=NOW, last_seen=NOW,
    )
    session.add(row)
    session.commit()
    return row


def _watched(watchlist: Path) -> list[CompanyEntry]:
    """What the copied file watches."""
    return load_watchlist(watchlist)


def test_the_queue_lists_boards_first_and_the_most_seen_before_the_rest(session: Session, watchlist: Path) -> None:
    """A company with nothing to watch yet sits below every one that can be approved today."""
    _candidate(session, "Tufalabs", ats=None, sightings=9)
    _candidate(session, "Mechanize", sightings=1)
    _candidate(session, "Prior Labs", sightings=3)
    _candidate(session, "Rejected Co", status=CandidateStatus.REJECTED)
    queue = review_queue(session, _watched(watchlist), NOW)
    assert [c.name for c in queue] == ["Prior Labs", "Mechanize", "Tufalabs"]


def test_the_queue_resolves_a_company_the_file_now_lists(session: Session, watchlist: Path) -> None:
    """A line added by hand takes the candidate out of the queue as approved."""
    _candidate(session, "Prior Labs", token="prior-labs")
    with watchlist.open("a") as handle:
        handle.write("- { slug: prior-labs, name: Prior Labs, ats: ashby, token: prior-labs, tier: discovered }\n")
    assert review_queue(session, load_watchlist(watchlist), NOW) == []
    row = session.scalar(select(CandidateCompany))
    assert row.status == CandidateStatus.APPROVED and row.slug == "prior-labs"


def test_a_candidate_is_found_by_id_or_by_name_and_a_stranger_is_an_error(session: Session) -> None:
    """The queue prints ids, but a name typed the way it was seen works too."""
    row = _candidate(session, "Prior Labs")
    assert find_candidate(session, str(row.id)) is row
    assert find_candidate(session, "prior labs") is row
    with pytest.raises(PromoteError, match="No candidate"):
        find_candidate(session, "Nobody")


def test_approving_appends_one_valid_line_and_marks_the_candidate(session: Session, watchlist: Path) -> None:
    """The file gains one entry under a header and the row records the slug."""
    before = watchlist.read_text()
    row = _candidate(session, "Prior Labs", token="prior-labs", jobs=24)
    done = approve(session, str(row.id), watchlist_path=watchlist, now=NOW)

    after = watchlist.read_text()
    assert after.startswith(before)
    assert after.count(DISCOVERED_HEADER) == 1
    assert done.line in after and done.slug == "prior-labs"
    entries = load_watchlist(watchlist)
    assert entries[-1].slug == "prior-labs" and entries[-1].ats_config == {"token": "prior-labs"}
    assert entries[-1].tier == "discovered"
    assert row.status == CandidateStatus.APPROVED and row.slug == "prior-labs"
    assert row.decided_at == NOW


def test_a_second_approval_shares_the_header(session: Session, watchlist: Path) -> None:
    """The header is written once above the first promoted entry."""
    approve(session, str(_candidate(session, "Prior Labs").id), watchlist_path=watchlist)
    approve(session, str(_candidate(session, "Mechanize").id), watchlist_path=watchlist)
    text = watchlist.read_text()
    assert text.count(DISCOVERED_HEADER) == 1
    assert [e.slug for e in load_watchlist(watchlist)[-2:]] == ["prior-labs", "mechanize"]


def test_the_slug_and_name_can_be_chosen_and_the_tier_too(session: Session, watchlist: Path) -> None:
    """What the model read is a default and not a verdict."""
    row = _candidate(session, "Artificial Intelligence Underwriting Company", token="aiuc")
    done = approve(session, str(row.id), slug="aiuc", name="AIUC", tier="lab", watchlist_path=watchlist)
    entry = load_watchlist(watchlist)[-1]
    assert (entry.slug, entry.name, entry.tier.value) == ("aiuc", "AIUC", "lab")
    assert done.slug == "aiuc"


def test_a_name_that_would_break_the_flow_style_is_quoted(session: Session, watchlist: Path) -> None:
    """A colon or a brace in a name is quoted so the file keeps loading."""
    row = _candidate(session, "Acme: Labs {AI}", token="acme")
    approve(session, str(row.id), slug="acme-labs", watchlist_path=watchlist)
    assert load_watchlist(watchlist)[-1].name == "Acme: Labs {AI}"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Prior Labs", "prior-labs"), ("A.Team", "a-team"), ("exe.dev", "exe-dev"), ("  Scale AI  ", "scale-ai")],
)
def test_a_slug_is_derived_from_the_name(name: str, expected: str) -> None:
    """Lowercase and hyphens for anything that is not a letter or digit."""
    assert slug_for(name) == expected


def test_a_name_with_no_letters_needs_a_slug_passed(session: Session) -> None:
    """There is nothing to derive from so the caller has to say."""
    with pytest.raises(PromoteError, match="--slug"):
        slug_for("!!!")


def test_watchlist_line_is_bare_when_it_can_be() -> None:
    """The file is hand-written in bare flow style and promoted lines should read the same."""
    line = watchlist_line("prior-labs", "Prior Labs", "ashby", "prior-labs", "discovered")
    assert line == "- { slug: prior-labs, name: Prior Labs, ats: ashby, token: prior-labs, tier: discovered }"
    assert 'name: "Acme: Inc"' in watchlist_line("acme", "Acme: Inc", "ashby", "acme", "discovered")
    # Without a tier the line ends at the token and loads with the default one.
    assert watchlist_line("prior-labs", "Prior Labs", "ashby", "prior-labs", None) == (
        "- { slug: prior-labs, name: Prior Labs, ats: ashby, token: prior-labs }"
    )


@pytest.mark.parametrize("name", ["42", "Yes", "null", "1Password", 'Say "hi"', "Züri Lab"])
def test_a_name_yaml_would_misread_is_quoted_and_survives_the_round_trip(name: str) -> None:
    """Numbers, booleans and non-ASCII names come back as the same string."""
    import yaml

    line = watchlist_line("x", name, "ashby", "x", "discovered")
    assert yaml.safe_load(line)[0]["name"] == name


def test_approving_without_a_board_is_refused_with_the_way_forward(session: Session, watchlist: Path) -> None:
    """Nothing can be watched without a board so the message points at `probe`."""
    row = _candidate(session, "Tufalabs", ats=None)
    before = watchlist.read_text()
    with pytest.raises(PromoteError, match="job-hunters probe"):
        approve(session, str(row.id), watchlist_path=watchlist)
    assert watchlist.read_text() == before and row.status == CandidateStatus.PENDING


def test_approving_twice_is_refused(session: Session, watchlist: Path) -> None:
    """The second approval would write a duplicate slug so it never gets that far."""
    row = _candidate(session, "Prior Labs")
    approve(session, str(row.id), watchlist_path=watchlist)
    with pytest.raises(PromoteError, match="already"):
        approve(session, str(row.id), watchlist_path=watchlist)


def test_a_slug_or_board_the_file_already_has_is_refused(session: Session, watchlist: Path) -> None:
    """Two entries with one slug would not load and two with one board would fetch it twice."""
    before = watchlist.read_text()
    taken = _candidate(session, "Anthropic Labs", ats="greenhouse", token="anthropic-labs")
    with pytest.raises(PromoteError, match="slug 'anthropic'"):
        approve(session, str(taken.id), slug="anthropic", watchlist_path=watchlist)
    same_board = _candidate(session, "Anthropic PBC", ats="greenhouse", token="anthropic")
    with pytest.raises(PromoteError, match="already watches"):
        approve(session, str(same_board.id), watchlist_path=watchlist)
    assert watchlist.read_text() == before
    assert taken.status == same_board.status == CandidateStatus.PENDING


def test_a_read_only_file_leaves_the_candidate_pending_and_prints_the_line(
    session: Session, watchlist: Path
) -> None:
    """Inside the container `config/` is read-only: the line is shown and nothing pretends it was written."""
    row = _candidate(session, "Prior Labs", token="prior-labs")
    watchlist.chmod(0o444)
    if os.access(watchlist, os.W_OK):  # pragma: no cover - root ignores file modes
        pytest.skip("this user can write to a read-only file")
    try:
        with pytest.raises(PromoteError) as exc:
            approve(session, str(row.id), watchlist_path=watchlist)
    finally:
        watchlist.chmod(0o644)
    assert "read-only" in str(exc.value) and "slug: prior-labs" in str(exc.value)
    assert row.status == CandidateStatus.PENDING and row.slug is None
    assert DISCOVERED_HEADER not in watchlist.read_text()


def test_rejecting_silences_a_candidate_and_keeps_its_row(session: Session, watchlist: Path) -> None:
    """The row stays so the next sighting counts against it instead of queueing it afresh."""
    row = _candidate(session, "Cascade Space", ats=None)
    rejected = reject(session, "Cascade Space", NOW)
    assert rejected is row and row.status == CandidateStatus.REJECTED and row.decided_at == NOW
    assert review_queue(session, _watched(watchlist), NOW) == []


def test_an_approved_company_cannot_be_rejected_from_here(session: Session, watchlist: Path) -> None:
    """Stopping watching a company is an edit to the file and not a decision in the queue."""
    row = _candidate(session, "Prior Labs")
    approve(session, str(row.id), watchlist_path=watchlist)
    with pytest.raises(PromoteError, match="watchlist"):
        reject(session, str(row.id))
    assert row.status == CandidateStatus.APPROVED
