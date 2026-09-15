"""Tests for the scan that turns the discovery sources into a queue of companies.

Nothing here opens a socket or calls the API. `FakeDiscoverySource` answers
from a script, `FakeAnthropic` names companies from a script and a fake prober
stands in for the HTTP request a real probe makes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import anthropic
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import FakeAnthropic, FakeDiscoverySource, api_error, failed, make_posting
from job_hunters.config import AppConfig, CompanyEntry, Secrets, SearchProfile, SystemConfig
from job_hunters.probe import Board
from job_hunters.discovery import DiscoveryReport, Watched, reconcile, run_discovery
from job_hunters.extract import Extraction, Extractor
from job_hunters.models import (
    CandidateCompany,
    CandidateStatus,
    Company,
    FetchRun,
    FetchStatus,
    JobSource,
)
from job_hunters.scoring import ScoringReport, select_candidates

NOW = datetime(2026, 9, 14, 6, 0, tzinfo=UTC)

PROFILE_DICT = {
    "titles": {"include": ["research scientist"], "exclude": ["sales"]},
    "keywords": {"strong": ["post-training"], "supporting": ["alignment", "evaluation"]},
    "location": {
        "base": "portugal",
        "priority": [{"work_modes": ["onsite", "hybrid"], "regions": ["portugal"]}],
        "work_authorization": {"have": ["eu"]},
    },
    "scoring": {
        "threshold": 70, "max_llm_scores_per_run": 300, "rubric": "Weight post-training.",
        "bands": [{"low": 0, "high": 100, "meaning": "The scale."}],
    },
}

ACME = CompanyEntry(slug="acme", name="Acme", ats="greenhouse", token="acme", tier="lab")


def _config(system: dict | None = None, watchlist: list[CompanyEntry] | None = None) -> AppConfig:
    """An AppConfig built from literals so a test can vary one setting at a time."""
    return AppConfig(
        search_profile=SearchProfile.model_validate(PROFILE_DICT),
        system=SystemConfig.model_validate(system or {}),
        watchlist=[ACME] if watchlist is None else watchlist,
        secrets=Secrets(_env_file=None),
    )


def _remoteok(*postings) -> dict:
    """A fake RemoteOK source carrying these postings, keyed the way `run_discovery` wants."""
    return {"remoteok": FakeDiscoverySource.returning("remoteok", *postings)}


def _hn(*postings) -> dict:
    """A fake Hacker News source carrying these postings."""
    return {"hn": FakeDiscoverySource.returning("hn", *postings)}


def sighting(source_job_id: str, company: str | None, title: str = "Research Scientist", **kw):
    """A structured aggregator posting naming its company."""
    kw.setdefault("source", "remoteok")
    return make_posting(source_job_id, title, company_name=company, location=None, **kw)


def comment(source_job_id: str, text: str):
    """A prose posting. The first line is the title and the company is in the text."""
    header, _, body = text.partition("\n")
    return make_posting(source_job_id, header, source="hn", description=body, location=None)


class FakeProber:
    """Answers probes from a table and records the calls for inspection."""

    def __init__(self, boards: dict[str, list[Board]] | None = None) -> None:
        """Takes the boards each lowercase name should find."""
        self.boards = boards or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, name: str, hints: tuple[str, ...]) -> list[Board]:
        """Records the call and answers from the table, matching on the lowercase name."""
        self.calls.append((name, hints))
        return list(self.boards.get(name.lower(), []))


def board(ats: str, token: str, jobs: int = 10) -> Board:
    """One board a probe might find."""
    return Board(ats, token, jobs, f"https://{ats}.test/{token}")


def _extractor(*answers) -> tuple[Extractor, FakeAnthropic]:
    """An extractor answering from a script of Extractions."""
    client = FakeAnthropic(*answers)
    return Extractor(client, "claude-haiku-4-5"), client


def _run(session: Session, sources: dict, *, prober=None, extractor=None, config=None,
         now: datetime = NOW, **kwargs) -> DiscoveryReport:
    """One discovery run with every network edge faked."""
    extractor = extractor or _extractor(Extraction(company=None, roles=[], careers_url=None))[0]
    kwargs.setdefault("only", list(sources))
    return run_discovery(
        sources=sources, prober=prober or FakeProber(), extractor=extractor,
        config=config or _config(), now=now, **kwargs,
    )


def _candidates(session: Session) -> list[CandidateCompany]:
    """Every candidate row (oldest first)."""
    session.expire_all()
    return session.scalars(select(CandidateCompany).order_by(CandidateCompany.id)).all()


def _sightings(session: Session) -> list[JobSource]:
    """Every stored sighting: a posting row with no company."""
    session.expire_all()
    return session.scalars(select(JobSource).where(JobSource.company_id.is_(None))).all()


def test_a_structured_posting_becomes_a_candidate_with_its_board(session: Session) -> None:
    """A named posting is stored as a sighting, probed once and queued with what the probe found."""
    prober = FakeProber({"prior labs": [board("ashby", "prior-labs", 24)]})
    report = _run(session, _remoteok(sighting("1", "Prior Labs")), prober=prober)

    assert report.total("matched") == 1 and report.total("new_candidates") == 1
    assert report.total("boards_found") == 1 and report.pending == 1
    [candidate] = _candidates(session)
    assert candidate.status == CandidateStatus.PENDING
    assert (candidate.ats_type, candidate.ats_token, candidate.board_jobs) == ("ashby", "prior-labs", 24)
    assert candidate.roles == ["Research Scientist"]
    assert candidate.sightings == 1 and candidate.evidence[0]["source"] == "remoteok"
    [row] = _sightings(session)
    assert row.job_id is None and row.source == "remoteok" and row.is_open
    run = session.scalar(select(FetchRun).where(FetchRun.source == "remoteok"))
    assert run.status == FetchStatus.OK and run.company_id is None and run.item_count == 1


def test_scanning_twice_changes_nothing(session: Session) -> None:
    """The stored sighting is the memory. A second run probes, names and queues nothing."""
    prober = FakeProber({"prior labs": [board("ashby", "prior-labs")]})
    sources = _remoteok(sighting("1", "Prior Labs"))
    _run(session, sources, prober=prober)
    report = _run(session, sources, prober=prober, now=NOW + timedelta(days=7))

    assert report.total("new_sightings") == 0 and report.total("seen_before") == 1
    assert report.total("new_candidates") == 0 and report.total("seen_again") == 0
    assert len(prober.calls) == 1
    [candidate] = _candidates(session)
    assert candidate.sightings == 1
    [row] = _sightings(session)
    assert row.last_seen.replace(tzinfo=UTC) == NOW + timedelta(days=7)


def test_supporting_keywords_and_excluded_titles_queue_nobody(session: Session) -> None:
    """Only a title term or a strong keyword keeps a posting."""
    report = _run(session, _remoteok(
        sighting("1", "HelloFresh", "Menu Planner", description="Alignment with the kitchen team."),
        sighting("2", "Acme Sales Co", "Sales Engineer", description="Sell our post-training stack."),
        sighting("3", "Nobody", "Backend Engineer", description="Java."),
    ))
    assert report.total("fetched") == 3 and report.total("matched") == 0
    assert _candidates(session) == [] and _sightings(session) == []


def test_title_terms_are_found_in_the_body_too(session: Session) -> None:
    """A prose posting lists its roles wherever it likes so the body counts."""
    extractor, _ = _extractor(Extraction(company="Aerdos", roles=["Research Engineer"], careers_url=None))
    report = _run(session, _hn(comment("10", "Aerdos | SF | Full time\nWe are hiring a research scientist.")),
                  extractor=extractor)
    assert report.total("matched") == 1 and report.total("new_candidates") == 1


def test_prose_is_named_by_the_model_up_to_the_cap_and_the_rest_waits(session: Session) -> None:
    """Two calls allowed and three comments: the third is neither stored nor named until next run."""
    extractor, client = _extractor(
        Extraction(company="Prior Labs", roles=["Research Scientist"], careers_url=None),
        Extraction(company="Tufalabs", roles=["Member of Technical Staff"], careers_url=None),
        Extraction(company="Mechanize", roles=["Research Engineer"], careers_url=None),
    )
    config = _config({"discovery": {"max_extractions_per_run": 2}})
    comments = [
        comment("1", "Prior Labs | Berlin\nResearch scientist wanted."),
        comment("2", "Tufalabs | Zurich\nPost-training research."),
        comment("3", "Mechanize | SF\nResearch scientist, alignment."),
    ]
    report = _run(session, _hn(*comments), extractor=extractor, config=config)
    assert report.total("extracted") == 2 and report.total("capped") == 1
    assert report.usage.output_tokens == 400
    assert len(client.requests) == 2
    assert {c.name for c in _candidates(session)} == {"Prior Labs", "Tufalabs"}
    assert len(_sightings(session)) == 2

    report = _run(session, _hn(*comments), extractor=extractor, config=config)
    assert report.total("extracted") == 1 and report.total("seen_before") == 2
    assert {c.name for c in _candidates(session)} == {"Prior Labs", "Tufalabs", "Mechanize"}


def test_a_posting_the_model_cannot_name_is_remembered_but_not_queued(session: Session) -> None:
    """A recruiter's post has no company. It is stored so it is never paid for twice."""
    extractor, client = _extractor(Extraction(company=None, roles=["Research Scientist"], careers_url=None))
    sources = _hn(comment("1", "Agency | Roles for clients\nResearch scientist positions."))
    report = _run(session, sources, extractor=extractor)
    assert report.total("unattributed") == 1 and _candidates(session) == []
    assert len(_sightings(session)) == 1

    _run(session, sources, extractor=extractor)
    assert len(client.requests) == 1


def test_a_careers_link_naming_a_board_is_tried_first_and_wins(session: Session) -> None:
    """The token in `jobs.ashbyhq.com/prior-labs` beats a busier board under another spelling."""
    extractor, _ = _extractor(
        Extraction(company="Prior Labs", roles=["Research Scientist"],
                   careers_url="https://jobs.ashbyhq.com/prior-labs/123")
    )
    prober = FakeProber({"prior labs": [board("greenhouse", "priorlabs", 99), board("ashby", "prior-labs", 3)]})
    _run(session, _hn(comment("1", "Prior Labs | Berlin\nResearch scientist.")), extractor=extractor, prober=prober)
    assert prober.calls == [("Prior Labs", ("prior-labs",))]
    [candidate] = _candidates(session)
    assert (candidate.ats_type, candidate.ats_token) == ("ashby", "prior-labs")
    assert candidate.careers_url == "https://jobs.ashbyhq.com/prior-labs/123"


def test_a_fatal_extraction_error_stops_the_prose_and_not_the_other_sources(session: Session) -> None:
    """A rejected key ends the model calls. The structured sources still queue their companies."""
    extractor = Extractor(FakeAnthropic(api_error(anthropic.AuthenticationError, 401)), "m")
    sources = {
        **_hn(comment("1", "Prior Labs | Berlin\nResearch scientist."),
              comment("2", "Tufalabs | Zurich\nResearch scientist.")),
        **_remoteok(sighting("3", "Mechanize")),
    }
    report = _run(session, sources, extractor=extractor)
    assert report.aborted and "AuthenticationError" in report.aborted
    assert report.total("extraction_failed") == 1
    assert {c.name for c in _candidates(session)} == {"Mechanize"}
    assert len(_sightings(session)) == 1
    assert not report.failures  # the sources themselves were read fine


def test_a_transient_extraction_error_skips_that_posting_for_next_run(session: Session) -> None:
    """A 429 on one comment leaves it unstored so the next run tries it again."""
    extractor, client = _extractor(
        api_error(anthropic.RateLimitError, 429),
        Extraction(company="Tufalabs", roles=[], careers_url=None),
    )
    comments = [comment("1", "Prior Labs | Berlin\nResearch scientist."),
                comment("2", "Tufalabs | Zurich\nResearch scientist.")]
    report = _run(session, _hn(*comments), extractor=extractor)
    assert report.total("extraction_failed") == 1 and report.total("extracted") == 1
    assert report.aborted is None
    assert {c.name for c in _candidates(session)} == {"Tufalabs"}


def test_a_missing_api_key_is_named_and_the_structured_sources_still_run(
    session: Session, monkeypatch
) -> None:
    """Without a key no prose can be named. The run says which variable and carries on."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sources = {**_hn(comment("1", "Prior Labs | Berlin\nResearch scientist.")),
               **_remoteok(sighting("2", "Mechanize"))}
    report = run_discovery(sources=sources, only=list(sources), prober=FakeProber(),
                           config=_config(), now=NOW)
    assert "ANTHROPIC_API_KEY" in report.aborted
    assert {c.name for c in _candidates(session)} == {"Mechanize"}


def test_a_watched_company_is_never_queued_by_name_or_by_board(session: Session) -> None:
    """Acme is in the watchlist, so "ACME, Inc." resolves to Acme's board. Neither is a candidate."""
    prober = FakeProber({"acme, inc.": [board("greenhouse", "acme")]})
    report = _run(session, _remoteok(sighting("1", "Acme"), sighting("2", "ACME, Inc.")), prober=prober)
    assert report.total("already_watched") == 2 and _candidates(session) == []
    assert len(_sightings(session)) == 2  # still remembered so neither is probed again


def test_a_company_removed_from_the_watchlist_is_not_proposed_again(session: Session) -> None:
    """A deactivated `companies` row was a decision and the loop must respect it."""
    session.add(Company(slug="old", name="Old Lab", ats_type="ashby", ats_config={"token": "old"}, active=False))
    session.commit()
    report = _run(session, _remoteok(sighting("1", "Old Lab")))
    assert report.total("already_watched") == 1 and _candidates(session) == []


def test_two_spellings_that_resolve_to_one_board_are_one_candidate(session: Session) -> None:
    """"Scale" and "Scale AI" both find `scaleai` so the second sighting joins the first."""
    prober = FakeProber({"scale": [board("greenhouse", "scaleai")], "scale ai": [board("greenhouse", "scaleai")]})
    report = _run(session, _remoteok(sighting("1", "Scale"), sighting("2", "Scale AI", "Research Scientist, Evals")),
                  prober=prober)
    assert report.total("new_candidates") == 1 and report.total("seen_again") == 1
    [candidate] = _candidates(session)
    assert candidate.name == "Scale" and candidate.sightings == 2
    assert candidate.roles == ["Research Scientist", "Research Scientist, Evals"]
    assert len(candidate.evidence) == 2


def test_a_company_without_a_board_is_probed_again_only_after_the_wait(session: Session) -> None:
    """No board today does not mean no board next month, but it is not worth thirty requests a week."""
    config = _config({"discovery": {"reprobe_after_days": 30}})
    prober = FakeProber()
    _run(session, _remoteok(sighting("1", "Mistral")), prober=prober, config=config)
    [candidate] = _candidates(session)
    assert candidate.ats_type is None and len(prober.calls) == 1

    _run(session, _remoteok(sighting("2", "Mistral")), prober=prober, config=config,
         now=NOW + timedelta(days=29))
    assert len(prober.calls) == 1

    prober.boards["mistral"] = [board("lever", "mistral")]
    _run(session, _remoteok(sighting("3", "Mistral")), prober=prober, config=config,
         now=NOW + timedelta(days=30))
    assert len(prober.calls) == 2
    [candidate] = _candidates(session)
    assert (candidate.ats_type, candidate.ats_token, candidate.sightings) == ("lever", "mistral", 3)


def test_a_watchlist_line_added_by_hand_resolves_the_candidate(session: Session) -> None:
    """Approval is defined by the file. A line written on another machine still counts."""
    prober = FakeProber({"prior labs": [board("ashby", "prior-labs")]})
    _run(session, _remoteok(sighting("1", "Prior Labs")), prober=prober)
    added = CompanyEntry(slug="prior-labs", name="Prior Labs", ats="ashby", token="prior-labs")
    report = _run(session, _remoteok(sighting("2", "Prior Labs")), prober=prober,
                  config=_config(watchlist=[ACME, added]))
    assert report.reconciled == 1 and report.pending == 0
    [candidate] = _candidates(session)
    assert candidate.status == CandidateStatus.APPROVED and candidate.slug == "prior-labs"
    assert report.total("already_watched") == 1


def test_reconcile_matches_on_the_board_when_the_name_differs(session: Session) -> None:
    """A candidate named "Scale" is resolved by a watchlist line for `scaleai` on the same board."""
    session.add(CandidateCompany(name="Scale", name_key="scale", ats_type="greenhouse",
                                 ats_token="scaleai", sightings=1, roles=[], evidence=[]))
    session.commit()
    watched = Watched.build([CompanyEntry(slug="scaleai", name="Scale AI", ats="greenhouse", token="scaleai")])
    assert reconcile(session, watched, NOW) == 1
    assert session.scalar(select(CandidateCompany)).slug == "scaleai"


def test_one_source_failing_does_not_stop_the_others(session: Session) -> None:
    """A 503 from one site is a failed `fetch_runs` row. The next site is read as usual."""
    sources = {
        "remoteok": FakeDiscoverySource("remoteok", failed("HTTP 503 for https://remoteok.com/api")),
        "arbeitnow": FakeDiscoverySource.returning("arbeitnow", sighting("1", "Prior Labs", source="arbeitnow")),
    }
    report = _run(session, sources)
    assert [s.source for s in report.failures] == ["remoteok"]
    assert report.total("new_candidates") == 1
    runs = {r.source: r for r in session.scalars(select(FetchRun))}
    assert runs["remoteok"].status == FetchStatus.FAILED and "503" in runs["remoteok"].error
    assert runs["arbeitnow"].status == FetchStatus.OK


def test_a_crash_while_scanning_one_source_is_recorded_and_the_run_continues(session: Session) -> None:
    """A bug in the probe for one source must not take the other sources down with it."""

    def exploding(name: str, hints: tuple[str, ...]) -> list[Board]:
        """Raises for one company only."""
        if name == "Prior Labs":
            raise RuntimeError("the probe is on fire")
        return []

    sources = {
        "remoteok": FakeDiscoverySource.returning("remoteok", sighting("1", "Prior Labs")),
        "remotive": FakeDiscoverySource.returning("remotive", sighting("2", "Tufalabs", source="remotive")),
    }
    report = _run(session, sources, prober=exploding)
    assert [s.source for s in report.failures] == ["remoteok"]
    assert "on fire" in report.failures[0].error
    assert {c.name for c in _candidates(session)} == {"Tufalabs"}
    run = session.scalar(select(FetchRun).where(FetchRun.source == "remoteok", FetchRun.status == "failed"))
    assert run is not None and "on fire" in run.error


def test_a_dry_run_counts_what_is_new_and_writes_nothing(session: Session) -> None:
    """Fetch and prefilter only: no sighting, no candidate, no run row, no model call and no probe."""
    extractor, client = _extractor(Extraction(company="Prior Labs", roles=[], careers_url=None))
    prober = FakeProber({"mechanize": [board("ashby", "mechanize")]})
    sources = {**_hn(comment("1", "Prior Labs | Berlin\nResearch scientist.")),
               **_remoteok(sighting("2", "Mechanize"))}
    report = _run(session, sources, extractor=extractor, prober=prober, dry_run=True)
    assert report.dry_run and report.total("new_sightings") == 2
    assert client.requests == [] and prober.calls == []
    assert _sightings(session) == [] and _candidates(session) == []
    assert session.scalar(select(FetchRun)) is None


def test_only_narrows_the_enabled_sources_and_never_switches_one_on(session: Session) -> None:
    """`--only hn` with hn switched off in the config reads nothing from Hacker News."""
    config = _config({"discovery": {"sources": {"hn": False}}})
    sources = {**_hn(comment("1", "Prior Labs | Berlin\nResearch scientist.")),
               **_remoteok(sighting("2", "Mechanize"))}
    report = _run(session, sources, config=config, only=["hn", "remoteok"])
    assert [s.source for s in report.sources] == ["remoteok"]
    assert sources["hn"].calls == 0


def test_sightings_never_reach_the_scoring_prefilter(session: Session) -> None:
    """A sighting is a posting row with no job and scoring only ever reads postings with one."""
    _run(session, _remoteok(sighting("1", "Prior Labs")))
    assert len(_sightings(session)) == 1
    report = ScoringReport()
    candidates = select_candidates(session, _config().search_profile, 1, report)
    assert candidates == [] and report.open_postings == 0
