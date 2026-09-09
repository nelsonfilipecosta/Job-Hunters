"""Tests for evaluation of the judge's calibration against hand-labeled postings."""

from __future__ import annotations

import pytest
import yaml

from conftest import FakeAnthropic, verdict
from job_hunters.config import ConfigError, load_search_profile
from job_hunters.evaluate import evaluate, load_labeled_jobs
from job_hunters.judge import Judge


def test_the_labeled_fixture_loads_with_both_labels_present() -> None:
    """The fixture is big enough to mean something and carries relevant and irrelevant postings."""
    jobs = load_labeled_jobs()
    assert len(jobs) >= 30
    assert {job.relevant for job in jobs} == {True, False}
    assert len({job.id for job in jobs}) == len(jobs)


def test_the_prefilter_keeps_every_relevant_labeled_posting() -> None:
    """The prefilter must be generous enough that no relevant posting is dropped before the judge."""
    report = evaluate(load_labeled_jobs(), load_search_profile(), judge=None)
    dropped = [row.job.id for row in report.rows if row.job.relevant and not row.reached_judge]
    assert report.prefilter_recall == 1.0, dropped
    assert not report.llm_used and report.recall == 1.0


def test_precision_and_recall_follow_the_judges_verdicts() -> None:
    """With a judge that only likes 'Post-Training' in the title, the numbers add up from its verdicts."""

    def answer(request: dict):
        """Scores by title alone, which is wrong often enough to produce every outcome."""
        title = request["messages"][0]["content"].split("\n")[1]
        return verdict(90 if "Post-Training" in title else 30)

    report = evaluate(load_labeled_jobs(), load_search_profile(), Judge(FakeAnthropic(answer), "m", "p"))

    counts = report.counts()
    shown = counts["hit"] + counts["false alarm"]
    assert shown == sum(1 for row in report.rows if row.verdict and row.verdict.score >= 70)
    assert report.precision == counts["hit"] / shown
    assert report.recall == counts["hit"] / report.relevant
    assert 0 < report.f1 < 1
    assert report.usage.cache_read_input_tokens > 0


def test_a_duplicate_id_or_a_missing_field_is_refused(tmp_path) -> None:
    """A bad fixture is a config error by name and not a wrong number at the end."""
    entry = {"id": "a", "company": "A", "title": "T", "relevant": True, "description": "d"}
    target = tmp_path / "labels.yaml"
    target.write_text(yaml.safe_dump([entry, entry]), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_labeled_jobs(target)

    target.write_text(yaml.safe_dump([{k: v for k, v in entry.items() if k != "relevant"}]), encoding="utf-8")
    with pytest.raises(ConfigError, match="relevant"):
        load_labeled_jobs(target)


def test_location_is_reported_but_never_applied() -> None:
    """The labels are about the role: a relevant on-site US posting still reaches the judge here."""
    report = evaluate(load_labeled_jobs(), load_search_profile(), judge=None)
    excluded_by_location = [row for row in report.rows if row.location_fit == "excluded" and row.job.relevant]
    assert excluded_by_location, "the fixture is expected to hold relevant on-site US postings"
    assert all(row.reached_judge for row in excluded_by_location)
