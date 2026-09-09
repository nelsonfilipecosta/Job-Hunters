"""Evaluate the judge's calibration against hand-labeled postings.

`tests/fixtures/labeled_jobs.yaml` holds real postings with a "relevant" label
saying whether the candidate would want to see each one. This module runs the
same title exclusions, keyword union and judge over them and reports precision,
recall and f1-score at the configured threshold. It also reports how many relevant
postings the prefilter would have dropped before the judge ever saw them.

Note that location is reported but not applied. The labels are about the role
and most of the labeled postings are on-site in the US, which the declared
location rules exclude. Applying them would measure the config, not the judge.
The whole point of the file is to notice when the judge's calibration is off.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from . import paths
from .config import (
    ConfigError,
    NonEmptyStr,
    SearchProfile,
    StrictModel,
    _format_validation_error,
    _read_yaml,
    load_all,
)
from .judge import (
    Judge,
    JudgeError,
    PostingText,
    Usage,
    Verdict,
    build_system_prompt,
    load_profile_text,
    make_client,
)
from .models import WorkMode
from .normalize import parse_location
from .scoring import Prefilter, location_fit

DEFAULT_LABELS_PATH = paths.PROJECT_ROOT / "tests" / "fixtures" / "labeled_jobs.yaml"


class LabeledJob(StrictModel):
    """One hand-labeled posting from the fixture file."""

    id: NonEmptyStr
    company: NonEmptyStr
    title: NonEmptyStr
    location: str | None = None
    work_mode: WorkMode | None = None
    relevant: bool
    note: str = ""
    description: NonEmptyStr


def load_labeled_jobs(path: Path | None = None) -> list[LabeledJob]:
    """Loads and validates the fixture, rejecting duplicate ids."""
    target = path or DEFAULT_LABELS_PATH
    raw = _read_yaml(target)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{target.name} must be a non-empty YAML list of labeled postings.")
    try:
        jobs = TypeAdapter(list[LabeledJob]).validate_python(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(target, exc)) from exc
    counts = Counter(job.id for job in jobs)
    duplicates = sorted(job_id for job_id, count in counts.items() if count > 1)
    if duplicates:
        raise ConfigError(f"{target.name}: duplicate id(s): {', '.join(duplicates)}")
    return jobs


@dataclass
class EvalRow:
    """What happened to one labeled posting on its way through the scorer."""

    job: LabeledJob
    location_fit: str
    excluded_by: str | None
    matched_on: str | None
    verdict: Verdict | None = None
    error: str | None = None

    @property
    def reached_judge(self) -> bool:
        """True when the prefilter would have sent this posting to the model."""
        return self.excluded_by is None and self.matched_on is not None

    def predicted(self, threshold: int, llm_used: bool) -> bool:
        """Whether the scorer would show this posting in the digest.

        Without the judge, everything the prefilter kept counts as shown, which
        is the most optimistic reading of the prefilter on its own.
        """
        if not self.reached_judge:
            return False
        if not llm_used:
            return True
        return self.verdict is not None and self.verdict.score >= threshold

    def outcome(self, threshold: int, llm_used: bool) -> str:
        """Comparing the prediction to the label: hit, miss, false alarm or ok."""
        predicted = self.predicted(threshold, llm_used)
        if self.job.relevant:
            return "hit" if predicted else "miss"
        return "false alarm" if predicted else "ok"


@dataclass
class EvalReport:
    """Precision, recall and f1-score of the judge over the labeled postings."""

    rows: list[EvalRow]
    threshold: int
    llm_used: bool
    usage: Usage = field(default_factory=Usage)

    def counts(self) -> Counter[str]:
        """How many rows ended as each outcome."""
        return Counter(row.outcome(self.threshold, self.llm_used) for row in self.rows)

    @property
    def relevant(self) -> int:
        """How many labeled postings the candidate would want to see."""
        return sum(1 for row in self.rows if row.job.relevant)

    @property
    def prefilter_recall(self) -> float:
        """The share of relevant postings that the prefilter lets through to the judge."""
        kept = sum(1 for row in self.rows if row.job.relevant and row.reached_judge)
        return kept / self.relevant if self.relevant else 0.0

    @property
    def precision(self) -> float:
        """The precision of relevant postings the scorer would show."""
        counts = self.counts()
        shown = counts["hit"] + counts["false alarm"]
        return counts["hit"] / shown if shown else 0.0

    @property
    def recall(self) -> float:
        """The recall of relevant postings the scorer would show."""
        counts = self.counts()
        return counts["hit"] / self.relevant if self.relevant else 0.0

    @property
    def f1(self) -> float:
        """The f1-score of relevant postings the scorer would show."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def evaluate(
    labeled: list[LabeledJob], profile: SearchProfile, judge: Judge | None
) -> EvalReport:
    """Runs the prefilter and the judge over the labeled postings, when a judge is given."""
    prefilter = Prefilter(profile)
    report = EvalReport(rows=[], threshold=profile.scoring.threshold, llm_used=judge is not None)
    for job in labeled:
        parsed = parse_location(job.location, workplace_type=job.work_mode)
        row = EvalRow(
            job=job,
            location_fit=location_fit(profile, parsed.region, parsed.regions, parsed.work_mode),
            excluded_by=prefilter.title_exclusion(job.title),
            matched_on=prefilter.matched_term(job.title, job.description),
        )
        if judge is not None and row.reached_judge:
            text = PostingText(
                company=job.company,
                title=job.title,
                location_raw=job.location,
                region=parsed.region,
                work_mode=parsed.work_mode,
                description=job.description,
            )
            try:
                row.verdict, usage = judge.judge(text)
                report.usage = report.usage + usage
            except JudgeError as exc:
                row.error = str(exc)
                if exc.fatal:
                    raise ConfigError(f"The judge cannot run: {exc}") from exc
        report.rows.append(row)
    return report


def run_evaluation(labels_path: Path | None = None, *, skip_llm: bool = False) -> EvalReport:
    """Loads config, the labels and the judge (unless skipped), then evaluates.
    
    What the command `job-hunters eval-scoring` runs."""
    config = load_all()
    labeled = load_labeled_jobs(labels_path)
    judge = None
    if not skip_llm:
        api_key = config.secrets.require("anthropic_api_key")
        judge = Judge(
            make_client(api_key),
            config.system.models.judge,
            build_system_prompt(config.search_profile, load_profile_text()),
        )
    return evaluate(labeled, config.search_profile, judge)
