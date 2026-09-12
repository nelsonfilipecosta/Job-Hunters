"""The command-line interface.

Defines the subcommands available at the terminal and connects each
one to the code that does the actual work:

    job-hunters show-config  loads, validates and summarises the config
    job-hunters check-git    refuses if `profile/`, `data/`, `backups/` or any `.env*` are git-tracked
    job-hunters init-db      creates the data directories and the database schema
    job-hunters ingest       fetches every watched board into the database
    job-hunters discover     finds which ATS and slug host a company's board
    job-hunters score        judges unscored postings with the LLM (prefilter then judge)
    job-hunters eval-scoring evaluates the scorer against hand-labeled postings
    job-hunters digest       builds the daily email and sends it
    job-hunters backup       writes a timestamped backup of the database to `backups/`

This file does no work of its own. `build_parser()` registers each subcommand
under a name. `main()` reads what was typed and calls whichever `cmd_*`
function was selected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import paths
from .backup import BackupError, backup_database
from .config import ConfigError, load_all
from .db import SchemaError, init_db
from .digest import run_digest
from .gitcheck import GitSafetyError, check_git_safety
from .ingest import run_ingest
from .discover import probe
from .evaluate import run_evaluation
from .judge import Usage, Verdict, cache_minimum_tokens
from .mailer import DeliveryError
from .scoring import Candidate, group_by_text, run_scoring


def _positive_int(value: str) -> int:
    """An argparse type for a count that must be positive and at least one."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def cmd_show_config(_args: argparse.Namespace) -> int:
    """Handle `job-hunters show-config`: load, validate and summarise the config."""
    config = load_all()
    # Summarise the search profile in `search_profile.yaml`
    profile = config.search_profile
    print(f"config directory       {paths.CONFIG_DIR}")
    print()
    print("search_profile.yaml")
    print(f"  titles               {len(profile.titles.include)} included, "
          f"{len(profile.titles.exclude)} excluded")
    print(f"  keywords             {len(profile.keywords.strong)} strong, "
          f"{len(profile.keywords.supporting)} supporting")
    print(f"  base                 {profile.location.base}")
    print(f"  priority             {len(profile.location.priority)} rule(s)")
    print(f"  acceptable           {len(profile.location.acceptable)} rule(s)")
    print(f"  no sponsorship in    {len(profile.eligible_regions())} countries")
    print(f"  needs sponsorship    "
          f"{', '.join(profile.location.work_authorization.need_sponsorship) or '-'}")
    print(f"  threshold            {profile.scoring.threshold} "
          f"(prompt version {profile.scoring.prompt_version})")
    scoring = profile.scoring
    print(f"  judge prompt         {len(scoring.rubric.split()) } rubric words, "
          f"{len(scoring.bands)} score band(s), {len(scoring.examples)} example(s)"
          f"{'' if scoring.guidance.strip() else ', no guidance'}")
    # Summarise the system configuration in `system_config.yaml`
    system = config.system
    print()
    print("system_config.yaml")
    print(f"  timezone             {system.timezone}")
    print(f"  base url             {system.base_url}  (every digest link is built from this)")
    print(f"  action links last    {system.actions.token_ttl_days} days")
    print(f"  ingest / score       {system.schedules.ingest} / {system.schedules.score}")
    print(f"  misfire grace        {system.schedules.misfire_grace_minutes} min "
          f"({system.schedules.digest_misfire_grace_minutes} min for the digest, "
          f"{system.schedules.backup_misfire_grace_minutes} min for the backup)")
    print(f"  backup               {system.schedules.backup} into {paths.BACKUP_DIR}")
    print(f"  digest               {system.schedules.digest} via "
          f"{system.email.smtp_host}:{system.email.smtp_port}")
    print(f"  judge / tailor       {system.models.judge} / {system.models.tailor}")
    suppression = system.digest.repeat_suppression
    if suppression.enabled:
        summary = (f"demote after {suppression.demote_after}, "
                   f"suppress after {suppression.suppress_after}")
    else:
        summary = "disabled"
    print(f"  repeat suppression   {summary}")
    # Summarise the companies watchlist in `companies_watchlist.yaml`
    print()
    print("companies_watchlist.yaml")
    print(f"  companies            {len(config.watchlist)} "
          f"({sum(1 for c in config.watchlist if c.active)} active)")
    by_ats: dict[str, int] = {}
    for entry in config.watchlist:
        by_ats[entry.ats.value] = by_ats.get(entry.ats.value, 0) + 1
    for ats, count in sorted(by_ats.items()):
        print(f"    {ats:<18} {count}")
    # What `.env` provides. Addresses are shown so they can be checked for
    # typos. Keys and passwords are only ever reported as present.
    print()
    print(".env")
    for name, value in config.secrets.summary().items():
        print(f"  {name:<20} {value}")
    return 0


def cmd_check_git(_args: argparse.Namespace) -> int:
    """Handle `job-hunters check-git`: refuse if any private path is git-tracked."""
    check_git_safety()
    print("No private data is tracked by git.")
    return 0


def cmd_init_db(_args: argparse.Namespace) -> int:
    """Handle `job-hunters init-db`: create the data directories and the schema."""
    paths.ensure_runtime_dirs()
    db_file = init_db()
    print(f"Schema ready at {db_file}.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Handle `job-hunters ingest`: fetch every watched board into the database."""
    report = run_ingest(only=args.only or None)
    width = max((len(c.slug) for c in report.companies), default=8)
    for c in report.companies:
        if c.failed:
            print(f"  {c.slug:<{width}}  FAILED  {c.error}")
        else:
            print(f"  {c.slug:<{width}}  ok      {c.fetched:4} fetched  "
                  f"{c.new_sources:3} new  {c.updated_sources:3} updated  "
                  f"{c.closed:3} closed  {c.new_jobs:3} new jobs")
    print()
    print(f"{len(report.companies)} companies, {len(report.failures)} failed | "
          f"{report.total('fetched')} postings fetched, {report.total('new_sources')} new, "
          f"{report.total('closed')} closed and {report.total('new_jobs')} new jobs.")
    if report.total("repointed"):
        print(f"{report.total('repointed')} job(s) moved onto a still-open posting "
              f"after the one they displayed closed.")
    return 1 if report.failures else 0


def cmd_discover(args: argparse.Namespace) -> int:
    """Handle `job-hunters discover <name>`: find which ATS and slug host a company's job board."""
    hits = probe(args.name)
    if not hits:
        print(f"No Greenhouse, Lever or Ashby board found for {args.name!r}.")
        print("It may use Workday or a proprietary careers site.")
        return 1
    slug = args.name.strip().lower().replace(" ", "-")
    for hit in hits:
        note = "  (board exists but has no postings)" if hit.job_count == 0 else ""
        print(f"  {hit.ats:<11} {hit.token:<20} {hit.job_count:4} jobs  {hit.url}{note}")
    with_postings = [h for h in hits if h.job_count > 0]
    if not with_postings:
        # A board that exists but has nothing on it is usually a squatted or
        # abandoned slug, not the company you are looking for. Do not suggest it.
        print()
        print("Every board found is empty, so none is worth adding yet.")
        return 1
    print()
    print("Add to config/companies_watchlist.yaml:")
    best = max(with_postings, key=lambda h: h.job_count)
    print("  " + best.watchlist_line(slug, args.name.strip()))
    return 0


def _print_verdict(candidate: Candidate, verdict: Verdict, usage: Usage) -> None:
    """Prints one judged posting: score, title, company, fit and what the call cost."""
    print(f"  {verdict.score:3}  {candidate.text.title} @ {candidate.text.company}  "
          f"[{candidate.location_fit}, {verdict.work_authorization}; "
          f"cache read {usage.cache_read_input_tokens:,}, "
          f"written {usage.cache_creation_input_tokens:,}, "
          f"uncached {usage.input_tokens:,}, out {usage.output_tokens:,}]")
    print(f"       {verdict.summary}")


def cmd_score(args: argparse.Namespace) -> int:
    """Handle `job-hunters score`: prefilter every open posting and then the judge up to the cap."""
    report = run_scoring(
        limit=args.limit,
        dry_run=args.dry_run,
        on_verdict=_print_verdict if args.verbose else None,
    )
    if args.verbose and report.judged:
        print()
    print(f"open postings          {report.open_postings}")
    print(f"  already judged       {report.already_judged}")
    print(f"  excluded by title    {report.eliminated_title}")
    print(f"  excluded by location {report.eliminated_location}")
    print(f"  no title or keyword  {report.unmatched}")
    if report.stale:
        print(f"  stale text           {report.stale}  (run `job-hunters ingest` first)")
    print(f"  candidates           {len(report.candidates)} postings, "
          f"{report.distinct_texts} distinct texts")
    if args.dry_run:
        groups = group_by_text(report.candidates)[: report.cap]
        print()
        print(f"Dry run. The first {len(groups)} of {report.distinct_texts} texts "
              f"that would be judged (cap {report.cap}):")
        for siblings in groups:
            lead = siblings[0]
            extra = f" (+{len(siblings) - 1} identical)" if len(siblings) > 1 else ""
            print(f"  {lead.text.title} @ {lead.text.company}  "
                  f"[{lead.location_fit}; matched on {lead.matched_on!r}]{extra}")
        return 0
    print()
    print(f"judged {report.judged} texts (cap {report.cap}), {report.scored} scores written, "
          f"{report.failed} failed, {report.carried_over} carried over")
    usage = report.usage
    print(f"tokens: cache written {usage.cache_creation_input_tokens:,}, "
          f"cache read {usage.cache_read_input_tokens:,}, "
          f"uncached input {usage.input_tokens:,}, output {usage.output_tokens:,}")
    if report.cold_calls:
        # Name the configured model, and its minimum only when it is known.
        minimum = cache_minimum_tokens(report.model)
        target = (f" ({minimum:,} tokens for {report.model})" if minimum is not None
                  else f" for {report.model}")
        print(f"Warning: {report.cold_calls} call(s) after the first read nothing from the "
              f"cache. The prefix is probably shorter than the minimum cacheable length"
              f"{target}. Add Markdown records to profile/ or set models.judge to a model "
              f"with a lower minimum.", file=sys.stderr)
    if report.aborted:
        print(f"Error: stopped early: {report.aborted}", file=sys.stderr)
        return 1
    return 0


def cmd_eval_scoring(args: argparse.Namespace) -> int:
    """Handle `job-hunters eval-scoring`: precision, recall and f1-score against the labeled postings."""
    report = run_evaluation(args.labels, skip_llm=args.skip_llm)
    llm = report.llm_used
    print(f"  {'label':<5} {'prefilter':<30} {'score':>5}  {'outcome':<11} posting")
    for row in report.rows:
        if row.excluded_by:
            prefilter = f"excluded ({row.excluded_by})"
        elif row.matched_on is None:
            prefilter = "no title or keyword match"
        else:
            prefilter = f"kept ({row.matched_on})"
        if row.error:
            score = "err"
        else:
            score = "-" if row.verdict is None else str(row.verdict.score)
        label = "REL" if row.job.relevant else "---"
        print(f"  {label:<5} {prefilter[:30]:<30} {score:>5}  "
              f"{row.outcome(report.threshold, llm):<11} "
              f"{row.job.title} @ {row.job.company}  [{row.location_fit}]")
    kept = sum(1 for row in report.rows if row.reached_judge)
    kept_relevant = sum(1 for row in report.rows if row.reached_judge and row.job.relevant)
    counts = report.counts()
    print()
    print(f"{len(report.rows)} labeled postings, {report.relevant} relevant.")
    print(f"prefilter kept {kept} of {len(report.rows)} ({kept_relevant} of {report.relevant} "
          f"relevant, recall {report.prefilter_recall:.0%})")
    if llm:
        print(f"at threshold {report.threshold}: precision {report.precision:.2f}, "
              f"recall {report.recall:.2f}, f1 {report.f1:.2f} "
              f"(hits {counts['hit']}, misses {counts['miss']}, "
              f"false alarms {counts['false alarm']})")
        usage = report.usage
        print(f"tokens: cache written {usage.cache_creation_input_tokens:,}, "
              f"cache read {usage.cache_read_input_tokens:,}, "
              f"uncached input {usage.input_tokens:,}, output {usage.output_tokens:,}")
        errors = sum(1 for row in report.rows if row.error)
        if errors:
            print(f"Warning: {errors} posting(s) could not be judged", file=sys.stderr)
    else:
        print(f"judge skipped (--skip-llm): if the judge accepted everything the prefilter kept, "
              f"precision would be {report.precision:.2f} and recall {report.recall:.2f}")
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    """Handle `job-hunters digest`: build the daily email and send it."""
    report = run_digest(dry_run=args.dry_run)
    digest = report.digest
    if args.dry_run:
        print(report.html)
    out = sys.stderr if args.dry_run else sys.stdout

    print(f"{digest.digest_date}  {digest.subject()}", file=out)
    for section in digest.sections:
        print(f"  {section.spec.title:<16} {len(section.entries)}", file=out)
    if digest.still_open:
        print(f"  {'still open':<16} {digest.still_open}  (suppressed: shown often "
              f"enough already)", file=out)
    if args.dry_run:
        print("Dry run. Nothing was sent and no appearance was recorded.", file=out)
    else:
        print(f"Sent to {report.sent_to}. {report.recorded} appearance(s) recorded.", file=out)
    return 0


def cmd_backup(_args: argparse.Namespace) -> int:
    """Handle `job-hunters backup`: write a timestamped backup of the database."""
    report = backup_database()
    print(f"Backed up {report.source}")
    print(f"        to {report.path}")
    print(f"           {report.megabytes:.1f} MB, {report.tables} tables, reopened and checked.")
    if report.missing:
        print(f"Warning: the database has no {', '.join(report.missing)} table(s), so the "
              f"backup has none either. The database is older than this code: run "
              f"`job-hunters init-db` to add them.", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the `job-hunters` command-line parser and register its subcommands.

    Each subcommand is a name (e.g., "init-db"), some help text and the function
    that should run when it is typed. Adding a new command later means adding
    one more block like the ones below. No other part of this file changes.
    """
    parser = argparse.ArgumentParser(prog="job-hunters", description=__doc__)
    # `required=True` means running `job-hunters` with no subcommand is an
    # error rather than silently doing nothing.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `job-hunters show-config`
    show = subparsers.add_parser(
        "show-config", help="validate and summarise the three config files"
    )
    # `set_defaults()` attaches the function to run onto the parsed arguments,
    # so `main()` below can call it without an if/elif chain over command names.
    show.set_defaults(func=cmd_show_config)

    # `job-hunters check-git`
    check_git = subparsers.add_parser(
        "check-git", help="refuse if `profile/`, `data/`, `backups/` or any `.env*` are git-tracked"
    )
    check_git.set_defaults(func=cmd_check_git)

    # `job-hunters init-db`
    init = subparsers.add_parser(
        "init-db", help="create the data directories and the database schema"
    )
    init.set_defaults(func=cmd_init_db)

    # `job-hunters ingest [--only SLUG ...]`
    ingest = subparsers.add_parser(
        "ingest", help="fetch every watched board into the database"
    )
    ingest.add_argument(
        "--only", action="append", metavar="SLUG",
        help="fetch only this company (repeatable)"
    )
    ingest.set_defaults(func=cmd_ingest)

    # `job-hunters discover NAME`
    discover = subparsers.add_parser(
        "discover", help="find which ATS and slug host a company's job board"
    )
    discover.add_argument(
        "name",
        help="company name, e.g. 'Scale AI'"
    )
    discover.set_defaults(func=cmd_discover)

    # `job-hunters score [--limit N] [--dry-run] [--verbose]`
    score = subparsers.add_parser(
        "score", help="judge unscored postings with the LLM (prefilter then judge)"
    )
    score.add_argument(
        "--limit", type=_positive_int, metavar="N",
        help="judge at most N distinct texts this run "
             "(default: scoring.max_llm_scores_per_run)"
    )
    score.add_argument(
        "--dry-run", action="store_true",
        help="run the prefilter only: list what would be judged, but call and write nothing"
    )
    score.add_argument(
        "--verbose", action="store_true",
        help="print every verdict with its token usage as it arrives"
    )
    score.set_defaults(func=cmd_score)

    # `job-hunters eval-scoring [--skip-llm] [--labels PATH]`
    evaluate = subparsers.add_parser(
        "eval-scoring",
        help="evaluate the scorer against hand-labeled postings in tests/fixtures/labeled_jobs.yaml"
    )
    evaluate.add_argument(
        "--skip-llm", action="store_true",
        help="report the prefilter alone without calling the judge"
    )
    evaluate.add_argument(
        "--labels", type=Path, metavar="PATH",
        help="a different labeled file"
    )
    evaluate.set_defaults(func=cmd_eval_scoring)

    # `job-hunters digest [--dry-run]`
    digest = subparsers.add_parser(
        "digest", help="build the daily email and send it"
    )
    digest.add_argument(
        "--dry-run", action="store_true",
        help="write the HTML to stdout instead (send nothing and record nothing)"
    )
    digest.set_defaults(func=cmd_digest)

    # `job-hunters backup`
    backup = subparsers.add_parser(
        "backup", help="write a timestamped backup of the database to `backups/`"
    )
    backup.set_defaults(func=cmd_backup)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `job-hunters` command.

    Parses the command line, runs whichever `cmd_*` function was selected and
    returns its exit code.

    The `argv` parameter is what makes this function usable both as the real
    command and as something a test can call directly. When you run
    `job-hunters show-config` in a terminal, Python automatically stores the
    words you typed in a global list called `sys.argv`. If this function is
    called with no `argv` given, it falls back to reading that global list,
    which is exactly what happens when this runs as the real command.

    A test does not want to touch that global list, so it can instead call
    `main(["show-config"])` directly, passing the words in by hand as an
    ordinary list of strings. This function only has to check for that case
    because `parse_args()` already understands both: pass it `None` and it
    reads `sys.argv` itself; pass it a list and it uses that list instead.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except GitSafetyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except DeliveryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except SchemaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except BackupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
