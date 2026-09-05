"""The command-line interface.

Defines the subcommands available at the terminal and connects each
one to the code that does the actual work:

    job-hunters show-config  loads, validates and summarises the config
    job-hunters check-git    refuses if `profile/`, `data/`, `backups/` or `.env` are git-tracked
    job-hunters init-db      creates the data directories and the database schema
    job-hunters ingest       fetches every watched board into the database
    job-hunters discover     finds which ATS and slug host a company's board
    
This file does no work of its own. `build_parser()` registers each subcommand
under a name. `main()` reads what was typed and calls whichever `cmd_*`
function was selected.
"""

from __future__ import annotations

import argparse
import sys

from . import paths
from .config import ConfigError, load_all
from .db import init_db
from .gitcheck import GitSafetyError, check_git_safety
from .ingest import run_ingest
from .discover import probe


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
    # Summarise the system configuration in `system_config.yaml`
    system = config.system
    print()
    print("system_config.yaml")
    print(f"  timezone             {system.timezone}")
    print(f"  ingest / score       {system.schedules.ingest} / {system.schedules.score}")
    print(f"  digest               {system.schedules.digest} -> {system.email.to}")
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
        "check-git", help="refuse if `profile/`, `data/`, `backups/` or `.env` are git-tracked"
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
        # A config mistake is a user error, not a crash. Print it plainly
        # instead of letting a Python traceback reach the terminal.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except GitSafetyError as exc:
        # Same reasoning as ConfigError: expected and should never dump a
        # traceback. This is the one error this project must never let
        # a user miss.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
