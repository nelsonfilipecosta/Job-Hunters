"""Command-line entry point.

Defines the subcommands available at the terminal and connects each
one to the code that does the actual work:

    job-hunters init-db      creates the data directories and the database schema
    job-hunters show-config  loads, validates and summarises the config

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


def cmd_init_db(_args: argparse.Namespace) -> int:
    """Handle `job-hunters init-db`: create the data directories and the schema.

    Takes no arguments of its own. `_args` exists only because every command
    function must accept the parsed arguments, whether it uses them or not.
    """
    paths.ensure_runtime_dirs()
    db_file = init_db()
    print(f"schema ready at {db_file}")
    return 0


def cmd_show_config(_args: argparse.Namespace) -> int:
    """Handle `job-hunters show-config`: load, validate and summarise the config.

    Useful on its own as a syntax check: it exits non-zero on any invalid file.
    """
    config = load_all()
    # Summarise the search profile in `search_profile.yaml`.
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
    # Summarise the system configuration in `system_config.yaml`.
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
    # Summarise the companies watchlist in `companies_watchlist.yaml`.
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


def build_parser() -> argparse.ArgumentParser:
    """Build the `job-hunters` command-line parser and register its subcommands.

    Each subcommand is a name (e.g., "init-db"), some help text and the function
    that should run when it is typed. Adding a new command later means adding
    one more block like the two below. No other part of this file changes.
    """
    parser = argparse.ArgumentParser(prog="job-hunters", description=__doc__)
    # `required=True` means running `job-hunters` with no subcommand is an
    # error rather than silently doing nothing.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `job-hunters init-db`
    init = subparsers.add_parser("init-db", help="create any missing tables")
    # `set_defaults()` attaches the function to run onto the parsed arguments,
    # so `main()` below can call it without an if/elif chain over command names.
    init.set_defaults(func=cmd_init_db)

    # `job-hunters show-config`
    show = subparsers.add_parser(
        "show-config", help="validate and summarise the three config files"
    )
    show.set_defaults(func=cmd_show_config)

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
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
