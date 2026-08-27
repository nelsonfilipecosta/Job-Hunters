"""Checks that private paths are never committed to git.

This is the second line of defense. `.gitignore` is what normally keeps
`profile/`, `data/` and `.env` out of git in the first place. This check exists
for the case where that failed (e.g., `git add -f` or an edited `.gitignore`)
and asks git directly, rather than trusting that nothing slipped through.

This runs automatically as git pre-commit hook, but can also be run manually
with `job-hunters check-git`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import paths

# Paths, relative to the repository root, that must never be git-tracked
PRIVATE_PATH_NAMES = ("profile", "data", ".env")


class GitSafetyError(Exception):
    """Raised when a private path is git-tracked or git itself is unusable here."""


def find_tracked_private_files(repo_root: Path | None = None) -> list[str]:
    """Returns every git-tracked file under `profile/`, `data/` and `.env`.

    An empty list means none of the private paths are tracked. Raises
    GitSafetyError if this isn't run inside a git repository at all, since a
    check that cannot run must never be mistaken for a check that passed.
    """
    root = repo_root or paths.PROJECT_ROOT
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", *PRIVATE_PATH_NAMES],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise GitSafetyError("git is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise GitSafetyError(
            f"Could not list git-tracked files in {root} "
            f"(is this a git repository?): {exc.stderr.strip()}"
        ) from exc

    return [line for line in result.stdout.splitlines() if line]


def check_git_safety(repo_root: Path | None = None) -> None:
    """Raises GitSafetyError naming every private file that is git-tracked."""
    tracked = find_tracked_private_files(repo_root)
    if tracked:
        listed = "\n".join(f"  {path}" for path in tracked)
        raise GitSafetyError(
            f"Private data is tracked by git and must be removed before "
            f"committing or pushing:\n{listed}\n"
            f"run: git rm --cached <path> for each file above"
        )
