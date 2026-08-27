"""Tests for the git-tracking safety check.

Every test builds a disposable git repository inside `tmp_path`, so these never
touch this project's real git history and never depend on it being clean at the
moment the suite happens to run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from job_hunters.gitcheck import (
    GitSafetyError,
    check_git_safety,
    find_tracked_private_files,
)


def _git(*args: str, cwd: Path) -> None:
    """Runs a git command against a specific repo, failing loudly if it errors."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An initialized empty git repository with no commits."""
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    return tmp_path


def test_a_clean_repo_has_nothing_tracked(repo: Path) -> None:
    """A repo where nothing was ever added reports no tracked private files."""
    assert find_tracked_private_files(repo) == []
    check_git_safety(repo)  # must not raise


def test_a_file_tracked_under_data_is_detected(repo: Path) -> None:
    """A file force-added under `data/` is reported by its exact relative path."""
    target = repo / "data" / "jobhunters.db"
    target.parent.mkdir()
    target.write_text("fake database contents")
    _git("add", "-f", "data/jobhunters.db", cwd=repo)

    assert find_tracked_private_files(repo) == ["data/jobhunters.db"]
    with pytest.raises(GitSafetyError, match="data/jobhunters.db"):
        check_git_safety(repo)


def test_a_tracked_env_file_is_detected(repo: Path) -> None:
    """A tracked `.env` file is caught and not just files under a directory."""
    (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-fake")
    _git("add", "-f", ".env", cwd=repo)

    assert find_tracked_private_files(repo) == [".env"]


def test_an_untracked_file_under_a_private_path_is_ignored(repo: Path) -> None:
    """A file that exists on disk but was never `git add`-ed is not reported."""
    target = repo / "profile" / "cv.md"
    target.parent.mkdir()
    target.write_text("my cv")

    assert find_tracked_private_files(repo) == []


def test_a_tracked_file_outside_the_private_paths_is_not_flagged(repo: Path) -> None:
    """Ordinary tracked project files are normal and must never be reported."""
    (repo / "README.md").write_text("hello")
    _git("add", "README.md", cwd=repo)

    assert find_tracked_private_files(repo) == []


def test_running_outside_a_git_repository_fails_loudly(tmp_path: Path) -> None:
    """No .git directory is a hard failure and never a silent pass."""
    with pytest.raises(GitSafetyError):
        find_tracked_private_files(tmp_path)
