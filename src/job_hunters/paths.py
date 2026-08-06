"""Filesystem locations.

Every path is overridable by an environment variable so the same code runs
unchanged on the host (paths relative to the repo) and inside a container
(paths under /app, set in `docker-compose.yml`).
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file looking for `pyproject.toml`."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT = _env_path("JOB_HUNTERS_ROOT", _find_project_root())

# Committed to git: the three YAML files that drive the system.
CONFIG_DIR = _env_path("JOB_HUNTERS_CONFIG_DIR", PROJECT_ROOT / "config")

# Never committed: the SQLite database, drafts and the Chroma index.
DATA_DIR = _env_path("JOB_HUNTERS_DATA_DIR", PROJECT_ROOT / "data")

# Never committed: CV, thesis, papers and achievement records. Read-only input.
PROFILE_DIR = _env_path("JOB_HUNTERS_PROFILE_DIR", PROJECT_ROOT / "profile")

# Bind-mounted so backups survive `docker compose down -v`.
BACKUP_DIR = _env_path("JOB_HUNTERS_BACKUP_DIR", PROJECT_ROOT / "backups")

SEARCH_PROFILE_PATH = CONFIG_DIR / "search_profile.yaml"
SYSTEM_CONFIG_PATH = CONFIG_DIR / "system_config.yaml"
WATCHLIST_PATH = CONFIG_DIR / "companies_watchlist.yaml"

DB_PATH = _env_path("JOB_HUNTERS_DB_PATH", DATA_DIR / "job_hunters.db")
DRAFTS_DIR = DATA_DIR / "drafts"

# Paths that must never be tracked by git.
PRIVATE_PATHS = (PROFILE_DIR, DATA_DIR, PROJECT_ROOT / ".env")


def ensure_runtime_dirs() -> None:
    """Create the directories the system writes to. Safe to call repeatedly."""
    for directory in (DATA_DIR, DRAFTS_DIR, BACKUP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
