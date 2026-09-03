"""Tests for the command-line interface."""

from __future__ import annotations

import pytest

from job_hunters.cli import main


def test_init_db_succeeds(tmp_path, monkeypatch) -> None:
    """`init-db` creates the schema and exits 0."""
    monkeypatch.setenv("JOB_HUNTERS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JOB_HUNTERS_BACKUP_DIR", str(tmp_path / "backups"))
    # `paths.py` and `db.py` read their module-level constants at import time,
    # so they must be reloaded after the environment changes.
    import importlib

    from job_hunters import db, paths

    importlib.reload(paths)
    importlib.reload(db)

    assert main(["init-db"]) == 0
    assert (tmp_path / "data" / "job_hunters.db").exists()

    db.reset_engine()
    importlib.reload(paths)
    importlib.reload(db)


def test_show_config_succeeds() -> None:
    """`show-config` validates this repository's real config and exits 0."""
    assert main(["show-config"]) == 0


def test_check_git_succeeds() -> None:
    """`check-git` passes in this repository where nothing private is tracked."""
    assert main(["check-git"]) == 0


def test_show_config_prints_a_summary(capsys) -> None:
    """The summary names each config file so the output is worth reading."""
    main(["show-config"])
    out = capsys.readouterr().out
    assert "search_profile.yaml" in out
    assert "system_config.yaml" in out
    assert "companies_watchlist.yaml" in out


def test_an_unknown_subcommand_exits_non_zero() -> None:
    """Argparse rejects a subcommand that was never registered."""
    with pytest.raises(SystemExit) as exc:
        main(["not-a-real-command"])
    assert exc.value.code != 0


def test_no_subcommand_is_an_error() -> None:
    """`required=True` means a bare `job-hunters` fails instead of doing nothing."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_a_config_error_exits_1_without_a_traceback(capsys, monkeypatch) -> None:
    """A broken config is a user error and not a code bug."""
    from job_hunters.cli import ConfigError

    def _raise(*_args, **_kwargs):
        raise ConfigError("`system_config.yaml` is invalid:\n  timezone: nope")

    monkeypatch.setattr("job_hunters.cli.load_all", _raise)

    assert main(["show-config"]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err.lower()
    assert "timezone" in captured.err


def test_a_git_safety_error_exits_1_without_a_traceback(capsys, monkeypatch) -> None:
    """Tracked private data is reported plainly and exits non-zero."""
    from job_hunters.cli import GitSafetyError

    def _raise(*_args, **_kwargs):
        raise GitSafetyError("Private data is tracked by git:\n  data/secret.db")

    monkeypatch.setattr("job_hunters.cli.check_git_safety", _raise)

    assert main(["check-git"]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err.lower()
    assert "data/secret.db" in captured.err
