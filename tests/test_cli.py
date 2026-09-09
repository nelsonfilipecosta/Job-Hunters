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


def test_show_config_reports_secrets_without_their_values(capsys) -> None:
    """The summary says whether each secret is set and never what it is."""
    main(["show-config"])
    out = capsys.readouterr().out
    assert ".env" in out and "ANTHROPIC_API_KEY" in out
    assert "sk-ant" not in out


def test_score_dry_run_needs_no_key_and_writes_nothing(session, capsys) -> None:
    """`score --dry-run` runs the prefilter on the database and exits 0 without an API key."""
    assert main(["score", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "candidates" in out and "Dry run" in out


def test_score_without_a_key_fails_by_name(session, monkeypatch, capsys) -> None:
    """A missing key is a config error naming the variable and not a traceback from the SDK."""
    from dataclasses import replace

    from job_hunters import scoring
    from job_hunters.config import Secrets

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    real_load_all = scoring.load_all

    def without_dotenv():
        """The real config, but with secrets read from the environment alone."""
        return replace(real_load_all(), secrets=Secrets(_env_file=None))

    monkeypatch.setattr(scoring, "load_all", without_dotenv)

    assert main(["score"]) == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "Traceback" not in err


def test_a_cache_warning_names_the_configured_model_not_a_hardcoded_one(capsys, monkeypatch) -> None:
    """The judge model comes from `system_config.yaml`, so the warning must quote that model's minimum."""
    from job_hunters import cli
    from job_hunters.scoring import ScoringReport

    def canned(**_kwargs):
        """A run that judged two texts and never read the cache back."""
        return ScoringReport(judged=2, scored=2, cold_calls=1, model="claude-sonnet-5")

    monkeypatch.setattr(cli, "run_scoring", canned)

    assert main(["score"]) == 0
    err = capsys.readouterr().err
    assert "1,024 tokens for claude-sonnet-5" in err
    assert "Haiku" not in err and "4,096" not in err


def test_a_cache_warning_omits_the_number_for_an_unrecognised_model(capsys, monkeypatch) -> None:
    """A model the table does not know is reported by name, without inventing a minimum for it."""
    from job_hunters import cli
    from job_hunters.scoring import ScoringReport

    def canned(**_kwargs):
        """A run on a model published after this table was last updated."""
        return ScoringReport(judged=2, scored=2, cold_calls=1, model="claude-haiku-9")

    monkeypatch.setattr(cli, "run_scoring", canned)

    assert main(["score"]) == 0
    err = capsys.readouterr().err
    assert "for claude-haiku-9" in err and "tokens for" not in err


def test_eval_scoring_can_report_the_prefilter_alone(capsys) -> None:
    """`eval-scoring --skip-llm` measures the prefilter against the labels without a key."""
    assert main(["eval-scoring", "--skip-llm"]) == 0
    out = capsys.readouterr().out
    assert "prefilter kept" in out and "judge skipped" in out
