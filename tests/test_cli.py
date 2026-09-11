"""Tests for the command-line interface."""

from __future__ import annotations

from dataclasses import replace

import pytest

from job_hunters import paths
from job_hunters.cli import main
from job_hunters.config import Secrets, load_all


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


@pytest.mark.parametrize("limit", ["-2", "0"])
def test_a_limit_below_one_is_refused_before_anything_is_judged(limit: str, capsys) -> None:
    """A negative limit would slice from the end and judge nearly everything, so argparse refuses it."""
    with pytest.raises(SystemExit) as exc:
        main(["score", "--limit", limit])
    assert exc.value.code != 0
    assert "1 or more" in capsys.readouterr().err


def test_the_labeled_fixture_is_where_the_container_will_look_for_it() -> None:
    """`eval-scoring` reads this path and the Dockerfile copies exactly it into the image."""
    from job_hunters.evaluate import DEFAULT_LABELS_PATH

    assert DEFAULT_LABELS_PATH.is_file()
    relative = DEFAULT_LABELS_PATH.relative_to(paths.PROJECT_ROOT).as_posix()
    dockerfile = (paths.PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert relative in dockerfile, f"The Dockerfile must copy {relative} for the container to evaluate."


def test_digest_dry_run_writes_html_to_stdout_and_the_summary_to_stderr(
    session, monkeypatch, capsys
) -> None:
    """`digest --dry-run > file.html` has to produce a file a browser can open."""
    monkeypatch.setenv("ACTION_TOKEN_SECRET", "a-secret")

    assert main(["digest", "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.out.lstrip().startswith("<div")
    assert "Dry run" in captured.err and "Priority" in captured.err
    assert "<div" not in captured.err, "the summary must not pollute the redirected html"


def test_digest_without_a_signing_secret_fails_by_name(session, monkeypatch, capsys) -> None:
    """Unsigned links could be forged, so a missing secret stops the run by variable name."""
    monkeypatch.delenv("ACTION_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(
        "job_hunters.digest.load_all",
        lambda: replace(load_all(), secrets=Secrets(_env_file=None)),
    )

    assert main(["digest", "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "ACTION_TOKEN_SECRET" in err and "Traceback" not in err


def test_a_delivery_failure_exits_1_without_a_traceback(capsys, monkeypatch) -> None:
    """A mail server saying no is expected and gets a sentence rather than a stack trace."""
    from job_hunters.mailer import DeliveryError

    def _raise(*_args, **_kwargs):
        raise DeliveryError("Could not send through smtp.example.com:587")

    monkeypatch.setattr("job_hunters.cli.run_digest", _raise)

    assert main(["digest"]) == 1
    captured = capsys.readouterr()
    assert "smtp.example.com:587" in captured.err
    assert "Traceback" not in captured.err


def test_backup_writes_a_copy_and_says_where(session, tmp_path, capsys, monkeypatch) -> None:
    """`job-hunters backup` has to name the file or you cannot tell it ever ran."""
    from job_hunters import backup as backup_module
    from job_hunters import cli

    real = backup_module.backup_database
    monkeypatch.setattr(
        cli, "backup_database", lambda: real(tmp_path / "backups")
    )

    assert main(["backup"]) == 0
    out = capsys.readouterr().out
    assert "Backed up" in out
    assert "job_hunters-" in out
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1


def test_a_backup_that_could_not_be_written_exits_1_without_a_traceback(
    capsys, monkeypatch
) -> None:
    """A silent failure here would only be discover when you needed it."""
    from job_hunters.backup import BackupError

    def _raise(*_args, **_kwargs):
        raise BackupError("There is no database at `data/job_hunters.db` to back up.")

    monkeypatch.setattr("job_hunters.cli.backup_database", _raise)

    assert main(["backup"]) == 1
    captured = capsys.readouterr()
    assert "no database" in captured.err
    assert "Traceback" not in captured.err


def test_eval_scoring_can_report_the_prefilter_alone(capsys) -> None:
    """`eval-scoring --skip-llm` measures the prefilter against the labels without a key."""
    assert main(["eval-scoring", "--skip-llm"]) == 0
    out = capsys.readouterr().out
    assert "prefilter kept" in out and "judge skipped" in out
