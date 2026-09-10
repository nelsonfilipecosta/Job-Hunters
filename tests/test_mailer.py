"""Tests for delivering the digest by email.

`smtplib` is replaced throughout, so nothing here opens a connection.
"""

from __future__ import annotations

import smtplib

import pytest

from job_hunters.mailer import DeliveryError, Message, RecordingSender, SmtpSender, build_email

MESSAGE = Message(
    to="me@example.com",
    sender="bot@example.com",
    subject="Job-Hunters Wed 09 Sep: 2 roles, 1 priority",
    text="PRIORITY\nResearch Scientist",
    html="<div>Research Scientist</div>",
)


class FakeSMTP:
    """Stands in for `smtplib.SMTP` and records what it was asked to do."""

    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None, context=None) -> None:
        """Records the connection details and registers itself for inspection."""
        self.host, self.port, self.context = host, port, context
        self.started_tls = False
        self.closed = False
        self.login_args: tuple[str, str] | None = None
        self.sent: list = []
        FakeSMTP.instances.append(self)

    def __enter__(self) -> "FakeSMTP":
        """Used as a context manager exactly as `SmtpSender` does."""
        return self

    def __exit__(self, *_exc) -> None:
        """Records that the connection was closed, regardless of how the block ended."""
        self.closed = True

    def starttls(self, context=None) -> None:
        """Records that the connection was upgraded before any credential was sent."""
        self.started_tls = True

    def login(self, username, password) -> None:
        """Records the credentials it was given."""
        self.login_args = (username, password)

    def send_message(self, email) -> None:
        """Records the assembled message."""
        self.sent.append(email)


@pytest.fixture
def smtp(monkeypatch):
    """Replaces both SMTP classes and hands back the list of connections opened."""
    FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP.instances


def _sender(port: int = 587) -> SmtpSender:
    """A sender pointed at a fake server."""
    return SmtpSender(host="smtp.example.com", port=port, username="bot@example.com",
                      password="app-password")


def test_the_message_carries_text_and_html_in_that_order() -> None:
    """Email clients read a multipart/alternative as last-is-best, so HTML must come second."""
    email = build_email(MESSAGE)
    parts = [part.get_content_type() for part in email.walk()]
    assert parts[0] == "multipart/alternative"
    assert parts[1:] == ["text/plain", "text/html"]


def test_the_message_is_addressed_and_marked_automatic() -> None:
    """A vacation responder answering a daily robot email would be its own small disaster."""
    email = build_email(MESSAGE)
    assert email["To"] == "me@example.com"
    assert email["From"] == "bot@example.com"
    assert email["Subject"] == MESSAGE.subject
    assert email["Auto-Submitted"] == "auto-generated"
    assert email["Date"] and email["Message-ID"]


def test_a_send_upgrades_to_tls_before_logging_in(smtp) -> None:
    """A password must never cross an unencrypted connection."""
    _sender(587).send(MESSAGE)
    connection = smtp[0]
    assert connection.started_tls, "STARTTLS on 587"
    assert connection.login_args == ("bot@example.com", "app-password")
    assert len(connection.sent) == 1


def test_port_465_uses_implicit_tls_instead(smtp) -> None:
    """465 is encrypted from the first byte, so there is nothing to upgrade."""
    _sender(465).send(MESSAGE)
    assert not smtp[0].started_tls
    assert smtp[0].context is not None


def test_a_rejected_login_says_what_to_fix(monkeypatch) -> None:
    """Gmail refusing a normal account password is the most likely first failure."""
    FakeSMTP.instances = []

    class Refusing(FakeSMTP):
        def login(self, username, password):
            raise smtplib.SMTPAuthenticationError(535, b"Application-specific password required")

    monkeypatch.setattr(smtplib, "SMTP", Refusing)
    with pytest.raises(DeliveryError) as exc:
        _sender().send(MESSAGE)
    assert "app password" in str(exc.value)
    assert "app-password" not in str(exc.value), "the message must not echo the secret"
    assert FakeSMTP.instances[0].closed, "a refused login still closes the socket"


def test_a_server_that_refuses_to_upgrade_never_sees_the_password(monkeypatch) -> None:
    """A failed STARTTLS has to end the attempt, not fall back to sending in the clear."""
    FakeSMTP.instances = []

    class Downgrading(FakeSMTP):
        def starttls(self, context=None):
            raise smtplib.SMTPNotSupportedError("STARTTLS extension not supported by server")

    monkeypatch.setattr(smtplib, "SMTP", Downgrading)
    with pytest.raises(DeliveryError):
        _sender().send(MESSAGE)
    connection = FakeSMTP.instances[0]
    assert connection.login_args is None and connection.sent == []
    assert connection.closed, "and the socket is closed rather than leaked"


def test_an_unreachable_server_is_a_delivery_error_not_a_traceback(monkeypatch) -> None:
    """A network failure is expected and gets a sentence, not a stack trace from smtplib."""
    def refuse(*_args, **_kwargs):
        raise OSError("Connection refused")

    monkeypatch.setattr(smtplib, "SMTP", refuse)
    with pytest.raises(DeliveryError) as exc:
        _sender().send(MESSAGE)
    assert "smtp.example.com:587" in str(exc.value)


def test_the_recording_sender_delivers_nothing() -> None:
    """What the tests elsewhere in this suite use instead of a mail server."""
    sender = RecordingSender()
    sender.send(MESSAGE)
    assert sender.sent == [MESSAGE]
