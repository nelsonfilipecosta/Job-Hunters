"""Delivering the digest by email."""

from __future__ import annotations

import logging
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Protocol

log = logging.getLogger("job_hunters.mailer")

# Gmail and most providers speak STARTTLS on 587 and implicit TLS on 465.
IMPLICIT_TLS_PORT = 465
DEFAULT_TIMEOUT_SECONDS = 30.0


class DeliveryError(Exception):
    """The digest was built but could not be handed to the mail server."""


@dataclass(frozen=True)
class Message:
    to: str
    sender: str
    subject: str
    text: str
    html: str


class Sender(Protocol):
    """Anything that can deliver a `Message`."""

    def send(self, message: Message) -> None:
        """Delivers one message or raises `DeliveryError`."""


def build_email(message: Message) -> EmailMessage:
    """Assembles the MIME message: plain text first and HTML as the alternative."""

    email = EmailMessage()
    email["Subject"] = message.subject
    email["From"] = message.sender
    email["To"] = message.to
    email["Date"] = formatdate(localtime=True)
    # Without one, some servers generate their own and threading gets strange.
    email["Message-ID"] = make_msgid(domain="job-hunters.local")
    # Marks the mail as automatic, so a vacation responder does not answer it.
    email["Auto-Submitted"] = "auto-generated"
    email.set_content(message.text)
    email.add_alternative(message.html, subtype="html")
    return email


@dataclass(frozen=True)
class SmtpSender:
    """Delivery over SMTP with TLS chosen from the port."""

    host: str
    port: int
    username: str
    password: str
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def send(self, message: Message) -> None:
        """Connects, authenticates and sends one message."""

        email = build_email(message)
        try:
            with self._connect() as smtp:
                if self.port != IMPLICIT_TLS_PORT:
                    # Credentials must never cross an unencrypted connection.
                    # A server that refuses to upgrade has to fail here rather
                    # than fall back. Done inside the `with` so that a refusal
                    # still closes the socket.
                    smtp.starttls(context=ssl.create_default_context())
                smtp.login(self.username, self.password)
                smtp.send_message(email)
        except smtplib.SMTPAuthenticationError as exc:
            raise DeliveryError(
                f"{self.host} rejected the login for {self.username}: {exc}. "
                f"For Gmail, SMTP_PASSWORD must be a 16-character app password "
                f"(https://myaccount.google.com/apppasswords), not the account password."
            ) from exc
        except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
            raise DeliveryError(
                f"Could not send through {self.host}:{self.port}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        log.info("Digest sent to %s via %s", message.to, self.host)

    def _connect(self) -> smtplib.SMTP:
        """Opens the connection, encrypted from the first byte on port 465."""
        if self.port == IMPLICIT_TLS_PORT:
            return smtplib.SMTP_SSL(
                self.host, self.port, timeout=self.timeout,
                context=ssl.create_default_context(),
            )
        return smtplib.SMTP(self.host, self.port, timeout=self.timeout)


@dataclass
class RecordingSender:
    """A sender that keeps messages instead of delivering them (for tests)."""

    sent: list[Message]

    def __init__(self) -> None:
        """Starts with nothing sent."""
        self.sent = []

    def send(self, message: Message) -> None:
        """Records the message and delivers nothing."""
        self.sent.append(message)
