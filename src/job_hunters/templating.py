"""The Jinja environment every template in this project is rendered through.

One environment rather than one per caller, because the settings below are safety
settings and a second copy is a second chance to get them wrong. `templates/`
holds both halves of the digest email and the pages the web application serves.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

TEMPLATE_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=1)
def environment() -> Environment:
    """The Jinja environment built once and reused."""
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        # Only `.html` is escaped. The plain-text half of the digest must not
        # be escaped or an ampersand in a job title would reach the reader as `&amp;`.
        autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(template: str, **context) -> str:
    """Renders one template by name with whatever it was given to show."""
    return environment().get_template(template).render(**context)
