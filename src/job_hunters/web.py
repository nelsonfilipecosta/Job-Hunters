"""The web application.

Phase 0 needs exactly one thing from this file: something for the `web`
service in `docker-compose.yml` to run and a `/health` endpoint proving that
process is alive and answering requests. Later phases add real routes here
(the signed action links (Phase 4) and the dashboard (Phase 4)) but nothing
beyond `/health` belongs in this file yet.

Config is validated once at startup rather than on each request. Without that
check the service would start happily on a broken `system_config.yaml` and
report itself healthy, because `/health` never reads config -- a silent
misconfiguration, which is harder to notice than a loud crash.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import paths
from .config import ConfigError, load_all
from .db import init_db

log = logging.getLogger("job_hunters.web")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Validates configuration and creates the schema before serving any request."""
    try:
        load_all()
    except ConfigError as exc:
        log.error("Cannot start: %s", exc)
        raise
    # Whichever of `web` and `scheduler` starts first creates the schema in the
    # otherwise empty Docker volume. Both calls are idempotent and `create_all`
    # issues "create table if not exists", so the two racing is harmless.
    paths.ensure_runtime_dirs()
    init_db()
    yield


app = FastAPI(title="Job-Hunters", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Returns 200 with a small body: proof the process is alive and serving.

    Deliberately a liveness check, not a readiness check. It does not touch the
    database or any external service, only confirms the web process itself can
    accept a request and respond.
    """
    return {"Status": "Ok"}
