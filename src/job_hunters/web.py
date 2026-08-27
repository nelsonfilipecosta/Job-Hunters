"""The web application.

Phase 0 needs exactly one thing from this file: something for the `web`
service in `docker-compose.yml` to run and a `/health` endpoint proving that
process is alive and answering requests. Later phases add real routes here
(the signed action links (Phase 4) and the dashboard (Phase 4)) but nothing
beyond `/health` belongs in this file yet.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Job-Hunters")


@app.get("/health")
def health() -> dict[str, str]:
    """Returns 200 with a small body: proof the process is alive and serving.

    Deliberately a liveness check, not a readiness check. It does not touch the
    database or any external service, only confirms the web process itself can
    accept a request and respond. Enough for Phase 0's Docker gate.
    """
    return {"Status": "Ok"}
