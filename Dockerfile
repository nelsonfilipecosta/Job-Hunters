# The image `web` and `scheduler` services run from.

FROM python:3.12-slim

# Debian slim ships no timezone database, so TZ would leave the container on UTC
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Lisbon

# Copy the uv binary from its official image rather than pip-installing it.
# It's faster and pins the version explicitly.
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /bin/uv

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependencies are installed BEFORE the source is copied
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY config/ ./config/
RUN uv sync --frozen --no-dev

# Put the virtualenv first on PATH so `uvicorn`, `python` and `job-hunters`
# all resolve to the installed project without needing `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

# `paths.py` reads these to locate config and data. Without them it would fall
# back to walking up from __file__, which happens to work here but would break
# quietly if the layout ever changed
ENV JOB_HUNTERS_ROOT=/app \
    JOB_HUNTERS_CONFIG_DIR=/app/config \
    JOB_HUNTERS_DATA_DIR=/app/data \
    JOB_HUNTERS_PROFILE_DIR=/app/profile \
    JOB_HUNTERS_BACKUP_DIR=/app/backups

EXPOSE 8000

# Overridden per service in `docker-compose.yml`. This default makes the image
# useful on its own (`docker run`), without compose.
CMD ["uvicorn", "job_hunters.web:app", "--host", "0.0.0.0", "--port", "8000"]
