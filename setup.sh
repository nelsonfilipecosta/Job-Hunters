#!/bin/sh
# Run once after cloning this repository. Safe to re-run at any time.
#
# This is deliberately a living checklist, not a finished installer. It
# documents exactly what a fresh clone needs right now and grows as later
# phases add more. Update it whenever a new one-time step is added.

set -eu
cd "$(dirname "$0")"

# 1. uv itself is a prerequisite, not something this script installs for you.
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not installed." >&2
    echo "       see https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# 2. Create the virtual environment and install every dependency needed.
uv sync

# 3. Create `profile/` and populate it with your personal records.
mkdir -p profile

# 4. Point git at the hooks tracked in `.githooks/`, instead of the untracked
#    per-clone `.git/hooks/` directory. `.git/hooks/` never travels with a clone,
#    so without this step the hooks in `.githooks/` exist in the repo but never
#    actually run. See `.githooks/pre-commit` for what it checks and why.
git config core.hooksPath .githooks

# 5. Create the data directory and the SQLite schema. Note that this is the
#    HOST database. The containers use a separate one in a Docker named volume.
uv run job-hunters init-db

# 6. Build the container images if the Docker daemon is reachable
if docker info >/dev/null 2>&1; then
    echo
    echo "Building container images..."
    docker compose build
    DOCKER_READY=1
else
    DOCKER_READY=0
fi

echo
echo "Setup complete."
echo
if [ "$DOCKER_READY" -eq 1 ]; then
    echo "Start the stack with:  docker compose up -d"
else
    echo "Docker is not running, so images were not built."
    echo "Start Docker Desktop, then:  docker compose build && docker compose up -d"
fi
echo
echo "Still manual, not handled by this script:"
echo
echo "1. Edit the three files in config/ to describe your search:"
echo "     search_profile.yaml"
echo "     system_config.yaml"
echo "     companies_watchlist.yaml"
echo
echo "2. Create .env for your secrets and addresses:  cp .env.example .env"
echo
echo "3. Put your CV in profile/ as markdown (profile/cv.md)."
echo
echo "Then check it worked:"
echo "  uv run job-hunters show-config                 validate config"
echo "  uv run job-hunters ingest                      fill the database"
echo "  uv run job-hunters score --dry-run             what would be judged (free)"
echo "  uv run job-hunters score --limit 20 --verbose  the first paid run"
echo "  uv run job-hunters digest --dry-run > /tmp/digest.html   preview the email"
echo "  uv run job-hunters digest                      send it for real"
