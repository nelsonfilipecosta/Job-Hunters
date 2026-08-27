#!/bin/sh
# Run once after cloning this repository. Safe to re-run at any time.
#
# This is deliberately a living checklist, not a finished installer. It
# documents exactly what a fresh clone needs right now and grows as later
# phases add more (Docker, .env, ...). Update it whenever a new one-time step
# is discovered, rather than letting it silently drift out of date.

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

# 3. Point git at the hooks tracked in `.githooks/`, instead of the untracked
#    per-clone `.git/hooks/` directory. `.git/hooks/` never travels with a clone,
#    so without this step the hooks in `.githooks/` exist in the repo but never
#    actually run. See `.githooks/pre-commit` for what it checks and why.
git config core.hooksPath .githooks

# 4. Create the data directory and the SQLite schema.
uv run job-hunters init-db

echo
echo "Setup complete."
echo
echo "Still manual, not yet handled by this script:"
echo "  - create a .env file with: SMTP_PASSWORD, ACTION_TOKEN_SECRET, ANTHROPIC_API_KEY"
echo "  - Docker isn't built yet, so 'docker compose up' isn't available"
