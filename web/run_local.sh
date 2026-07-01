#!/usr/bin/env bash
# Run the findmytext demo locally with the repo's virtualenv.
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"

# Optionally point at a custom config: export FINDMYTEXT_CONFIG=/path/to/config.json
exec "${PY}" wsgi.py
