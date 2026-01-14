#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv-wsl" ]]; then
  echo "Virtual env .venv-wsl not found" >&2
  exit 1
fi

source .venv-wsl/bin/activate
export HF_HUB_OFFLINE=1
exec uvicorn service:app --host 0.0.0.0 --port 8013
