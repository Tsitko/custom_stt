#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv-qwen" ]]; then
  echo "Virtual env .venv-qwen not found" >&2
  exit 1
fi

source .venv-qwen/bin/activate

# Offline mode - все модели в кэше
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

exec uvicorn service:app --host 0.0.0.0 --port 8013
