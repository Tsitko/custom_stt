#!/usr/bin/env bash
set -euo pipefail

if [[ ${#} -lt 1 ]]; then
  echo "Usage: $0 <audio_path> [--json] [extra stt_app.py args]" >&2
  echo "Env overrides: STT_DEVICE (default: cuda), STT_MODEL, STT_CONFIG, GPU_VENV" >&2
  exit 1
fi

audio_path="$1"
shift || true

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${GPU_VENV:-$project_root/.venv-stt-gpu}"
config_path="${STT_CONFIG:-$project_root/config.yml}"
device="${STT_DEVICE:-cuda}"
model_override="${STT_MODEL:-}"
use_nightly="${GPU_TORCH_NIGHTLY:-1}"

tmp_cfg="$(mktemp /tmp/stt_gpu_alt_config.XXXX.yml)"
trap 'rm -f "$tmp_cfg"' EXIT

if [[ ! -f "$config_path" ]]; then
  echo "Config not found: $config_path" >&2
  exit 3
fi

if [[ ! -x "$venv_path/bin/python" ]]; then
  echo "[test_stt_gpu_altvenv] creating venv at $venv_path" >&2
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$venv_path"
  else
    python -m venv "$venv_path"
  fi
fi

if ! "$venv_path/bin/python" - <<'PY' >/dev/null 2>&1; then
import torch  # noqa: F401
import gigaam  # noqa: F401
PY
  "$venv_path/bin/pip" install --upgrade pip wheel
  "$venv_path/bin/pip" install gigaam
fi

if [[ "$device" == "cuda" && "$use_nightly" != "0" ]]; then
  if ! "$venv_path/bin/python" - <<'PY' >/dev/null 2>&1; then
import torch
v = torch.__version__
ok = ("dev" in v) and ("cu128" in v)
raise SystemExit(0 if ok else 1)
PY
    "$venv_path/bin/pip" install --upgrade --pre --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  fi
fi

cat "$config_path" > "$tmp_cfg"
{
  echo
  if [[ -n "$model_override" ]]; then
    echo "stt_model: $model_override"
  fi
  echo "stt_device: $device"
} >> "$tmp_cfg"

echo "[test_stt_gpu_altvenv] audio=$audio_path device=$device venv=$venv_path config=$tmp_cfg" >&2
PYTHONPATH="$project_root" "$venv_path/bin/python" "$project_root/stt_app.py" --audio "$audio_path" --config "$tmp_cfg" "$@"
