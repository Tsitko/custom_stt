#!/usr/bin/env bash
set -euo pipefail

if [[ ${#} -lt 1 ]]; then
  echo "Usage: $0 <audio_path> [--json] [extra stt_app.py args]" >&2
  echo "Env overrides: STT_DEVICE (default: cuda), STT_MODEL, STT_CONFIG" >&2
  exit 1
fi

audio_path="$1"
shift || true

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stt_python="$project_root/.venv-stt/bin/python"
config_path="${STT_CONFIG:-$project_root/config.yml}"
device="${STT_DEVICE:-cuda}"
model_override="${STT_MODEL:-}"

tmp_cfg="$(mktemp /tmp/stt_gpu_config.XXXX.yml)"
trap 'rm -f "$tmp_cfg"' EXIT

if [[ ! -x "$stt_python" ]]; then
  echo "STT runtime not found: $stt_python" >&2
  exit 2
fi

if [[ ! -f "$config_path" ]]; then
  echo "Config not found: $config_path" >&2
  exit 3
fi

cat "$config_path" > "$tmp_cfg"
{
  echo
  if [[ -n "$model_override" ]]; then
    echo "stt_model: $model_override"
  fi
  echo "stt_device: $device"
} >> "$tmp_cfg"

echo "[test_stt_gpu] audio=$audio_path device=$device config=$tmp_cfg" >&2
"$stt_python" "$project_root/stt_app.py" --audio "$audio_path" --config "$tmp_cfg" "$@"
