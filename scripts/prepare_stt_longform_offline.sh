#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${GPU_VENV:-$project_root/.venv-stt-gpu}"
python_bin="$venv_path/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "STT venv python not found: $python_bin" >&2
  exit 1
fi

if [[ "${ALLOW_TORCH_UPGRADE:-0}" == "1" ]]; then
  "$python_bin" -m pip install --upgrade webrtcvad >/dev/null
else
  torch_ver="$("$python_bin" - <<'PY'
import importlib.util
if importlib.util.find_spec("torch"):
    import torch
    print(torch.__version__)
PY
)"
  torchaudio_ver="$("$python_bin" - <<'PY'
import importlib.util
if importlib.util.find_spec("torchaudio"):
    import torchaudio
    print(torchaudio.__version__)
PY
)"
  constraints="$(mktemp)"
  if [[ -n "$torch_ver" ]]; then
    echo "torch==$torch_ver" >> "$constraints"
  fi
  if [[ -n "$torchaudio_ver" ]]; then
    echo "torchaudio==$torchaudio_ver" >> "$constraints"
  fi
  "$python_bin" -m pip install webrtcvad -c "$constraints" >/dev/null
  rm -f "$constraints"
fi

export HF_HUB_OFFLINE=1
unset HF_TOKEN

"$python_bin" - <<'PY'
import math
import os
import torch
from stt.vad_segmenter import segment_audio_vad

os.environ["HF_HUB_OFFLINE"] = "1"

sample_rate = 16000
duration = 4.0
t = torch.arange(int(sample_rate * duration), dtype=torch.float32) / sample_rate
tone = (0.2 * torch.sin(2 * math.pi * 440.0 * t)).float()
silence = torch.zeros(int(sample_rate * 1.0), dtype=torch.float32)
audio = torch.cat([silence, tone, silence])
audio_int16 = (audio * 32767.0).to(torch.int16)

segments, bounds = segment_audio_vad(audio_int16, sample_rate)
print(f"Segments: {len(segments)} bounds={bounds}")
PY
