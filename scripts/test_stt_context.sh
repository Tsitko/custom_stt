#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-http://127.0.0.1:8013}
AUDIO=${AUDIO:-/home/denis/Projects/custom_tts/chunk_0003.mp3}
CONTEXT=${CONTEXT:-"Тема: детский сад и кружки."}

if [[ ! -f "$AUDIO" ]]; then
  echo "Audio file not found: $AUDIO" >&2
  exit 1
fi

curl -sS --max-time 30 -X POST "$HOST/stt" \
  -F "file=@${AUDIO}" \
  -F "context=${CONTEXT}"

echo
