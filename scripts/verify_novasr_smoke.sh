#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://localhost:19161}"
OUT_DIR="${OUT_DIR:-/tmp/novasr_smoke}"
FFMPEG_PATH="${FFMPEG_PATH:-ffmpeg}"

mkdir -p "$OUT_DIR"

if [ ! -f "/tmp/examples_20cases_final/00001_____20____ASMR/out_icl.mp3" ]; then
  echo "[error] missing source file: /tmp/examples_20cases_final/00001_____20____ASMR/out_icl.mp3" >&2
  echo "Run scripts/test_examples_20cases.py first." >&2
  exit 1
fi

SRC="/tmp/examples_20cases_final/00001_____20____ASMR/out_icl.mp3"
SR_WAV="$OUT_DIR/sr.wav"
SR_MP3="$OUT_DIR/sr.mp3"

curl -sS -X POST "$URL/v1/audio/super-resolve" \
  -F "audio=@${SRC}" \
  -F "output_format=wav" \
  -o "$SR_WAV"

"$FFMPEG_PATH" -y -hide_banner -loglevel error -i "$SR_WAV" -ac 1 -ar 48000 -codec:a libmp3lame "$SR_MP3"

echo "[ok] generated: $SR_WAV"
echo "[ok] generated: $SR_MP3"
