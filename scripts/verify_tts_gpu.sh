#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://localhost:19160}"
REF_TEXT="${REF_TEXT:-The sun set behind the mountains, painting the sky in shades of gold and violet. Birds sang their evening songs as the world grew quiet.}"
INSTRUCT="${INSTRUCT:-A calm male narrator with deep voice}"
TEXT="${TEXT:-Hello, this is a voice cloning test on GPU. The voice should sound similar.}"
LANGUAGE="${LANGUAGE:-english}"
REF_DURATION_SECONDS="${REF_DURATION_SECONDS:-8}"
CLONE_DURATION_SECONDS="${CLONE_DURATION_SECONDS:-20}"
OUT_DIR="${OUT_DIR:-/tmp}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-tts-serve}"
USE_DIRECT="${USE_DIRECT:-0}"
AUTO_REF_TEXT="${AUTO_REF_TEXT:-0}"
SKIP_XV="${SKIP_XV:-0}"
CURL_MAX_TIME="${CURL_MAX_TIME:-300}"

export REF_TEXT
export INSTRUCT
export TEXT
export LANGUAGE
export REF_DURATION_SECONDS
export CLONE_DURATION_SECONDS

VD_MP3="${OUT_DIR}/vd.mp3"
REF_WAV="${OUT_DIR}/ref.wav"
PROMPT_ICL="${OUT_DIR}/prompt_icl.json"
PROMPT_XV="${OUT_DIR}/prompt_xv.json"
OUT_ICL="${OUT_DIR}/out_icl.mp3"
OUT_XV="${OUT_DIR}/out_xv.mp3"

export PROMPT_ICL
export PROMPT_XV

check_audio_response() {
  local file="$1"
  local status_code="$2"
  local label="$3"

  if [ "$status_code" != "200" ]; then
    echo "[error] ${label} failed with HTTP ${status_code}" >&2
    sed -n '1,120p' "$file" >&2 || true
    return 1
  fi

  python3 - "$file" "$label" <<'PY'
import json
import sys

path = sys.argv[1]
label = sys.argv[2]
raw = open(path, 'rb').read()
if not raw:
    print(f"[error] {label} returned empty body", file=sys.stderr)
    raise SystemExit(1)

stripped = raw.lstrip()
if stripped.startswith(b'{') or stripped.startswith(b'['):
    text = raw.decode('utf-8', errors='ignore')
    try:
        payload = json.loads(text)
        print(f"[error] {label} returned JSON body: {payload}", file=sys.stderr)
    except Exception:
        print(f"[error] {label} returned non-audio text body: {text[:400]}", file=sys.stderr)
    raise SystemExit(1)
PY
}

echo "[1/5] VoiceDesign -> ref.wav"
python3 - <<'PY' >"${OUT_DIR}/payload_vd.json"
import json
import os

payload = {
  'text': os.environ.get('REF_TEXT', ''),
  'instruct': os.environ.get('INSTRUCT', ''),
  'language': os.environ.get('LANGUAGE', 'english'),
  'options': {'duration_seconds': int(os.environ.get('REF_DURATION_SECONDS', '20'))}
}
print(json.dumps(payload))
PY

VD_STATUS=$(curl -sS --max-time "$CURL_MAX_TIME" -X POST "$URL/v1/audio/voice-design" \
  -H "Content-Type: application/json" \
  -d "@${OUT_DIR}/payload_vd.json" \
  -o "$VD_MP3" \
  -w "%{http_code}")
check_audio_response "$VD_MP3" "$VD_STATUS" "voice-design"
ffmpeg -y -i "$VD_MP3" -ac 1 -ar 24000 -c:a pcm_s16le "$REF_WAV" >/dev/null 2>&1

if [ "$AUTO_REF_TEXT" = "1" ]; then
  echo "[1/5] Transcribe ref.wav for REF_TEXT"
  docker cp "$REF_WAV" "${CONTAINER_NAME}:/tmp/ref.wav"
  REF_TEXT=$(docker exec -e CUDA_VISIBLE_DEVICES= -i "$CONTAINER_NAME" python3 - <<'PY'
import whisper, warnings
warnings.filterwarnings('ignore')
model = whisper.load_model('tiny')
result = model.transcribe('/tmp/ref.wav', fp16=False)
print(result['text'])
PY
)
  export REF_TEXT
fi

if [ "$USE_DIRECT" = "1" ]; then
  echo "[2/5] VoiceClone synth (direct mode - no prompt roundtrip)"
  ICL_STATUS=$(curl -sS --max-time "$CURL_MAX_TIME" -X POST "$URL/v1/audio/voice-clone" \
    -F "ref_audio=@${REF_WAV}" \
    -F "ref_text=${REF_TEXT}" \
    -F "text=${TEXT}" \
    -F "language=${LANGUAGE}" \
    -F "options={\"duration_seconds\": ${CLONE_DURATION_SECONDS}}" \
    -o "$OUT_ICL" \
    -w "%{http_code}")
  check_audio_response "$OUT_ICL" "$ICL_STATUS" "voice-clone-icl"

  if [ "$SKIP_XV" != "1" ]; then
    XV_STATUS=$(curl -sS --max-time "$CURL_MAX_TIME" -X POST "$URL/v1/audio/voice-clone" \
      -F "ref_audio=@${REF_WAV}" \
      -F "x_vector_only=1" \
      -F "text=${TEXT}" \
      -F "language=${LANGUAGE}" \
      -F "options={\"duration_seconds\": ${CLONE_DURATION_SECONDS}}" \
      -o "$OUT_XV" \
      -w "%{http_code}")
    check_audio_response "$OUT_XV" "$XV_STATUS" "voice-clone-xv"
  fi
else
  echo "[2/5] Create prompts (ICL + x_vector_only)"
  curl -s --max-time "$CURL_MAX_TIME" -X POST "$URL/v1/audio/voice-clone/prompt" \
    -F "ref_audio=@${REF_WAV}" \
    -F "ref_text=${REF_TEXT}" \
    -o "$PROMPT_ICL"

  if [ "$SKIP_XV" != "1" ]; then
    curl -s --max-time "$CURL_MAX_TIME" -X POST "$URL/v1/audio/voice-clone/prompt" \
      -F "ref_audio=@${REF_WAV}" \
      -F "x_vector_only=1" \
      -o "$PROMPT_XV"
  fi

  echo "[3/5] VoiceClone synth"
  python3 - <<'PY' >"${OUT_DIR}/payload_icl.json"
import json
import os

prompt_path = os.environ.get('PROMPT_ICL', '')
with open(prompt_path) as f:
    prompt = json.load(f)['prompt']
payload = {
  'text': os.environ.get('TEXT', ''),
  'language': os.environ.get('LANGUAGE', 'english'),
  'options': {'duration_seconds': int(os.environ.get('CLONE_DURATION_SECONDS', '20'))},
  'prompt': prompt
}
print(json.dumps(payload))
PY

  if [ "$SKIP_XV" != "1" ]; then
    python3 - <<'PY' >"${OUT_DIR}/payload_xv.json"
import json
import os

prompt_path = os.environ.get('PROMPT_XV', '')
with open(prompt_path) as f:
    prompt = json.load(f)['prompt']
payload = {
  'text': os.environ.get('TEXT', ''),
  'language': os.environ.get('LANGUAGE', 'english'),
  'options': {'duration_seconds': int(os.environ.get('CLONE_DURATION_SECONDS', '20'))},
  'prompt': prompt
}
print(json.dumps(payload))
PY
  fi

  ICL_STATUS=$(curl -sS --max-time "$CURL_MAX_TIME" -X POST "$URL/v1/audio/voice-clone/prompted" \
    -H "Content-Type: application/json" \
    -d "@${OUT_DIR}/payload_icl.json" \
    -o "$OUT_ICL" \
    -w "%{http_code}")
  check_audio_response "$OUT_ICL" "$ICL_STATUS" "voice-clone-prompted-icl"

  if [ "$SKIP_XV" != "1" ]; then
    XV_STATUS=$(curl -sS --max-time "$CURL_MAX_TIME" -X POST "$URL/v1/audio/voice-clone/prompted" \
      -H "Content-Type: application/json" \
      -d "@${OUT_DIR}/payload_xv.json" \
      -o "$OUT_XV" \
      -w "%{http_code}")
    check_audio_response "$OUT_XV" "$XV_STATUS" "voice-clone-prompted-xv"
  fi
fi

echo "[4/5] Duration check"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_ICL"
if [ "$SKIP_XV" != "1" ]; then
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_XV"
fi

echo "[5/5] STT (Whisper in container)"
docker cp "$OUT_ICL" "${CONTAINER_NAME}:/tmp/out_icl.mp3"
if [ "$SKIP_XV" != "1" ]; then
  docker cp "$OUT_XV" "${CONTAINER_NAME}:/tmp/out_xv.mp3"
fi

docker exec -e CUDA_VISIBLE_DEVICES= -i "$CONTAINER_NAME" python3 - <<'PY'
import os
import whisper, warnings
warnings.filterwarnings('ignore')
model = whisper.load_model('tiny')
for name in ['out_icl.mp3', 'out_xv.mp3']:
    path = f"/tmp/{name}"
    if not os.path.exists(path):
        continue
    result = model.transcribe(path, fp16=False)
    print(f"{name}: {result['text']}")
PY

echo "Done. Outputs:"
echo "- ${OUT_ICL}"
echo "- ${OUT_XV}"
