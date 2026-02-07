#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://localhost:19160}"
OUT_DIR="${OUT_DIR:-/tmp/stress}"
ITERATIONS="${ITERATIONS:-50}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.5}"
MAX_TIME="${MAX_TIME:-180}"
REF_WAV="${REF_WAV:-examples/data/clone_2.wav}"

mkdir -p "$OUT_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

gpu_mem() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true
}

post_voice_design() {
  local label="$1"
  local language="$2"
  local text="$3"
  local instruct="$4"
  local out_mp3="$5"

  STRESS_TEXT="$text" STRESS_INSTRUCT="$instruct" STRESS_LANGUAGE="$language" python3 - <<'PY' >"$OUT_DIR/payload.json"
import json
import os

payload = {
  "text": os.environ.get("STRESS_TEXT", ""),
  "instruct": os.environ.get("STRESS_INSTRUCT", ""),
  "language": os.environ.get("STRESS_LANGUAGE", "english"),
  "options": {"duration_seconds": 15}
}
print(json.dumps(payload))
PY

  local pre_mem
  pre_mem=$(gpu_mem | tr '\n' ';')
  local start
  start=$(date +%s)
  local http_code
  http_code=$(curl -sS --max-time "$MAX_TIME" -w "%{http_code}" -o "$out_mp3" \
    -X POST "$URL/v1/audio/voice-design" \
    -H "Content-Type: application/json" \
    -d "@$OUT_DIR/payload.json" || echo "curl_failed")
  local end
  end=$(date +%s)
  local post_mem
  post_mem=$(gpu_mem | tr '\n' ';')
  local size
  size=$(stat -c%s "$out_mp3" 2>/dev/null || echo 0)

  printf "%s label=%s code=%s sec=%s size=%s pre_mem=[%s] post_mem=[%s]\n" \
    "$(ts)" "$label" "$http_code" "$((end-start))" "$size" "$pre_mem" "$post_mem"
}

post_voice_clone_direct() {
  local label="$1"
  local language="$2"
  local text="$3"
  local out_mp3="$4"

  local pre_mem
  pre_mem=$(gpu_mem | tr '\n' ';')
  local start
  start=$(date +%s)
  local http_code
  http_code=$(curl -sS --max-time "$MAX_TIME" -w "%{http_code}" -o "$out_mp3" \
    -X POST "$URL/v1/audio/voice-clone" \
    -F "ref_audio=@${REF_WAV}" \
    -F "text=${text}" \
    -F "language=${language}" \
    -F 'options={"duration_seconds": 15}' || echo "curl_failed")
  local end
  end=$(date +%s)
  local post_mem
  post_mem=$(gpu_mem | tr '\n' ';')
  local size
  size=$(stat -c%s "$out_mp3" 2>/dev/null || echo 0)

  printf "%s label=%s code=%s sec=%s size=%s pre_mem=[%s] post_mem=[%s]\n" \
    "$(ts)" "$label" "$http_code" "$((end-start))" "$size" "$pre_mem" "$post_mem"
}

echo "$(ts) starting stress: ITERATIONS=$ITERATIONS URL=$URL"

for i in $(seq 1 "$ITERATIONS"); do
  case $((i % 4)) in
    0)
      post_voice_design "en_short" "english" \
        "Hello. This is a short test." \
        "A calm narrator." \
        "$OUT_DIR/vd_en_short_${i}.mp3"
      ;;
    1)
      post_voice_design "en_long" "english" \
        "The sun set behind the mountains, painting the sky in shades of gold and violet. Birds sang their evening songs as the world grew quiet." \
        "A calm male narrator with deep voice" \
        "$OUT_DIR/vd_en_long_${i}.mp3"
      ;;
    2)
      post_voice_design "ko_short" "korean" \
        "안녕하세요. 짧은 테스트입니다." \
        "차분한 남성 화자" \
        "$OUT_DIR/vd_ko_short_${i}.mp3"
      ;;
    3)
      post_voice_design "ko_long" "korean" \
        "오늘은 날씨가 맑고 바람이 선선합니다. 우리는 공원에서 산책을 했습니다." \
        "차분한 남성 화자" \
        "$OUT_DIR/vd_ko_long_${i}.mp3"
      ;;
  esac

  case $((i % 2)) in
    0)
      post_voice_clone_direct "vc_en" "english" \
        "Hello, this is a direct voice clone stress test." \
        "$OUT_DIR/vc_en_${i}.mp3"
      ;;
    1)
      post_voice_clone_direct "vc_ko" "korean" \
        "안녕하세요. 이것은 direct voice clone 스트레스 테스트입니다." \
        "$OUT_DIR/vc_ko_${i}.mp3"
      ;;
  esac

  sleep "$SLEEP_SECONDS"
done

echo "$(ts) done"
