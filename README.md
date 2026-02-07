# Voice Stack (Qwen3 + NovaSR)

`docker compose up` starts one public API host (nginx) with three endpoints:

- `/voice-design` (Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- `/voice-clone` (Qwen3-TTS-12Hz-1.7B-Base)
- `/voice-sr` (NovaSR super-resolution)

The server only supports those two Qwen models for TTS and keeps request audio in memory.

## Start

```bash
docker compose up --build
```

Default exposed address is `http://127.0.0.1:8080`.

## Environment

- `HF_TOKEN`: optional Hugging Face token
- `NVIDIA_VISIBLE_DEVICES`: GPU id/uuid (default `0`)
- `CUDA_VISIBLE_DEVICES`: app-level GPU visibility (default `0`)
- `VOICE_DEVICE`: model device map (default `cuda`)
- `VOICE_DTYPE`: `bfloat16`, `float16`, or `float32`
- `VOICE_ATTN_IMPL`: optional (`flash_attention_2` only if flash-attn is installed)
- `NOVASR_DEVICE`: `cuda`, `cpu`, or `auto` (default `cpu`)
- `NOVASR_HALF`: `1` or `0`
- `HOST_BIND_IP`: nginx bind IP (default `0.0.0.0`)
- `HOST_PORT`: nginx port (default `8080`)

## API contracts

### `POST /voice-design`

Request JSON:

```json
{
  "text": "안녕하세요",
  "voice_description": "Warm and calm female voice",
  "language": "Auto",
  "max_new_tokens": 2048
}
```

Response: `audio/wav`

### `POST /voice-clone`

`multipart/form-data`

- `reference_audio`: wav bytes (required)
- `target_text`: string (required)
- `reference_text`: string (required unless `x_vector_only_mode=true`)
- `x_vector_only_mode`: `true|false` (default `false`)
- `language`: string (default `Auto`)
- `max_new_tokens`: int (default `2048`)

Response: `audio/wav`

### `POST /voice-sr`

`multipart/form-data`

- `audio`: wav bytes

Response: `audio/wav`

## Tests

Run contract tests:

```bash
pytest -q
```

### End-to-end verification with Whisper tiny

This runs outside Docker against nginx and does not write generated audio to local files.

```bash
python scripts/verify_pipeline.py --base-url http://127.0.0.1:8080 --examples data/examples.txt --limit 5 --threshold 0.65 --whisper-device cpu
```

Pipeline:

1. `voice-design` from `examples.txt`
2. result is reused as clone reference in `voice-clone`
3. clone output is processed by `voice-sr`
4. Whisper tiny transcribes output audio in memory
5. similarity score against original script text is reported
