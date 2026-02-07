# AGENTS.md

Repository guide for coding agents working in `server-q3t-fk`.

## 1) Project Summary

- Stack: FastAPI + Qwen3-TTS + NovaSR + nginx + Docker Compose.
- Public API endpoints served through nginx:
  - `POST /voice-design`
  - `POST /voice-clone`
  - `POST /voice-sr`
  - `GET /healthz`
- Main app entrypoint: `app.py`.
- Core model package: `qwen_tts/`.
- Verification scripts: `scripts/`.
- Tests: `tests/`.

## 2) Source Layout

- `app.py`: API routes, service initialization, runtime config.
- `qwen_tts/`: local TTS implementation and tokenizer/model internals.
- `nginx/nginx.conf`: reverse-proxy config for public port.
- `docker-compose.yml`: service topology and runtime env defaults.
- `Dockerfile`: runtime container build definition.
- `data/examples.txt`: reference prompts for evaluation.
- `scripts/verify_pipeline.py`: end-to-end quality verification.
- `tests/test_api_contract.py`: route contract tests.
- `tests/test_examples_dataset.py`: dataset shape validation.

## 3) Build / Run / Test Commands

### Core runtime

- Build + run stack:
  - `docker compose up --build -d`
- Stop stack:
  - `docker compose down`
- View status:
  - `docker compose ps`
- Validate compose syntax:
  - `docker compose config`
- Tail API logs:
  - `docker compose logs -f voice-api`

### Python tests

- Run all tests:
  - `python3 -m pytest -q`
- Run one test file:
  - `python3 -m pytest -q tests/test_api_contract.py`
- Run one test case (single test):
  - `python3 -m pytest -q tests/test_api_contract.py::test_voice_design_contract`
- Run tests matching keyword:
  - `python3 -m pytest -q -k voice_clone`

### End-to-end verification (recommended)

- Full pipeline (local Whisper, API in Docker):
- `python3 scripts/verify_pipeline.py --base-url http://127.0.0.1:19161 --examples data/examples.txt --limit 0 --threshold 0.65 --whisper-device cpu`
- Quick smoke (single case):
- `python3 scripts/verify_pipeline.py --base-url http://127.0.0.1:19161 --examples data/examples.txt --limit 1 --threshold 0.65 --whisper-device cpu`

### Notes on verification scope

- Keep Whisper dependency local (host-side), not required in API container.
- API stack must run in Docker; verification scripts call nginx URL.
- Do not persist generated request/response audio files in app code paths.

## 4) Environment Variables (Important Defaults)

- `HF_TOKEN`: Hugging Face token for model download if needed.
- `NVIDIA_VISIBLE_DEVICES`: target GPU visibility.
- `CUDA_VISIBLE_DEVICES`: app-level CUDA visibility.
- `VOICE_DEVICE`: `auto|cuda|cpu` (compose default: `auto`).
- `VOICE_DTYPE`: `bfloat16|float16|float32`.
- `VOICE_ATTN_IMPL`: optional flash-attn implementation string.
- `NOVASR_DEVICE`: `cpu|cuda|auto` (compose default currently `cpu`).
- `NOVASR_HALF`: `0|1` (compose default currently `0`).
- `HOST_BIND_IP`, `HOST_PORT`: nginx bind target.
- `HF_CACHE_ROOT`: host-mounted Hugging Face cache location.

## 5) Coding Style Guidelines

### Python version/style

- Use Python 3.10+ compatible syntax.
- Prefer explicit type hints for public functions and core helpers.
- Use builtin generic syntax (`dict[str, str]`, `list[int]`) where possible.
- Keep functions small and single-purpose for endpoint helpers.

### Imports

- Group imports in this order:
  1. standard library
  2. third-party packages
  3. local modules
- Avoid wildcard imports.
- Keep import-time side effects minimal.

### Naming conventions

- `snake_case`: functions, variables, module-level helpers.
- `PascalCase`: classes (`Settings`, request models, service classes).
- `UPPER_SNAKE_CASE`: module constants (`VOICE_DESIGN_MODEL_ID`).
- Endpoint handlers should have descriptive verb names (`voice_design`, `voice_clone`, `voice_sr`).

### Formatting

- Follow existing formatting in repository (PEP8-like, 4-space indentation).
- Keep lines readable; break long argument lists vertically.
- Keep route handlers concise and delegate heavy logic to service methods.

### Types and data modeling

- Use Pydantic `BaseModel` for JSON request contracts.
- Use dataclasses for runtime settings/config object groups.
- Keep `Optional[...]` usage explicit where `None` is a valid input.
- Validate and normalize external input early (e.g., text/audio parsing helpers).

### Error handling

- For request validation failures, raise `HTTPException` with explicit status and detail.
- For internal runtime failures, raise explicit exceptions with actionable message.
- Do not swallow exceptions silently.
- Preserve causal stack where useful (`raise ... from exc`).

### API behavior

- Return audio payloads using `StreamingResponse` with `media_type="audio/wav"`.
- Keep endpoint contract stable:
  - `/voice-design` expects JSON body.
  - `/voice-clone` and `/voice-sr` expect multipart uploads.
- Keep `/healthz` lightweight and always available.

### Concurrency and lifecycle

- Keep heavyweight model initialization lazy and thread-safe.
- Reuse singleton-like service instances guarded by lock.
- Avoid repeated model loading per request.

### Audio handling

- Normalize arrays into `float32` and clamp to `[-1.0, 1.0]`.
- Convert multichannel to mono when required by pipeline.
- Decode/encode audio using in-memory buffers (`BytesIO`) in API paths.

## 6) Testing Guidelines

- Prefer fast contract tests using `FastAPI TestClient` and monkeypatched fakes.
- Avoid heavyweight model downloads in unit tests.
- Validate both success and failure contracts (e.g., missing form field -> `400`).
- Keep dataset integrity tests for `data/examples.txt`.
- For regression checks, run end-to-end verification script after API changes.

## 7) Agent Operational Rules for This Repo

- Do not commit `models/` cache or `.env` secrets.
- Respect `.gitignore` and LFS setup in `.gitattributes`.
- If changing endpoint contracts, update:
  - `README.md`
  - tests under `tests/`
  - `scripts/verify_pipeline.py` if flow assumptions change.
- Prefer minimal, targeted changes over broad refactors.
- Preserve Korean dataset and prompt content as-is unless asked.

## 8) Cursor/Copilot Rules Detection

- `.cursorrules`: not found.
- `.cursor/rules/`: not found.
- `.github/copilot-instructions.md`: not found.
- If these are added later, treat them as higher-priority repository policy and merge into this guide.

## 9) Quick Pre-PR Checklist

- `python3 -m pytest -q`
- `docker compose config`
- `docker compose up --build -d` and `docker compose ps` healthy
- `python3 scripts/verify_pipeline.py ... --whisper-device cpu` passes threshold
- Docs updated when contract/runtime behavior changed
