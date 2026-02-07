import gc
import inspect
import io
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

if hasattr(torch.utils, "_pytree"):
    pytree_mod = torch.utils._pytree
    legacy_register = getattr(pytree_mod, "_register_pytree_node", None)
    current_register = getattr(pytree_mod, "register_pytree_node", None)
    if callable(legacy_register):
        def _compat_register_pytree_node(type_, flatten_fn, unflatten_fn, *args, **kwargs):
            fn = getattr(pytree_mod, "_register_pytree_node", None)
            if not callable(fn):
                raise RuntimeError("torch._pytree legacy register function unavailable")
            return fn(type_, flatten_fn, unflatten_fn)

        if not callable(current_register):
            pytree_mod.register_pytree_node = _compat_register_pytree_node

from qwen_tts import Qwen3TTSModel


VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
VOICE_CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


@dataclass
class Settings:
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    voice_device: str = os.getenv("VOICE_DEVICE", "cuda")
    voice_dtype: str = os.getenv("VOICE_DTYPE", "bfloat16")
    voice_cuda_fallback_dtype: str = os.getenv("VOICE_CUDA_FALLBACK_DTYPE", "float32")
    voice_require_cuda: bool = os.getenv("VOICE_REQUIRE_CUDA", "1") == "1"
    voice_allow_cpu_fallback: bool = os.getenv("VOICE_ALLOW_CPU_FALLBACK", "0") == "1"
    voice_attn: Optional[str] = os.getenv("VOICE_ATTN_IMPL")
    novasr_device: str = os.getenv("NOVASR_DEVICE", "cuda")
    novasr_half: bool = os.getenv("NOVASR_HALF", "1") == "1"


def _torch_dtype(value: str) -> torch.dtype:
    lowered = value.strip().lower()
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"float16", "fp16", "half"}:
        return torch.float16
    if lowered in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported VOICE_DTYPE={value}")


def _normalize_audio(wav: np.ndarray) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    max_abs = np.max(np.abs(x)) if x.size else 0.0
    if max_abs > 1.0:
        x = x / max_abs
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def decode_wav_bytes(blob: bytes) -> tuple[np.ndarray, int]:
    try:
        wav, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported audio payload: {exc}") from exc
    return _normalize_audio(wav), int(sr)


def encode_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    out = io.BytesIO()
    sf.write(out, _normalize_audio(wav), sr, format="WAV")
    return out.getvalue()


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


class VoiceDesignRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = "Auto"
    instruct: Optional[str] = None
    voice_description: Optional[str] = None
    max_new_tokens: int = 2048


class TTSService:
    def __init__(self, settings: Settings):
        requested_device = settings.voice_device.strip().lower()
        device = requested_device
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if settings.voice_require_cuda and not device.startswith("cuda"):
            raise RuntimeError(
                f"VOICE_REQUIRE_CUDA=1 but selected VOICE_DEVICE resolves to '{device}'. "
                "Set VOICE_DEVICE=cuda or disable VOICE_REQUIRE_CUDA."
            )

        dtype = _torch_dtype(settings.voice_dtype)
        if device == "cpu":
            dtype = torch.float32

        cuda_fallback_dtype = _torch_dtype(settings.voice_cuda_fallback_dtype)
        if cuda_fallback_dtype == torch.bfloat16:
            cuda_fallback_dtype = torch.float32

        self._voice_design_path = snapshot_download(repo_id=VOICE_DESIGN_MODEL_ID, token=settings.hf_token)
        self._voice_clone_path = snapshot_download(repo_id=VOICE_CLONE_MODEL_ID, token=settings.hf_token)
        self._voice_attn = settings.voice_attn
        self._hf_token = settings.hf_token
        self._runtime_lock = threading.Lock()

        def _load_model(model_path: str, device_map: str, load_dtype: torch.dtype) -> Qwen3TTSModel:
            model_kwargs = {
                "device_map": device_map,
                "dtype": load_dtype,
                "token": self._hf_token,
            }
            if self._voice_attn:
                model_kwargs["attn_implementation"] = self._voice_attn
            return Qwen3TTSModel.from_pretrained(model_path, **model_kwargs)

        def _load(device_map: str, load_dtype: torch.dtype):
            design_model = _load_model(self._voice_design_path, device_map, load_dtype)
            clone_model = _load_model(self._voice_clone_path, device_map, load_dtype)
            return design_model, clone_model

        self._load_model = _load_model

        if device.startswith("cuda"):
            cuda_attempts = [dtype]
            if dtype != cuda_fallback_dtype:
                cuda_attempts.append(cuda_fallback_dtype)

            last_error: Optional[Exception] = None
            for candidate_dtype in cuda_attempts:
                try:
                    self.voice_design, self.voice_clone = _load(device, candidate_dtype)
                    return
                except Exception as exc:
                    last_error = exc

            if settings.voice_allow_cpu_fallback and not settings.voice_require_cuda:
                self.voice_design, self.voice_clone = _load("cpu", torch.float32)
                return

            raise RuntimeError(
                "Failed to load TTS models on CUDA. "
                f"Tried dtypes {[str(d) for d in cuda_attempts]}. "
                "Check GPU free memory, CUDA visibility, and model compatibility."
            ) from last_error

        self.voice_design, self.voice_clone = _load(device, dtype)

    def generate_design(self, text: str, language: str, instruct: str, max_new_tokens: int) -> tuple[np.ndarray, int]:
        wavs, sr = self.voice_design.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            non_streaming_mode=True,
            max_new_tokens=max_new_tokens,
        )
        return wavs[0], int(sr)

    def generate_clone(
        self,
        ref_audio: tuple[np.ndarray, int],
        ref_text: Optional[str],
        target_text: str,
        language: str,
        x_vector_only_mode: bool,
        max_new_tokens: int,
    ) -> tuple[np.ndarray, int]:
        def _run_clone() -> tuple[list[np.ndarray], int]:
            return self.voice_clone.generate_voice_clone(
                text=target_text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
                non_streaming_mode=True,
                max_new_tokens=max_new_tokens,
            )

        try:
            wavs, sr = _run_clone()
        except RuntimeError as exc:
            error_text = str(exc)
            if "replication_pad1d_cuda" not in error_text or "BFloat16" not in error_text:
                raise

            with self._runtime_lock:
                model_param = next(self.voice_clone.model.parameters())
                device_map = "cuda" if str(model_param.device).startswith("cuda") else "cpu"
                if device_map == "cuda" and model_param.dtype == torch.bfloat16:
                    old_clone = self.voice_clone
                    del self.voice_clone
                    del old_clone
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.voice_clone = self._load_model(self._voice_clone_path, device_map, torch.float32)

            wavs, sr = _run_clone()

        return wavs[0], int(sr)


class NovaSRService:
    def __init__(self, settings: Settings):
        from NovaSR import FastSR

        sig = inspect.signature(FastSR)
        target_device = settings.novasr_device.strip().lower()
        if target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {}
        if "half" in sig.parameters:
            kwargs["half"] = settings.novasr_half
        try:
            self.model: Any = FastSR(**kwargs)
        except RuntimeError:
            fallback_kwargs = {}
            if "half" in sig.parameters:
                fallback_kwargs["half"] = False
            self.model = FastSR(**fallback_kwargs)

        if target_device == "cuda" and torch.cuda.is_available():
            try:
                self.model.model = self.model.model.cuda()
                self.model.device = torch.device("cuda")
            except RuntimeError:
                self.model.model = self.model.model.cpu()
                self.model.device = torch.device("cpu")
                self.model.half = False

        if target_device == "cpu":
            self.model.model = self.model.model.cpu()
            self.model.device = torch.device("cpu")
            self.model.half = False
        self.backend = "FastSR"

    def enhance(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if self.model is None:
            raise RuntimeError("NovaSR model failed to initialize")

        source = wav
        if sr != 16000:
            source = librosa.resample(y=source, orig_sr=sr, target_sr=16000).astype(np.float32)

        lowres = torch.from_numpy(source).float().unsqueeze(0)
        lowres = torchaudio.functional.resample(lowres, 16000, 16000, resampling_method="kaiser_window")
        lowres = lowres.unsqueeze(1)

        device = getattr(self.model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        lowres = lowres.to(device)
        if bool(getattr(self.model, "half", False)):
            lowres = lowres.half()

        with torch.no_grad():
            if hasattr(self.model, "infer"):
                output = self.model.infer(lowres)
            elif callable(self.model):
                output = self.model(lowres)
            else:
                raise RuntimeError("NovaSR model does not expose infer/callable API")

        if isinstance(output, tuple):
            output = output[0]
        detach_fn = getattr(output, "detach", None)
        if callable(detach_fn):
            detached = detach_fn()
            cpu_fn = getattr(detached, "cpu", None)
            if callable(cpu_fn):
                detached = cpu_fn()
            numpy_fn = getattr(detached, "numpy", None)
            if callable(numpy_fn):
                output = numpy_fn()
            else:
                output = detached

        enhanced = np.asarray(output, dtype=np.float32)
        if enhanced.ndim > 1:
            enhanced = enhanced[0]
        return _normalize_audio(enhanced), 48000


app = FastAPI(title="Qwen3 Voice API", version="1.0.0")
_settings = Settings()
_tts_service: Optional[TTSService] = None
_sr_service: Optional[NovaSRService] = None
_lock = threading.Lock()


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        with _lock:
            if _tts_service is None:
                _tts_service = TTSService(_settings)
    return _tts_service


def get_sr_service() -> NovaSRService:
    global _sr_service
    if _sr_service is None:
        with _lock:
            if _sr_service is None:
                _sr_service = NovaSRService(_settings)
    return _sr_service


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/voice-design")
def voice_design(payload: VoiceDesignRequest):
    instruction = (payload.instruct or payload.voice_description or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="Either 'instruct' or 'voice_description' is required")

    wav, sr = get_tts_service().generate_design(
        text=payload.text.strip(),
        language=payload.language,
        instruct=instruction,
        max_new_tokens=payload.max_new_tokens,
    )
    return StreamingResponse(io.BytesIO(encode_wav_bytes(wav, sr)), media_type="audio/wav")


@app.post("/voice-clone")
async def voice_clone(
    reference_audio: UploadFile = File(...),
    target_text: str = Form(...),
    language: str = Form("Auto"),
    reference_text: Optional[str] = Form(None),
    x_vector_only_mode: str = Form("false"),
    max_new_tokens: int = Form(2048),
):
    use_xvector = _parse_bool(x_vector_only_mode)
    if not use_xvector and not (reference_text and reference_text.strip()):
        raise HTTPException(status_code=400, detail="reference_text is required unless x_vector_only_mode=true")

    ref_wav, ref_sr = decode_wav_bytes(await reference_audio.read())
    wav, sr = get_tts_service().generate_clone(
        ref_audio=(ref_wav, ref_sr),
        ref_text=reference_text.strip() if reference_text else None,
        target_text=target_text.strip(),
        language=language,
        x_vector_only_mode=use_xvector,
        max_new_tokens=max_new_tokens,
    )
    return StreamingResponse(io.BytesIO(encode_wav_bytes(wav, sr)), media_type="audio/wav")


@app.post("/voice-sr")
async def voice_sr(audio: UploadFile = File(...)):
    wav, sr = decode_wav_bytes(await audio.read())
    enhanced, enhanced_sr = get_sr_service().enhance(wav, sr)
    return StreamingResponse(io.BytesIO(encode_wav_bytes(enhanced, enhanced_sr)), media_type="audio/wav")
