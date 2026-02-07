import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("torchaudio")

import app


def _wav_bytes(seconds: float = 0.2, sr: int = 16000) -> bytes:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    wav = 0.1 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    out = io.BytesIO()
    sf.write(out, wav, sr, format="WAV")
    return out.getvalue()


class _FakeTTS:
    def generate_design(self, text, language, instruct, max_new_tokens):
        return np.zeros(8000, dtype=np.float32), 16000

    def generate_clone(self, ref_audio, ref_text, target_text, language, x_vector_only_mode, max_new_tokens):
        return np.zeros(16000, dtype=np.float32), 16000


class _FakeSR:
    def enhance(self, wav, sr):
        return wav, 48000


def test_voice_design_contract(monkeypatch):
    monkeypatch.setattr(app, "get_tts_service", lambda: _FakeTTS())
    client = TestClient(app.app)
    resp = client.post(
        "/voice-design",
        json={"text": "hello", "voice_description": "calm voice", "language": "Auto"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    wav, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    assert sr == 16000
    assert wav.size > 0


def test_voice_clone_contract(monkeypatch):
    monkeypatch.setattr(app, "get_tts_service", lambda: _FakeTTS())
    client = TestClient(app.app)
    resp = client.post(
        "/voice-clone",
        files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        data={
            "target_text": "target",
            "reference_text": "reference",
            "language": "Auto",
            "x_vector_only_mode": "false",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")


def test_voice_sr_contract(monkeypatch):
    monkeypatch.setattr(app, "get_sr_service", lambda: _FakeSR())
    client = TestClient(app.app)
    resp = client.post(
        "/voice-sr",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200
    wav, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    assert sr == 48000
    assert wav.size > 0


def test_voice_clone_requires_ref_text(monkeypatch):
    monkeypatch.setattr(app, "get_tts_service", lambda: _FakeTTS())
    client = TestClient(app.app)
    resp = client.post(
        "/voice-clone",
        files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        data={"target_text": "target", "x_vector_only_mode": "false"},
    )
    assert resp.status_code == 400
