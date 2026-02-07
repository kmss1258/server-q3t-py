#!/usr/bin/env python3
import argparse
import difflib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
import whisper


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s가-힣]", "", lowered)
    return lowered


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_normalize_text(a), b=_normalize_text(b)).ratio()


def _parse_examples(path: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 4:
            continue
        rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()))
    return rows


def _as_wav_bytes(resp: requests.Response) -> bytes:
    resp.raise_for_status()
    return resp.content


def _transcribe_bytes(model, wav_bytes: bytes) -> str:
    wav, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    if sr != 16000:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=16000)
    result = model.transcribe(np.asarray(wav, dtype=np.float32), language="ko")
    return result.get("text", "")


@dataclass
class CaseResult:
    case_id: str
    title: str
    score: float
    reference_text: str
    transcribed_text: str


def run(base_url: str, examples_path: Path, limit: int, threshold: float, whisper_device: str) -> int:
    rows = _parse_examples(examples_path)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("No valid rows found in examples.txt")

    model = whisper.load_model("tiny", in_memory=True, device=whisper_device)
    results: list[CaseResult] = []

    for _, case_id, script_text, design_prompt in rows:
        design_resp = requests.post(
            f"{base_url}/voice-design",
            json={"text": script_text, "voice_description": design_prompt, "language": "Auto"},
            timeout=600,
        )
        design_wav = _as_wav_bytes(design_resp)

        clone_resp = requests.post(
            f"{base_url}/voice-clone",
            files={"reference_audio": ("design.wav", design_wav, "audio/wav")},
            data={
                "target_text": script_text,
                "reference_text": script_text,
                "language": "Auto",
                "x_vector_only_mode": "false",
            },
            timeout=600,
        )
        clone_wav = _as_wav_bytes(clone_resp)

        sr_resp = requests.post(
            f"{base_url}/voice-sr",
            files={"audio": ("clone.wav", clone_wav, "audio/wav")},
            timeout=600,
        )
        sr_wav = _as_wav_bytes(sr_resp)

        transcribed = _transcribe_bytes(model, sr_wav)
        score = _similarity(script_text, transcribed)
        results.append(CaseResult(case_id=case_id, title=design_prompt, score=score, reference_text=script_text, transcribed_text=transcribed))

    scores = [r.score for r in results]
    avg_score = float(sum(scores) / len(scores))
    min_score = float(min(scores))
    passed = avg_score >= threshold and min_score >= threshold * 0.75

    payload = {
        "count": len(results),
        "threshold": threshold,
        "avg_score": avg_score,
        "min_score": min_score,
        "passed": passed,
        "results": [r.__dict__ for r in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end verification: voice-design -> voice-clone -> voice-sr -> whisper tiny")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Public nginx URL")
    parser.add_argument("--examples", default="data/examples.txt", help="Path to examples file")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to test (0=all)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Average similarity threshold")
    parser.add_argument("--whisper-device", default=os.getenv("WHISPER_DEVICE", "cpu"), help="Whisper device (cpu or cuda)")
    args = parser.parse_args()

    return run(
        base_url=args.base_url.rstrip("/"),
        examples_path=Path(args.examples),
        limit=args.limit,
        threshold=args.threshold,
        whisper_device=args.whisper_device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
