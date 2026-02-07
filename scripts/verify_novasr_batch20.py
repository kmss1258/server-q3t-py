#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import requests


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.lower())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def similarity_ratio(expected: str, actual: str) -> float:
    exp = normalize(expected)
    act = normalize(actual)
    if not exp or not act:
        return 0.0
    if exp in act:
        return 1.0
    if act in exp:
        return min(len(act) / len(exp), 1.0)
    dist = levenshtein(exp, act)
    return 1.0 - (dist / max(len(exp), len(act)))


def run_transcribe_local(audio_path: Path, transcribe_path: str, model: str, language: str) -> str:
    cmd = ["python3", transcribe_path, "--model", model]
    if language and language != "auto":
        cmd += ["--language", language]
    cmd += ["--json", str(audio_path)]

    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"transcribe failed: {out.stderr.strip()}")
    payload = _parse_transcribe_output(out.stdout)
    if not payload:
        return ""
    if payload[0].get("error"):
        raise RuntimeError(f"transcribe error: {payload[0]['error']}")
    return str(payload[0].get("transcription", "")).strip()


def run_transcribe_container(audio_path: Path, container: str, model: str, language: str) -> str:
    remote_name = f"novasr_stt_{uuid.uuid4().hex}{audio_path.suffix or '.wav'}"
    remote_path = f"/tmp/{remote_name}"

    cp = subprocess.run(
        ["docker", "cp", str(audio_path), f"{container}:{remote_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"docker cp failed: {cp.stderr.strip()}")

    cmd = [
        "docker",
        "exec",
        container,
        "python3",
        "/usr/local/bin/transcribe.py",
        "--model",
        model,
    ]
    if language and language != "auto":
        cmd += ["--language", language]
    cmd += ["--json", remote_path]

    out = subprocess.run(cmd, capture_output=True, text=True, check=False)

    subprocess.run(
        ["docker", "exec", container, "rm", "-f", remote_path],
        capture_output=True,
        text=True,
        check=False,
    )

    if out.returncode != 0:
        raise RuntimeError(f"container transcribe failed: {out.stderr.strip()}")

    payload = _parse_transcribe_output(out.stdout)
    if not payload:
        return ""
    if payload[0].get("error"):
        raise RuntimeError(f"transcribe error: {payload[0]['error']}")
    return str(payload[0].get("transcription", "")).strip()


def _parse_transcribe_output(stdout: str) -> list[dict]:
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("transcribe output missing JSON payload")
    blob = stdout[start : end + 1]
    return json.loads(blob)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    examples_file = Path(os.getenv("EXAMPLES_FILE", str(root / "examples/data/examples.txt")))
    src_root = Path(os.getenv("SRC_ROOT", "/tmp/examples_20cases_final"))
    out_root = Path(os.getenv("OUT_ROOT", "/tmp/examples_20cases_sr"))
    url = os.getenv("SR_URL", "http://localhost:19161/v1/audio/super-resolve")
    ffmpeg = os.getenv("FFMPEG_PATH", "ffmpeg")
    transcribe_path = os.getenv("TRANSCRIBE_PATH", str(root / "scripts/transcribe.py"))
    stt_container = os.getenv("STT_CONTAINER", "server-q3t-rust-tts-gpu-0-1")
    whisper_model = os.getenv("WHISPER_MODEL", "tiny")
    whisper_language = os.getenv("WHISPER_LANGUAGE", "auto")
    threshold = float(os.getenv("MATCH_THRESHOLD", "0.75"))
    min_coverage = float(os.getenv("MIN_COVERAGE", "0.90"))

    listen_root = root / "listen_batch20_sr_compare"
    orig_dir = listen_root / "original"
    sr_dir = listen_root / "sr_novasr"
    orig_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)
    for p in list(orig_dir.glob("*")) + list(sr_dir.glob("*")):
        if p.is_file():
            p.unlink()

    out_root.mkdir(parents=True, exist_ok=True)

    rows = ["case_id\tname\toriginal\tsr\tscore\tstatus"]
    pass_count = 0
    total = 0

    lines = examples_file.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 4:
            continue
        case_id, name, text = parts[0], parts[1], parts[2]
        total += 1

        matches = sorted([d for d in src_root.iterdir() if d.is_dir() and d.name.startswith(case_id + "_")])
        if not matches:
            print(f"[{case_id}] missing source directory")
            rows.append(f"{case_id}\t{name}\tMISSING\tMISSING\t0.000\tFAIL")
            continue

        case_dir = matches[0]
        source = case_dir / "out_icl.mp3"
        if not source.exists():
            print(f"[{case_id}] missing source audio: {source}")
            rows.append(f"{case_id}\t{name}\tMISSING\tMISSING\t0.000\tFAIL")
            continue

        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        sr_wav = out_root / f"{case_id}_{safe_name}_sr.wav"
        with source.open("rb") as fh:
            resp = requests.post(
                url,
                files={"audio": (source.name, fh, "audio/mpeg")},
                data={"output_format": "wav"},
                timeout=180,
            )

        if resp.status_code != 200:
            print(f"[{case_id}] SR API failed: HTTP {resp.status_code} {resp.text[:160]}")
            rows.append(f"{case_id}\t{name}\t{source.name}\tFAILED\t0.000\tFAIL")
            continue

        sr_wav.write_bytes(resp.content)

        sr_mp3 = sr_dir / f"{case_id}_{safe_name}_sr.mp3"
        src_copy = orig_dir / f"{case_id}_{safe_name}_orig.mp3"
        shutil.copy2(source, src_copy)

        ff = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(sr_wav),
                "-ac",
                "1",
                "-ar",
                "48000",
                "-codec:a",
                "libmp3lame",
                str(sr_mp3),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if ff.returncode != 0:
            print(f"[{case_id}] ffmpeg encode failed: {ff.stderr.strip()}")
            rows.append(f"{case_id}\t{name}\t{src_copy.name}\tFAILED\t0.000\tFAIL")
            continue

        try:
            if stt_container:
                sr_text = run_transcribe_container(sr_wav, stt_container, whisper_model, whisper_language)
            else:
                sr_text = run_transcribe_local(sr_wav, transcribe_path, whisper_model, whisper_language)
        except Exception as exc:  # noqa: BLE001
            print(f"[{case_id}] transcribe failed: {exc}")
            rows.append(f"{case_id}\t{name}\t{src_copy.name}\t{sr_mp3.name}\t0.000\tFAIL")
            continue

        score = similarity_ratio(text, sr_text)
        exp_n = normalize(text)
        act_n = normalize(sr_text)
        if act_n and exp_n and act_n in exp_n and (len(act_n) / len(exp_n)) < min_coverage:
            score = min(score, 0.0)

        status = "PASS" if score >= threshold else "FAIL"
        if status == "PASS":
            pass_count += 1

        print(f"[{case_id}] score={score:.3f} status={status}")
        print(f"  expected: {text}")
        print(f"  sr_stt:   {sr_text}")
        rows.append(f"{case_id}\t{name}\t{src_copy.name}\t{sr_mp3.name}\t{score:.3f}\t{status}")

    manifest = listen_root / "manifest.tsv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print("\n=== SR result ===")
    print(f"PASS ({pass_count}/{total})")
    print(f"manifest: {manifest}")
    print(f"listen original: {orig_dir}")
    print(f"listen sr: {sr_dir}")

    return 0 if pass_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
