#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import wave
from pathlib import Path

import numpy as np


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


def safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s or "case"


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False)


def convert_to_wav(ffmpeg: str, src: Path, dst: Path, sample_rate: int = 24000) -> None:
    out = run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"ffmpeg failed for {src}")


def convert_wav_to_mp3(ffmpeg: str, src_wav: Path, dst_mp3: Path, sample_rate: int = 24000) -> None:
    out = run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_wav),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-codec:a",
            "libmp3lame",
            str(dst_mp3),
        ]
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"ffmpeg mp3 encode failed for {src_wav}")


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 24000) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        channels = w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels)[:, 0]
    return sr, data


def max_shifted_cosine(a: np.ndarray, b: np.ndarray, sr: int, max_shift_sec: float = 0.2) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    max_shift = int(sr * max_shift_sec)
    best = -1.0
    step = max(1, sr // 200)  # 5ms
    for shift in range(-max_shift, max_shift + 1, step):
        if shift >= 0:
            x = a[shift:]
            y = b[: len(x)]
        else:
            x = a[: len(a) + shift]
            y = b[-shift:]
        if len(x) < sr // 4:
            continue
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx == 0.0 or ny == 0.0:
            continue
        score = float(np.dot(x, y) / (nx * ny))
        if score > best:
            best = score
    return max(best, 0.0)


def parse_transcribe_output(stdout: str) -> list[dict]:
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("transcribe output missing JSON payload")
    return json.loads(stdout[start : end + 1])


def transcribe_in_container(container: str, audio_path: Path, model: str, language: str) -> str:
    remote_name = f"leakcheck_{uuid.uuid4().hex}{audio_path.suffix or '.wav'}"
    remote_path = f"/tmp/{remote_name}"

    cp = run(["docker", "cp", str(audio_path), f"{container}:{remote_path}"])
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

    out = run(cmd)
    run(["docker", "exec", container, "rm", "-f", remote_path])
    if out.returncode != 0:
        raise RuntimeError(f"container transcribe failed: {out.stderr.strip()}")

    payload = parse_transcribe_output(out.stdout)
    if not payload:
        return ""
    if payload[0].get("error"):
        raise RuntimeError(payload[0]["error"])
    return str(payload[0].get("transcription", "")).strip()


def make_target_text(case_id: str) -> str:
    return (
        f"이 문장은 음성 누출 검증용 {case_id} 번 테스트입니다. "
        "참조 문장의 끝부분을 반복하지 말고 새로운 내용만 또렷하게 읽어 주세요."
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    examples_file = Path(os.getenv("EXAMPLES_FILE", str(root / "examples/data/examples.txt")))
    out_root = Path(os.getenv("OUT_ROOT", "/tmp/examples_20cases_leakage"))

    ffmpeg = os.getenv("FFMPEG_PATH", "ffmpeg")
    url = os.getenv("URL", "http://localhost:19161")
    container = os.getenv("CONTAINER_NAME", "server-q3t-rust-tts-gpu-0-1")
    max_attempts = os.getenv("MAX_ATTEMPTS", "5")
    sleep_seconds = os.getenv("SLEEP_SECONDS", "1")
    match_threshold = os.getenv("MATCH_THRESHOLD", "0.75")
    min_coverage = os.getenv("MIN_COVERAGE", "0.90")
    ref_duration_seconds = os.getenv("REF_DURATION_SECONDS", "8")
    clone_duration_seconds = os.getenv("CLONE_DURATION_SECONDS", "12")
    whisper_model = os.getenv("WHISPER_MODEL", "tiny")
    whisper_lang = os.getenv("WHISPER_LANGUAGE", "auto")
    segment_seconds = float(os.getenv("SEGMENT_SECONDS", "2.0"))
    leak_cos_threshold = float(os.getenv("LEAK_COS_THRESHOLD", "0.30"))

    listen_root = root / "listen_batch20_leakage_compare"
    ref_full_dir = listen_root / "reference_full"
    clone_full_dir = listen_root / "clone_full"
    ref_tail_dir = listen_root / "reference_tail"
    clone_head_dir = listen_root / "clone_head"
    for d in [ref_full_dir, clone_full_dir, ref_tail_dir, clone_head_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*"):
            if p.is_file():
                p.unlink()

    out_root.mkdir(parents=True, exist_ok=True)
    lines = examples_file.read_text(encoding="utf-8").splitlines()

    rows = [
        "case_id\tname\ttail_head_cos\thead_vs_ref\thead_vs_target\tleak_flag\tstatus\tref_tail_stt\tclone_head_stt"
    ]

    total = 0
    run_fail = 0
    leak_count = 0

    for line in lines:
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) >= 5:
            case_id = parts[0]
            name = parts[2]
            ref_text = parts[3]
            instruct = parts[4]
        elif len(parts) >= 4:
            case_id = parts[0]
            name = parts[1]
            ref_text = parts[2]
            instruct = parts[3]
        else:
            continue
        target_text = make_target_text(case_id)
        total += 1

        case_dir = out_root / f"{case_id}_{safe_name(name)}"
        case_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "URL": url,
                "CONTAINER_NAME": container,
                "OUT_DIR": str(case_dir),
                "LANGUAGE": "korean",
                "USE_DIRECT": "1",
                "SKIP_XV": "1",
                "REF_TEXT": ref_text,
                "TEXT": target_text,
                "INSTRUCT": instruct,
                "MAX_ATTEMPTS": max_attempts,
                "SLEEP_SECONDS": sleep_seconds,
                "MATCH_THRESHOLD": match_threshold,
                "MIN_COVERAGE": min_coverage,
                "REF_DURATION_SECONDS": ref_duration_seconds,
                "CLONE_DURATION_SECONDS": clone_duration_seconds,
            }
        )

        proc = subprocess.run(
            ["bash", "scripts/verify_tts_gpu.sh"],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            run_fail += 1
            rows.append(
                f"{case_id}\t{name}\t0.000\t0.000\t0.000\tUNKNOWN\tRUN_FAIL\t\t"
            )
            print(f"[{case_id}] RUN_FAIL")
            continue

        ref_mp3 = case_dir / "vd.mp3"
        clone_mp3 = case_dir / "out_icl.mp3"
        if not ref_mp3.exists() or not clone_mp3.exists():
            run_fail += 1
            rows.append(
                f"{case_id}\t{name}\t0.000\t0.000\t0.000\tUNKNOWN\tMISSING_OUTPUT\t\t"
            )
            print(f"[{case_id}] MISSING_OUTPUT")
            continue

        ref_wav = case_dir / "ref_full.wav"
        clone_wav = case_dir / "clone_full.wav"
        convert_to_wav(ffmpeg, ref_mp3, ref_wav)
        convert_to_wav(ffmpeg, clone_mp3, clone_wav)

        sr_ref, ref_samples = read_wav(ref_wav)
        sr_clone, clone_samples = read_wav(clone_wav)
        if sr_ref != sr_clone:
            raise RuntimeError(f"sample-rate mismatch: ref={sr_ref}, clone={sr_clone}")

        seg_len = int(sr_ref * segment_seconds)
        if seg_len <= 0:
            seg_len = sr_ref

        ref_tail = ref_samples[-seg_len:] if len(ref_samples) >= seg_len else ref_samples
        clone_head = clone_samples[:seg_len] if len(clone_samples) >= seg_len else clone_samples

        tail_wav = case_dir / "ref_tail.wav"
        head_wav = case_dir / "clone_head.wav"
        write_wav(tail_wav, ref_tail, sr_ref)
        write_wav(head_wav, clone_head, sr_ref)

        tail_head_cos = max_shifted_cosine(ref_tail, clone_head, sr_ref)

        ref_tail_stt = ""
        clone_head_stt = ""
        head_vs_ref = 0.0
        head_vs_target = 0.0
        try:
            ref_tail_stt = transcribe_in_container(container, tail_wav, whisper_model, whisper_lang)
            clone_head_stt = transcribe_in_container(container, head_wav, whisper_model, whisper_lang)
            head_vs_ref = similarity_ratio(ref_text, clone_head_stt)
            head_vs_target = similarity_ratio(target_text, clone_head_stt)
        except Exception as exc:  # noqa: BLE001
            print(f"[{case_id}] STT_FAIL: {exc}")

        leak_flag = tail_head_cos >= leak_cos_threshold and head_vs_ref > (head_vs_target + 0.10)
        if leak_flag:
            leak_count += 1

        status = "PASS" if not leak_flag else "LEAK"
        print(
            f"[{case_id}] cos={tail_head_cos:.3f} head_vs_ref={head_vs_ref:.3f} "
            f"head_vs_target={head_vs_target:.3f} status={status}"
        )

        base = f"{case_id}_{safe_name(name)}"
        shutil.copy2(ref_mp3, ref_full_dir / f"{base}_ref.mp3")
        shutil.copy2(clone_mp3, clone_full_dir / f"{base}_clone.mp3")
        convert_wav_to_mp3(ffmpeg, tail_wav, ref_tail_dir / f"{base}_ref_tail.mp3", 24000)
        convert_wav_to_mp3(ffmpeg, head_wav, clone_head_dir / f"{base}_clone_head.mp3", 24000)

        rows.append(
            f"{case_id}\t{name}\t{tail_head_cos:.3f}\t{head_vs_ref:.3f}\t{head_vs_target:.3f}\t"
            f"{int(leak_flag)}\t{status}\t{ref_tail_stt}\t{clone_head_stt}"
        )

    manifest = listen_root / "manifest.tsv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print("\n=== Leakage result ===")
    print(f"TOTAL: {total}")
    print(f"RUN_FAIL: {run_fail}")
    print(f"LEAK: {leak_count}")
    print(f"PASS: {max(0, total - run_fail - leak_count)}")
    print(f"manifest: {manifest}")
    print(f"listen ref full: {ref_full_dir}")
    print(f"listen clone full: {clone_full_dir}")
    print(f"listen ref tail: {ref_tail_dir}")
    print(f"listen clone head: {clone_head_dir}")

    if run_fail > 0 or leak_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
