#!/usr/bin/env python3
import os
import re
import sys
import subprocess
from pathlib import Path


def parse_case_line(line: str) -> tuple[str, str, str, str] | None:
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 4:
        raise ValueError(f"invalid case line (expected 4 fields): {line!r}")
    case_id, name, text, instruct = parts[0], parts[1], parts[2], parts[3]
    return case_id, name, text, instruct


def safe_dir_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s or "case"


def estimate_duration_seconds(
    text: str,
    *,
    chars_per_second: float,
    base_seconds: float,
    min_seconds: int,
    max_seconds: int,
) -> int:
    normalized = re.sub(r"\s+", "", text)
    length = len(normalized)
    seconds = int(round((length / chars_per_second) + base_seconds))
    return max(min_seconds, min(max_seconds, seconds))


def resolve_duration(raw: str, text: str, kind: str) -> int:
    value = raw.strip().lower()
    if value in {"auto", "dynamic"}:
        if kind == "ref":
            return estimate_duration_seconds(
                text,
                chars_per_second=5.0,
                base_seconds=2.0,
                min_seconds=8,
                max_seconds=20,
            )
        return estimate_duration_seconds(
            text,
            chars_per_second=5.5,
            base_seconds=2.5,
            min_seconds=6,
            max_seconds=30,
        )
    return int(float(raw))


def main() -> int:
    examples_file = Path(os.getenv("EXAMPLES_FILE", "examples/data/examples.txt"))
    url = os.getenv("URL", "http://localhost:19160")
    container_name = os.getenv("CONTAINER_NAME", "server-q3t-rust-tts-gpu-0-1")
    out_root = Path(os.getenv("OUT_ROOT", "/tmp/examples_20cases"))

    use_direct = os.getenv("USE_DIRECT", "1")
    skip_xv = os.getenv("SKIP_XV", "1")

    ref_duration_seconds = os.getenv("REF_DURATION_SECONDS", "auto")
    clone_duration_seconds = os.getenv("CLONE_DURATION_SECONDS", "auto")

    max_attempts = os.getenv("MAX_ATTEMPTS", "5")
    sleep_seconds = os.getenv("SLEEP_SECONDS", "1")
    match_threshold = os.getenv("MATCH_THRESHOLD", "0.75")
    min_coverage = os.getenv("MIN_COVERAGE", "0.90")

    if not examples_file.exists():
        print(f"examples file not found: {examples_file}", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)

    cases: list[tuple[str, str, str, str]] = []
    for raw in examples_file.read_text(encoding="utf-8").splitlines():
        raw = raw.rstrip("\r")
        parsed = parse_case_line(raw)
        if parsed:
            cases.append(parsed)

    if len(cases) != 20:
        print(f"warning: expected 20 cases, found {len(cases)}")

    print(
        "running examples test with:\n"
        f"- URL={url}\n"
        f"- CONTAINER_NAME={container_name}\n"
        f"- USE_DIRECT={use_direct} SKIP_XV={skip_xv}\n"
        f"- REF_DURATION_SECONDS={ref_duration_seconds} CLONE_DURATION_SECONDS={clone_duration_seconds}\n"
        f"- MAX_ATTEMPTS={max_attempts} SLEEP_SECONDS={sleep_seconds}\n"
        f"- MATCH_THRESHOLD={match_threshold} MIN_COVERAGE={min_coverage}\n"
        f"- OUT_ROOT={out_root}\n"
        f"- EXAMPLES_FILE={examples_file}\n"
    )

    failures: list[str] = []
    for idx, (case_id, name, text, instruct) in enumerate(cases, start=1):
        case_dir = out_root / safe_dir_name(f"{case_id}_{name}")
        case_dir.mkdir(parents=True, exist_ok=True)

        ref_duration_for_case = resolve_duration(ref_duration_seconds, text, "ref")
        clone_duration_for_case = resolve_duration(clone_duration_seconds, text, "clone")

        env = os.environ.copy()
        env.update(
            {
                "URL": url,
                "CONTAINER_NAME": container_name,
                "OUT_DIR": str(case_dir),
                "LANGUAGE": "korean",
                "INSTRUCT": instruct,
                "REF_TEXT": text,
                "TEXT": text,
                "USE_DIRECT": use_direct,
                "SKIP_XV": skip_xv,
                "REF_DURATION_SECONDS": str(ref_duration_for_case),
                "CLONE_DURATION_SECONDS": str(clone_duration_for_case),
                "MAX_ATTEMPTS": max_attempts,
                "SLEEP_SECONDS": sleep_seconds,
                "MATCH_THRESHOLD": match_threshold,
                "MIN_COVERAGE": min_coverage,
            }
        )

        print(
            f"\n[{idx}/{len(cases)}] case {case_id} {name} "
            f"(ref={ref_duration_for_case}s clone={clone_duration_for_case}s)"
        )

        proc = subprocess.run(
            [sys.executable, "scripts/verify_tts_gpu_loop.py"],
            text=True,
            capture_output=True,
            env=env,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            failures.append(f"{case_id} {name}")

    print("\n=== result ===")
    if failures:
        print(f"FAIL ({len(failures)}/{len(cases)})")
        for item in failures:
            print(f"- {item}")
        return 1

    print(f"PASS ({len(cases)}/{len(cases)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
