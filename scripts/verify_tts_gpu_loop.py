#!/usr/bin/env python3
import os
import re
import sys
import time
import subprocess


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.lower())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # O(min(n, m)) memory DP
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
        coverage = len(act) / len(exp)
        # Only treat substring as match if most of expected is present.
        return min(coverage, 1.0)

    dist = levenshtein(exp, act)
    return 1.0 - (dist / max(len(exp), len(act)))


def main() -> int:
    max_attempts = int(os.getenv("MAX_ATTEMPTS", "20"))
    sleep_seconds = float(os.getenv("SLEEP_SECONDS", "2"))
    match_threshold = float(os.getenv("MATCH_THRESHOLD", "1.0"))
    min_coverage = float(os.getenv("MIN_COVERAGE", "0.9"))

    target_text = os.getenv("TARGET_TEXT")
    if not target_text:
        target_text = os.getenv("TEXT")
    if not target_text:
        target_text = (
            "Hello, this is a voice cloning test on GPU. The voice should sound similar."
        )
    if not target_text:
        target_text = os.getenv(
            "REF_TEXT",
            "The sun set behind the mountains, painting the sky in shades of gold and violet. "
            "Birds sang their evening songs as the world grew quiet.",
        )

    env = os.environ.copy()

    for attempt in range(1, max_attempts + 1):
        print(f"[attempt {attempt}/{max_attempts}] Running verify_tts_gpu.sh")
        proc = subprocess.run(
            ["bash", "scripts/verify_tts_gpu.sh"],
            capture_output=True,
            text=True,
            env=env,
        )

        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            print("verify_tts_gpu.sh failed; stopping.", file=sys.stderr)
            return proc.returncode

        icl_match = re.search(r"out_icl\.mp3:\s*(.*)", proc.stdout)
        xv_match = re.search(r"out_xv\.mp3:\s*(.*)", proc.stdout)

        icl_text = icl_match.group(1).strip() if icl_match else ""
        xv_text = xv_match.group(1).strip() if xv_match else ""

        if not icl_text:
            print("ICL STT text not found; retrying.")
        else:
            score = similarity_ratio(target_text, icl_text)

            # If actual is a substring of expected, require sufficient coverage.
            exp_n = normalize(target_text)
            act_n = normalize(icl_text)
            if act_n and exp_n and act_n in exp_n and (len(act_n) / len(exp_n)) < min_coverage:
                score = min(score, 0.0)

            if score >= match_threshold:
                print(f"ICL STT matches target text (score={score:.3f}).")
                if xv_text:
                    print(f"x_vector_only STT: {xv_text}")
                return 0

            print("ICL STT mismatch; retrying.")
            print(f"Target: {target_text}")
            print(f"ICL:    {icl_text}")
            print(f"score:  {score:.3f}")
            if xv_text:
                print(f"xv:     {xv_text}")

        if attempt < max_attempts:
            time.sleep(sleep_seconds)

    print("Max attempts reached without an ICL match.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
