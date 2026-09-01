#!/usr/bin/env python3
"""Assemble a prompt-response video from the pre-rendered puppet library.

Takes prompt tokens (directions + digits), maps them onto the pre-rendered
segment library (see puppet_library.py), and concatenates the matching files
with ffmpeg stream-copy.  No neural inference happens here -- assembly takes
about a second regardless of prompt length.

Every library segment starts and ends at the neutral pose, so arbitrary
concatenation is continuous by construction.

Usage:
    puppet_assemble.py --lib /var/lib/deep-live-cam/puppet_lib \
        -o /var/lib/deep-live-cam/renders/puppet_live.mp4 \
        "turn_left" "blink" "say 4-7-2"

With --set-source the assembled video is immediately written to the receiver's
prerecorded-source state file, so the virtual camera switches to it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

IDLE_CHUNKS = ("2", "1", "0.5")  # available idle segment lengths, seconds


def idle_segments(seconds: float, lib: Path) -> list[Path]:
    """Compose an idle hold of ~seconds from the available idle chunks."""
    remaining = max(seconds, 0.0)
    out: list[Path] = []
    while remaining > 0.01:
        for chunk in IDLE_CHUNKS:
            c = float(chunk)
            if c <= remaining + 0.01 or chunk == IDLE_CHUNKS[-1]:
                out.append(lib / f"idle_{chunk}.mp4")
                remaining -= c
                break
    return out


def digits_segments(digits: list[str], lib: Path) -> list[Path]:
    """Per-digit articulation with a short silent gap between digits."""
    out: list[Path] = []
    for i, d in enumerate(digits):
        if i:
            out.append(lib / "idle_0.5.mp4")
        out.append(lib / f"digits_{int(d)}.mp4")
    return out


def resolve_prompt(tokens: list[str], lib: Path) -> list[Path]:
    segments: list[Path] = []
    missing: list[str] = []
    for tok in tokens:
        parts = tok.split()
        name = parts[0]
        if name == "neutral":
            dur = float(parts[-1].rstrip("s")) if len(parts) > 1 else 1.0
            segments += idle_segments(dur, lib)
        elif name in ("turn_left", "turn_right", "look_up", "look_down",
                      "blink"):
            segments.append(lib / f"{name}.mp4")
        elif name == "say":
            digits = parts[1].split("-") if len(parts) >= 2 else []
            if not digits:
                raise SystemExit(f"say requires digits: {tok!r}")
            segments += digits_segments(digits, lib)
        else:
            raise SystemExit(f"unknown prompt token {name!r}")
    for seg in segments:
        if not seg.exists():
            missing.append(seg.name)
    if missing:
        raise SystemExit(f"missing library segments: {missing}")
    return segments


def concat(segments: list[Path], out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for seg in segments:
            f.write(f"file '{seg.resolve()}'\n")
        list_path = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", list_path,
             "-c", "copy", "-movflags", "+faststart", str(out)],
            check=True)
    finally:
        Path(list_path).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="/var/lib/deep-live-cam/puppet_lib")
    ap.add_argument("-o", "--out",
                    default="/var/lib/deep-live-cam/renders/puppet_live.mp4")
    ap.add_argument("--set-source", action="store_true",
                    help="activate the result as the receiver's "
                         "prerecorded source")
    ap.add_argument("prompt", nargs="+")
    args = ap.parse_args()

    t0 = time.time()
    lib = Path(args.lib)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    segments = resolve_prompt(args.prompt, lib)
    duration = 0.0
    for seg in segments:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(seg)], capture_output=True, text=True)
        duration += float(r.stdout.strip())
    print(f"plan: {len(segments)} segments, {duration:.1f}s", flush=True)

    concat(segments, out)
    dt = time.time() - t0
    print(f"assembled {out} in {dt:.2f}s", flush=True)

    if args.set_source:
        source_file = Path("/run/deep-live-cam/prerecorded-source.txt")
        source_file.write_text(str(out))
        print(f"set prerecorded source -> {out}", flush=True)


if __name__ == "__main__":
    main()
