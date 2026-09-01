#!/usr/bin/env python3
"""Build the puppet segment library from a guided recording.

Pipeline:
  1. face-swap the whole recording with the offline renderer (the tuned
     A-/B+ quality pipeline), if not given an already-swapped video
  2. cut the swapped footage into per-action segments using the cue sheet
     written by the puppet recorder
  3. encode every segment with uniform, concat-safe parameters (keyframes
     every 12 frames, no B-frames) into the library directory

The output library is the direct input of puppet_assemble.py: same segment
names (turn_left, digits_4, idle_2, ...), same concat-safe encoding.

Usage:
  puppet_library_build.py --recording puppet_recording_X.mp4 \
      --cues puppet_recording_X.cues.json --source identity_face.jpg \
      [--out /var/lib/deep-live-cam/puppet_lib] [--quality balanced]
      [--swapped already_swapped.mp4]   # skip step 1 (testing)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BIN = Path(__file__).resolve().parent
DEFAULT_OUT = Path("/var/lib/deep-live-cam/puppet_lib")

# uniform concat-safe encode for every segment.  CRF 14 / slow keeps the
# re-encode from the CRF15 swap render visually lossless -- segments are
# produced once, so spend encode time here, not at playback.
ENC = ["-c:v", "libx264", "-preset", "slow", "-crf", "14",
       "-pix_fmt", "yuv420p", "-g", "12", "-keyint_min", "12",
       "-sc_threshold", "0", "-bf", "0", "-movflags", "+faststart"]

IDLE_CHUNKS = (2.0, 1.0, 0.5)


def log(msg: str) -> None:
    print(msg, flush=True)


def probe_fps(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    num, den = r.stdout.strip().split("/")
    return float(num) / float(den)


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True,
        check=True)
    return float(r.stdout.strip())


def run_swap(recording: Path, face: Path, quality: str,
             work_dir: Path) -> Path:
    """Face-swap the recording with the offline renderer (A-/B+ pipeline)."""
    swapped = work_dir / "swapped.mp4"
    log(f"[swap] offline renderer ({quality}) -> {swapped}")
    cmd = [sys.executable, str(BIN / "offline_renderer.py"),
           str(recording), str(swapped),
           "--face", str(face), "--quality", quality]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(f"[swap] {line}")
    proc.wait()
    if proc.returncode != 0 or not swapped.exists():
        raise SystemExit(f"[swap] renderer failed rc={proc.returncode}")
    return swapped


def cut_segment(src: Path, start: float, duration: float, dst: Path) -> None:
    """Re-encode one segment with the uniform concat-safe parameters."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{max(start, 0.0):.3f}", "-t", f"{duration:.3f}",
           "-i", str(src),
           *ENC, str(dst)]
    subprocess.run(cmd, check=True)


def build_idles(neutral_cues: list[dict], src: Path, out_dir: Path) -> None:
    """Cut the canonical idle chunks out of the longest neutral cue."""
    if not neutral_cues:
        raise SystemExit("[cut] no neutral cues in the session")
    donor = max(neutral_cues, key=lambda c: c["t_end"] - c["t_start"])
    span = donor["t_end"] - donor["t_start"]
    if span < sum(IDLE_CHUNKS):
        raise SystemExit(
            f"[cut] longest neutral cue is {span:.2f}s; need >= "
            f"{sum(IDLE_CHUNKS):.1f}s for idle_2/idle_1/idle_0.5")
    t = donor["t_start"]
    for chunk in IDLE_CHUNKS:
        dst = out_dir / f"idle_{chunk:g}.mp4"
        cut_segment(src, t, chunk, dst)
        log(f"[cut] idle_{chunk:g}.mp4  <- neutral @{t:.2f}s ({chunk:.1f}s)")
        t += chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--cues", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True,
                    help="identity face image for the swap")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quality", default="high",
                    choices=["fast", "balanced", "high", "max", "auto"],
                    help="offline renderer quality for the swap "
                         "(default: high -- one-time cost, best seams)")
    ap.add_argument("--swapped", type=Path, default=None,
                    help="already-swapped video; skips the renderer")
    ap.add_argument("--margin", type=float, default=0.12,
                    help="seconds of context kept on each side of a cue")
    args = ap.parse_args()

    t_start = time.time()
    cues_doc = json.loads(args.cues.read_text(encoding="utf-8"))
    cues = cues_doc["cues"]
    action_cues = [c for c in cues if c["action"] != "neutral"]
    neutral_cues = [c for c in cues if c["action"] == "neutral"]
    log(f"[plan] {len(action_cues)} action cues, "
        f"{len(neutral_cues)} neutral cues")

    work_dir = Path("/var/lib/deep-live-cam/puppet_build_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.swapped is not None:
        swapped = args.swapped
        log(f"[swap] using pre-swapped input {swapped}")
    else:
        swapped = run_swap(args.recording, args.source, args.quality,
                           work_dir)

    fps = probe_fps(swapped)
    total = probe_duration(swapped)
    log(f"[plan] swapped footage {total:.1f}s @ {fps:.2f} fps")

    new_lib = work_dir / "lib_new"
    if new_lib.exists():
        shutil.rmtree(new_lib)
    new_lib.mkdir(parents=True)

    # action segments
    for cue in action_cues:
        start = max(cue["t_start"] - args.margin, 0.0)
        end = min(cue["t_end"] + args.margin, total)
        if end - start < 0.3:
            log(f"[cut] skip {cue['action']} (too short)")
            continue
        dst = new_lib / f"{cue['action']}.mp4"
        cut_segment(swapped, start, end - start, dst)
        log(f"[cut] {dst.name}  <- {start:.2f}s +{end - start:.2f}s")

    # canonical idle chunks from the longest neutral cue
    build_idles(neutral_cues, swapped, new_lib)

    # sanity: verify every segment decodes and has no B-frames
    for seg in sorted(new_lib.glob("*.mp4")):
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=has_b_frames", "-of", "csv=p=0",
             str(seg)], capture_output=True, text=True)
        if r.stdout.strip() != "0":
            raise SystemExit(f"[verify] {seg.name} has B-frames; "
                             "concat contract broken")
    log("[verify] all segments concat-safe (0 B-frames)")

    # atomic-ish install
    out = args.out
    if out.exists():
        backup = out.with_name(out.name + f".bak-{int(t_start)}")
        shutil.move(str(out), str(backup))
        log(f"[install] old library -> {backup.name}")
    shutil.move(str(new_lib), str(out))
    log(f"[install] library -> {out}")

    names = sorted(p.name for p in out.glob("*.mp4"))
    log(f"[done] {len(names)} segments in {(time.time() - t_start) / 60:.1f}min: "
        + ", ".join(names))


if __name__ == "__main__":
    main()
