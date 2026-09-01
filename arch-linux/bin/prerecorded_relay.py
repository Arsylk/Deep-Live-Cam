#!/usr/bin/env python3
"""Set the prerecorded video source for the receiver.

Writes the selected video path to the receiver's state file. The receiver's
internal decoder reads the file directly using ffmpeg with -stream_loop.

Usage:
    prerecorded_relay.py video.mp4 [--mode loop|once|freeze]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path


DEFAULT_STATE_FILE = Path("/run/deep-live-cam/prerecorded-source.txt")

# Keep the process alive so the manager can track it and kill it to deactivate.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def relay_video(video_path: Path, mode: str) -> int:
    """Write the video path to the receiver state file and stay alive."""
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 1

    # Write the video path so the receiver picks it up on its next retry.
    try:
        DEFAULT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_STATE_FILE.write_text(str(video_path.resolve()))
    except PermissionError:
        print(f"ERROR: cannot write {DEFAULT_STATE_FILE}", file=sys.stderr)
        return 1

    print(f"[+] {mode}: {video_path.name} → receiver file_relay")

    # Stay alive until killed (manager uses this PID to know relay is active).
    try:
        while True:
            time.sleep(1.0)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Clear the source file on exit so the receiver stops looping.
        try:
            DEFAULT_STATE_FILE.write_text("")
        except (PermissionError, OSError):
            pass

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set prerecorded video source for the receiver",
    )
    parser.add_argument("video", type=Path, help="Video file to replay")
    parser.add_argument(
        "--mode",
        choices=("loop", "once", "freeze"),
        default="loop",
        help="Playback mode (stored for the receiver; default: loop)",
    )
    # Keep --output and --port for backward compat but ignore them.
    parser.add_argument("--output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return relay_video(args.video, args.mode)


if __name__ == "__main__":
    sys.exit(main())
