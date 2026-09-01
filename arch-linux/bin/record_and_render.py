#!/usr/bin/env python3
"""Record a camera source, render it offline with face-swap, and publish as a prerecorded input."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def record_camera(
    device: Path,
    duration: float,
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> None:
    """Record from a V4L2 device to an MP4 file."""
    print(f"[*] Recording from {device} for {duration:.1f}s → {output_path}", flush=True)
    
    # Detect format: try MJPEG first, fall back to YUYV
    input_format = "mjpeg"
    
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
        "-input_format",
        input_format,
        "-framerate",
        str(fps),
        "-video_size",
        f"{width}x{height}",
        "-i",
        str(device),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    
    print(f"[*] Starting recording (format={input_format})...", flush=True)
    result = subprocess.run(command, check=False, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        # Try again with YUYV if MJPEG failed
        if input_format == "mjpeg" and "Invalid argument" in result.stderr:
            print(f"[!] MJPEG not available, retrying with YUYV...", flush=True)
            command[7] = "yuyv422"
            result = subprocess.run(command, check=False)
        
        if result.returncode != 0:
            print(f"[x] Recording failed with exit code {result.returncode}", file=sys.stderr, flush=True)
            sys.exit(1)
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[+] Recording saved to {output_path} ({size_mb:.1f} MB)", flush=True)
    else:
        print(f"[x] Recording failed: output file not created", file=sys.stderr, flush=True)
        sys.exit(1)


def render_offline(
    input_path: Path,
    output_path: Path,
    face_path: Path,
    processing_settings: dict | None = None,
) -> None:
    """Render face-swap with unlimited time per frame via run.py --headless."""
    print(f"[*] Rendering {input_path} → {output_path} (offline, no time constraints)")
    
    # Build run.py command with processing settings
    command = [
        sys.executable,
        "run.py",
        "--headless",
        "-s",
        str(face_path),
        "-t",
        str(input_path),
        "-o",
        str(output_path),
        "--execution-provider",
        "cpu",  # Use CPU for quality, not speed
    ]
    
    # Add processing settings if provided
    if processing_settings:
        for key, value in processing_settings.items():
            # Convert Python keys to CLI flags
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
    
    result = subprocess.run(command, check=False, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        sys.exit(f"[x] Rendering failed with exit code {result.returncode}")
    print(f"[+] Rendered output saved to {output_path}")


def stream_to_receiver(video_path: Path, port: int = 11_010) -> None:
    """Stream the rendered video to the receiver's local_prerecorded port in a loop."""
    print(f"[*] Streaming {video_path} to udp://127.0.0.1:{port} (looping)")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        str(video_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-b:v",
        "5M",
        "-maxrate",
        "5M",
        "-bufsize",
        "5M",
        "-g",
        "30",
        "-bf",
        "0",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        "-flush_packets",
        "1",
        "-bsf:v",
        "dump_extra=freq=keyframe",
        "-f",
        "mpegts",
        f"udp://127.0.0.1:{port}?pkt_size=1316",
    ]
    try:
        subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print("\n[!] Streaming stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record, render offline, and stream as a prerecorded input"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Record subcommand
    record_parser = subparsers.add_parser("record", help="Record from a camera")
    record_parser.add_argument(
        "--device",
        type=Path,
        default=Path("/dev/video0"),
        help="Camera device (default: /dev/video0)",
    )
    record_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10.0)",
    )
    record_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path",
    )
    
    # Render subcommand
    render_parser = subparsers.add_parser("render", help="Render face-swap offline")
    render_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video file",
    )
    render_parser.add_argument(
        "--face",
        type=Path,
        required=True,
        help="Face image for swapping",
    )
    render_parser.add_argument(
        "--output",
        type=Path,
        default=Path("rendered.mp4"),
        help="Output file path (default: rendered.mp4)",
    )
    render_parser.add_argument(
        "--settings",
        type=Path,
        help="JSON file with processing settings",
    )
    
    # Stream subcommand
    stream_parser = subparsers.add_parser("stream", help="Stream to receiver")
    stream_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Video file to stream",
    )
    stream_parser.add_argument(
        "--port",
        type=int,
        default=11_010,
        help="UDP port (default: 11010, receiver's LOCAL_PRERECORDED_PORT)",
    )
    
    # All-in-one subcommand
    all_parser = subparsers.add_parser("all", help="Record, render, and stream")
    all_parser.add_argument(
        "--device",
        type=Path,
        default=Path("/dev/video0"),
        help="Camera device (default: /dev/video0)",
    )
    all_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10.0)",
    )
    all_parser.add_argument(
        "--face",
        type=Path,
        required=True,
        help="Face image for swapping",
    )
    all_parser.add_argument(
        "--settings",
        type=Path,
        help="JSON file with processing settings",
    )
    
    args = parser.parse_args()
    
    if args.command == "record":
        record_camera(args.device, args.duration, args.output)
    
    elif args.command == "render":
        settings = None
        if args.settings:
            with open(args.settings, encoding="utf-8") as f:
                settings = json.load(f)
        render_offline(args.input, args.output, args.face, settings)
    
    elif args.command == "stream":
        stream_to_receiver(args.input, args.port)
    
    elif args.command == "all":
        recorded = Path("recorded_temp.mp4")
        rendered = Path("rendered_temp.mp4")
        
        try:
            record_camera(args.device, args.duration, recorded)
            
            settings = None
            if args.settings:
                with open(args.settings, encoding="utf-8") as f:
                    settings = json.load(f)
            render_offline(recorded, rendered, args.face, settings)
            
            stream_to_receiver(rendered)
        finally:
            # Clean up temp files
            for path in (recorded, rendered):
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    main()
