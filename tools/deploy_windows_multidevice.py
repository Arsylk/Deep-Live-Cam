#!/usr/bin/env python3
"""Stage, verify, and optionally activate the Windows five-slot overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from winhash_client import scp_to, ssh_exec


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "windows"
HOST = "192.168.1.35"
USER = "Zuzia"
KEY = "/home/arsylk/.ssh/id_ed25519"
REMOTE_ROOT = r"C:\Deep-Live-Cam"
REMOTE_STAGE = REMOTE_ROOT + r"\runtime\multidevice-stage"
FILES = {
    "activate-multidevice.ps1": WINDOWS / "activate-multidevice.ps1",
    "run_network.py": WINDOWS / "run_network.py",
    "run-network-service.cmd": WINDOWS / "run-network-service.cmd",
    "run.py": ROOT / "run.py",
    r"modules\device_slots.py": WINDOWS / "modules" / "device_slots.py",
    r"modules\network_router.py": WINDOWS / "modules" / "network_router.py",
    r"modules\live_stream.py": WINDOWS / "modules" / "live_stream.py",
    r"modules\live_processor.py": WINDOWS / "modules" / "live_processor.py",
    r"modules\remote_control.py": WINDOWS / "modules" / "remote_control.py",
    r"modules\globals.py": ROOT / "modules" / "globals.py",
    r"modules\face_analyser.py": ROOT / "modules" / "face_analyser.py",
    r"modules\face_tracking.py": ROOT / "modules" / "face_tracking.py",
    r"modules\quality_pipeline.py": ROOT / "modules" / "quality_pipeline.py",
    r"modules\pipeline_benchmark.py": ROOT / "modules" / "pipeline_benchmark.py",
    r"modules\swapper_contract.py": ROOT / "modules" / "swapper_contract.py",
    r"modules\instyle256_swapper.py": ROOT / "modules" / "instyle256_swapper.py",
    r"modules\simswap512_swapper.py": ROOT / "modules" / "simswap512_swapper.py",
    r"modules\processors\frame\face_swapper.py": (
        ROOT / "modules" / "processors" / "frame" / "face_swapper.py"
    ),
    r"modules\processors\frame\frequency_repair.py": (
        ROOT / "modules" / "processors" / "frame" / "frequency_repair.py"
    ),
    r"modules\processors\frame\boundary_repair.py": (
        ROOT / "modules" / "processors" / "frame" / "boundary_repair.py"
    ),
    r"tools\stability_report.py": ROOT / "tools" / "stability_report.py",
}


def fail(result, operation: str) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout).strip()
    raise RuntimeError(f"{operation} failed: {detail}")


def ps_quote(value: str) -> str:
    return value.replace("'", "''")


def stage() -> None:
    prepare = ssh_exec(
        HOST,
        USER,
        KEY,
        "$ErrorActionPreference='Stop'\n"
        f"New-Item -ItemType Directory -Force -Path '{REMOTE_STAGE}' | Out-Null\n"
        f"New-Item -ItemType Directory -Force -Path '{REMOTE_STAGE}\\modules' | Out-Null\n"
        f"New-Item -ItemType Directory -Force -Path '{REMOTE_STAGE}\\modules\\processors\\frame' | Out-Null\n"
        f"New-Item -ItemType Directory -Force -Path '{REMOTE_STAGE}\\tools' | Out-Null",
    )
    fail(prepare, "create staging directory")
    for relative, local in FILES.items():
        remote = (REMOTE_STAGE + "\\" + relative).replace("\\", "/")
        code = scp_to(HOST, USER, KEY, local, remote)
        if code != 0:
            raise RuntimeError(f"upload failed: {relative}")

    expected = {
        relative: hashlib.sha256(local.read_bytes()).hexdigest().upper()
        for relative, local in FILES.items()
    }
    paths = ",".join(
        f"'{ps_quote(REMOTE_STAGE + chr(92) + relative)}'" for relative in FILES
    )
    validate = ssh_exec(
        HOST,
        USER,
        KEY,
        "$ErrorActionPreference='Stop'\n"
        f"$paths=@({paths})\n"
        "$hashes=@{}\n"
        "foreach($path in $paths){$relative=$path.Substring('"
        + ps_quote(REMOTE_STAGE)
        + "'.Length+1);$hashes[$relative]=(Get-FileHash -Algorithm SHA256 $path).Hash}\n"
        f"python -m py_compile '{REMOTE_STAGE}\\run_network.py' "
        f"'{REMOTE_STAGE}\\run.py' "
        f"'{REMOTE_STAGE}\\modules\\device_slots.py' "
        f"'{REMOTE_STAGE}\\modules\\network_router.py' "
        f"'{REMOTE_STAGE}\\modules\\live_stream.py' "
        f"'{REMOTE_STAGE}\\modules\\live_processor.py' "
        f"'{REMOTE_STAGE}\\modules\\remote_control.py' "
        f"'{REMOTE_STAGE}\\modules\\globals.py' "
        f"'{REMOTE_STAGE}\\modules\\face_analyser.py' "
        f"'{REMOTE_STAGE}\\modules\\face_tracking.py' "
        f"'{REMOTE_STAGE}\\modules\\quality_pipeline.py' "
        f"'{REMOTE_STAGE}\\modules\\pipeline_benchmark.py' "
        f"'{REMOTE_STAGE}\\modules\\swapper_contract.py' "
        f"'{REMOTE_STAGE}\\modules\\instyle256_swapper.py' "
        f"'{REMOTE_STAGE}\\modules\\simswap512_swapper.py' "
        f"'{REMOTE_STAGE}\\modules\\processors\\frame\\face_swapper.py' "
        f"'{REMOTE_STAGE}\\modules\\processors\\frame\\frequency_repair.py' "
        f"'{REMOTE_STAGE}\\modules\\processors\\frame\\boundary_repair.py' "
        f"'{REMOTE_STAGE}\\tools\\stability_report.py'\n"
        "$hashes | ConvertTo-Json -Compress",
    )
    fail(validate, "Windows staging validation")
    actual = json.loads(validate.stdout.strip().splitlines()[-1])
    if actual != expected:
        raise RuntimeError(f"remote hashes differ: expected={expected} actual={actual}")
    print("Windows overlay staged, hashed, and compiled successfully.")


def activate() -> None:
    script = (
        "$ErrorActionPreference='Stop'\n"
        f"& '{REMOTE_STAGE}\\activate-multidevice.ps1'"
    )
    result = ssh_exec(HOST, USER, KEY, script)
    fail(result, "activate Windows overlay")
    print(result.stdout.strip())


def status() -> None:
    """Print bounded live diagnostics without modifying the Windows host."""
    script = r"""
$ErrorActionPreference='Continue'
$log='C:\Deep-Live-Cam\runtime\network-live\service.log'
if (Test-Path $log) { Get-Content -Path $log -Tail 160 }
'--- processes ---'
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq 'ffmpeg.exe' -or $_.CommandLine -match 'run_network.py' } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine |
    ConvertTo-Json -Depth 3
"""
    result = ssh_exec(HOST, USER, KEY, script)
    fail(result, "read Windows camera status")
    print(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="back up live files, replace them, and restart the scheduled task",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="show the live Windows service log and worker command lines",
    )
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    stage()
    if args.activate:
        activate()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"deploy_windows_multidevice: {exc}", file=sys.stderr)
        raise SystemExit(1)
