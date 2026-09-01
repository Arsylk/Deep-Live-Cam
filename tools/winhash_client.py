#!/usr/bin/env python3
"""
Simple SSH interface for sending hashcat jobs from this machine to the Win11 GPU host.
"""

import argparse
import base64
import json
import os
import secrets
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path


def powershell_encode(script: str) -> str:
    return base64.b64encode(script.encode("utf-16le")).decode("ascii")


def run_command(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, text=True, capture_output=capture)


def parse_connection(args):
    target = args.target or os.environ.get("WINHASH_TARGET", f"{args.user}@{args.host}")
    if "@" in target:
        user, host = target.split("@", 1)
        return user, host, args.key
    return args.user, target, args.key


def win_path(path: str) -> str:
    return path.replace("/", "\\")


def quote_ps(value: str) -> str:
    return value.replace("'", "''")


def ps_from_script_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def ssh_exec(host: str, user: str, key: str, script: str) -> subprocess.CompletedProcess:
    encoded = powershell_encode(script)
    remote_cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        encoded,
    ]
    return run_command(
        [
            "ssh",
            "-i",
            key,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{user}@{host}",
            " ".join(remote_cmd),
        ],
        capture=True,
    )


def scp_to(host: str, user: str, key: str, local_path: Path, remote_path: str) -> int:
    res = run_command(
        [
            "scp",
            "-i",
            key,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            str(local_path),
            f"{user}@{host}:{remote_path}",
        ],
        capture=True,
    )
    if res.returncode != 0:
        print(res.stderr.strip(), file=sys.stderr)
    return res.returncode


def build_run_script(
    *,
    hashcat_path: str,
    job_id: str,
    work_dir: str,
    hash_file: str,
    mode: str,
    backend_device: str,
    args_payload: str,
    force: bool,
    status_timer: int,
) -> str:
    return ps_from_script_lines(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$hashcatPath = '{quote_ps(hashcat_path)}'",
            f"$workDir = '{quote_ps(work_dir)}'",
            f"$jobId = '{job_id}'",
            f"$hashFile = '{quote_ps(hash_file)}'",
            f"$mode = '{mode}'",
            f"$backendDevice = '{backend_device}'",
            f"$forceRun = ${str(force).lower()}",
            f"$statusTimer = {status_timer}",
            f"$payload = '{quote_ps(args_payload)}'",
            f"$jobDir = Join-Path $workDir $jobId",
            "New-Item -ItemType Directory -Force -Path $jobDir | Out-Null",
            "$jobArgs = @('-m', $mode)",
            "$jobArgs += '--backend-devices=' + $backendDevice",
            "$jobArgs += '--status'",
            "$jobArgs += '--status-json'",
            "$jobArgs += '--status-timer=' + $statusTimer",
            "$jobArgs += '--session=' + $jobId",
            "$jobArgs += '--outfile=' + (Join-Path $jobDir 'cracked.txt')",
            "if ($payload) {",
            "  try {",
            "    $json = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($payload)) | ConvertFrom-Json",
            "    if ($json) {",
            "      $jobArgs += $json",
            "    }",
            "  } catch {",
            "    Write-Error 'Failed to decode user arguments.'",
            "    exit 2",
            "  }",
            "}",
            "$jobArgs += $hashFile",
            "if (-not (Test-Path $hashcatPath)) {",
            "  Write-Output ('{\"error\":\"hashcat_missing\",\"detail\":\"Hashcat path was not found. Set --hashcat to a valid executable.\"}')",
            "  exit 25",
            "}",
            "$cameraBusy = $false",
            "if (-not $forceRun) {",
            "  try {",
            "    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/healthz' -TimeoutSec 2",
            "    if ($null -ne $health -and [bool]$health.streaming -and [bool]$health.processing.enabled) {",
            "      $cameraBusy = $true",
            "    }",
            "  } catch {",
            "    $cameraBusy = $false",
            "  }",
            "  if ($cameraBusy) {",
            "    Write-Output (('{\"error\":\"camera_busy\",\"detail\":\"Streaming pipeline is active. Use --force to bypass.\"}') )",
            "    exit 23",
            "  }",
            "}",
            "$existing = Get-Process -Name hashcat -ErrorAction SilentlyContinue",
            "if ($existing) {",
            "  Write-Output (('{\"error\":\"hashcat_already_running\",\"detail\":\"Stop existing hashcat process first.\"}') )",
            "  exit 24",
            "}",
            "$stdout = Join-Path $jobDir 'stdout.log'",
            "$stderr = Join-Path $jobDir 'stderr.log'",
            "$statePath = Join-Path $jobDir 'state.json'",
            "$state = @{",
            "  job_id = $jobId;",
            "  mode = $mode;",
            "  hash_file = $hashFile;",
            "  working_dir = $jobDir;",
            "  started_at = (Get-Date).ToString('o')",
            "  status = 'starting'",
            "  pid = $null",
            "}",
            "([pscustomobject]$state) | ConvertTo-Json -Compress | Set-Content -Path $statePath -Encoding UTF8",
            "$proc = Start-Process -FilePath $hashcatPath -ArgumentList $jobArgs -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WorkingDirectory $jobDir -PassThru -WindowStyle Hidden",
            "$state.status = 'running'",
            "$state.pid = $proc.Id",
            "([pscustomobject]$state) | ConvertTo-Json -Compress | Set-Content -Path $statePath -Encoding UTF8",
            "Write-Output ($proc | Select-Object Id, Path | ConvertTo-Json -Compress)",
        ]
    )


def build_simple_ps(cmd: str) -> str:
    return ps_from_script_lines([cmd])


def action_run(args):
    user, host, key = parse_connection(args)
    mode = str(args.mode)
    work_dir = win_path(args.workdir)
    hashcat_path = win_path(args.hashcat)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    remote_dir = f"{work_dir}\\{job_id}"
    remote_hash = f"{remote_dir}\\hashes.txt"

    if not Path(args.hash_file).exists():
        print(f"Local hash file not found: {args.hash_file}", file=sys.stderr)
        return 1

    prep = ps_from_script_lines(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$path = '{quote_ps(remote_dir)}'",
            "New-Item -ItemType Directory -Force -Path $path | Out-Null",
        ]
    )
    prep_res = ssh_exec(host, user, key, prep)
    if prep_res.returncode != 0:
        print(prep_res.stdout or prep_res.stderr)
        return prep_res.returncode

    scp_code = scp_to(host, user, key, Path(args.hash_file), remote_hash)
    if scp_code != 0:
        return scp_code

    payload = base64.b64encode(json.dumps(args.hashcat_args).encode()).decode()
    run_script = build_run_script(
        hashcat_path=hashcat_path,
        job_id=job_id,
        work_dir=work_dir,
        hash_file=remote_hash,
        mode=mode,
        backend_device=str(args.backend_device),
        args_payload=payload,
        force=args.force,
        status_timer=args.status_timer,
    )
    result = ssh_exec(host, user, key, run_script)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def action_devices(args):
    user, host, key = parse_connection(args)
    script = build_simple_ps(f"& '{quote_ps(win_path(args.hashcat))}' -I")
    result = ssh_exec(host, user, key, script)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def action_status(args):
    user, host, key = parse_connection(args)
    script = build_simple_ps(
        "\n".join(
            [
                "$ErrorActionPreference = 'Continue'",
                "Write-Output '--- Camera service ---'",
                "try {",
                "  $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/healthz' -TimeoutSec 2",
                "  $health | ConvertTo-Json -Compress",
                "} catch {",
                "  Write-Output '{\"error\":\"camera_health_unavailable\"}'",
                "}",
                "Write-Output '--- Hashcat processes ---'",
                "Get-Process -Name hashcat -ErrorAction SilentlyContinue | Select-Object Id,Path,StartTime | ConvertTo-Json -Depth 2",
                "Write-Output '--- Work dir ---'",
                f"if (Test-Path '{quote_ps(win_path(args.workdir))}') {{ Get-ChildItem -Path '{quote_ps(win_path(args.workdir))}' -Directory | Select-Object Name,LastWriteTime | Sort-Object LastWriteTime -Descending | ConvertTo-Json -Depth 2 }}",
                "else { Write-Output '{\"error\":\"workdir_missing\"}' }",
            ]
        )
    )
    result = ssh_exec(host, user, key, script)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def action_stop(args):
    user, host, key = parse_connection(args)
    script = build_simple_ps(
        "Get-Process -Name hashcat -ErrorAction SilentlyContinue | Stop-Process -Force; Write-Output 'hashcat stop requested'"
    )
    result = ssh_exec(host, user, key, script)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def add_common(parser: argparse.ArgumentParser):
    parser.add_argument("--user", default=os.environ.get("WINHASH_USER", "Zuzia"))
    parser.add_argument("--host", default=os.environ.get("WINHASH_HOST", "192.168.1.35"))
    parser.add_argument("--key", default=os.path.expanduser(os.environ.get("WINHASH_KEY", "~/.ssh/id_ed25519")))
    parser.add_argument("--hashcat", default="C:\\Users\\Zuzia\\Downloads\\hashcat-7.1.2\\hashcat.exe")
    parser.add_argument("--workdir", default="C:\\HashcatWorker")
    parser.add_argument("--force", action="store_true", help="Run even when camera pipeline reports busy.")


def main():
    parser = argparse.ArgumentParser(prog="winhash_client.py")
    add_common(parser)
    parser.add_argument("--target", default=os.environ.get("WINHASH_TARGET"))
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="Deploy a single hash list and launch hashcat")
    run.add_argument("hash_file", help="Path to local hash list file")
    run.add_argument("hashcat_args", nargs="*", help="Additional hashcat args (appended after hash file)")
    run.add_argument("--mode", "-m", default="0", help="Hash type mode (defaults to 0)")
    run.add_argument("--backend-device", default="1", help="CUDA/OpenCL backend device id")
    run.add_argument("--status-timer", type=int, default=10, help="Hashcat status interval (seconds)")
    run.set_defaults(func=action_run)

    devices = sub.add_parser("devices", help="List hashcat backends on Windows")
    devices.set_defaults(func=action_devices)

    status = sub.add_parser("status", help="Check camera state and hashcat status")
    status.set_defaults(func=action_status)

    stop = sub.add_parser("stop", help="Stop all remote hashcat processes")
    stop.set_defaults(func=action_stop)

    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
