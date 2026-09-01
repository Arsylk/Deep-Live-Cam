#!/usr/bin/env python3
"""Offline C2PA manifest generation, signing, and verification helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


AI_EDIT_SOURCE_TYPE = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/"
    "compositeWithTrainedAlgorithmicMedia"
)
OFFLINE_SETTINGS = """version = 1

[verify]
verify_after_reading = true
verify_after_sign = true
remote_manifest_fetch = false
"""


def build_manifest(
    title: str,
    *,
    version: str = "2.1.6",
    description: str = "Face identity transformed using a trained algorithmic model.",
) -> dict[str, Any]:
    """Build a minimal C2PA v2 edit manifest accepted by ``c2patool``.

    Signing credentials and a timestamp authority are intentionally absent.
    Production credentials must be supplied explicitly at signing time, and
    offline operation never silently contacts a timestamp service.
    """
    software_agent = {"name": "Deep-Live-Cam", "version": version}
    return {
        "claim_generator": f"Deep-Live-Cam/{version}",
        "claim_generator_info": [software_agent],
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.edited",
                            "digitalSourceType": AI_EDIT_SOURCE_TYPE,
                            "softwareAgent": software_agent,
                            "description": description,
                        }
                    ],
                    "allActionsIncluded": True,
                },
            },
            {
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "VideoObject",
                    "name": title,
                    "description": (
                        "Video containing a face identity transformation."
                    ),
                },
            },
        ],
    }


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return semantic errors relevant to this application's C2PA manifest."""
    errors: list[str] = []
    if not str(manifest.get("claim_generator", "")).startswith("Deep-Live-Cam/"):
        errors.append("claim_generator must identify Deep-Live-Cam and a version")
    assertions = manifest.get("assertions")
    if not isinstance(assertions, list):
        return errors + ["assertions must be a list"]
    actions = None
    for assertion in assertions:
        if isinstance(assertion, dict) and assertion.get("label") == "c2pa.actions.v2":
            data = assertion.get("data")
            if isinstance(data, dict):
                actions = data.get("actions")
            break
    if not isinstance(actions, list) or not actions:
        return errors + ["a c2pa.actions.v2 action is required"]
    edit = next(
        (
            action
            for action in actions
            if isinstance(action, dict) and action.get("action") == "c2pa.edited"
        ),
        None,
    )
    if edit is None:
        errors.append("the manifest must declare c2pa.edited")
    else:
        if edit.get("digitalSourceType") != AI_EDIT_SOURCE_TYPE:
            errors.append("c2pa.edited must declare the trained-algorithm composite")
        if not str(edit.get("description", "")).strip():
            errors.append("c2pa.edited must include a human-readable description")
    return errors


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid C2PA manifest: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validation_entries(value: Any) -> list[Any]:
    if isinstance(value, dict):
        entries: list[Any] = []
        for key, child in value.items():
            if key in {"validation_status", "validationStatus"} and isinstance(
                child, list
            ):
                entries.extend(child)
            else:
                entries.extend(_validation_entries(child))
        return entries
    if isinstance(value, list):
        entries = []
        for child in value:
            entries.extend(_validation_entries(child))
        return entries
    return []


def inspect_asset(asset: Path, c2patool: str | None = None) -> dict[str, Any]:
    """Inspect an asset without network access or modification."""
    executable = c2patool or shutil.which("c2patool")
    if not executable:
        return {
            "tool_available": False,
            "network_access": False,
            "has_manifest": None,
            "valid": None,
            "detail": "c2patool is not installed; provenance was not verified",
        }
    with tempfile.TemporaryDirectory(prefix="dlc-c2pa-offline-") as directory:
        settings = Path(directory) / "c2pa.toml"
        settings.write_text(OFFLINE_SETTINGS, encoding="utf-8")
        completed = subprocess.run(
            [executable, str(asset), "--settings", str(settings)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    output = completed.stdout.strip()
    try:
        report: Any = json.loads(output) if output else {}
    except json.JSONDecodeError:
        report = {"raw_output": output[:4000]}
    has_manifest = False
    if isinstance(report, dict):
        has_manifest = bool(
            report.get("active_manifest")
            or report.get("activeManifest")
            or report.get("manifests")
        )
    combined = (completed.stdout + "\n" + completed.stderr).lower()
    if "no claim found" in combined or "no c2pa" in combined:
        has_manifest = False
    validation = _validation_entries(report)
    return {
        "tool_available": True,
        "network_access": False,
        "tool": executable,
        "exit_code": completed.returncode,
        "has_manifest": has_manifest,
        "valid": bool(has_manifest and completed.returncode == 0 and not validation),
        "validation_status": validation,
        "report": report,
        "stderr": completed.stderr.strip()[:4000],
    }


def signing_command(
    asset: Path,
    output: Path,
    manifest_path: Path,
    *,
    parent: Path,
    certificate: Path,
    private_key: Path,
    algorithm: str = "es256",
    c2patool: str = "c2patool",
) -> tuple[list[str], dict[str, Any]]:
    """Return the offline signing command and credential-augmented manifest.

    The caller writes the returned manifest to ``manifest_path``. No command
    is executed here, which makes credential handling explicit and testable.
    """
    if algorithm not in {
        "ps256",
        "ps384",
        "ps512",
        "es256",
        "es384",
        "es512",
        "ed25519",
    }:
        raise ValueError("unsupported C2PA signing algorithm")
    for label, path in (
        ("asset", asset),
        ("parent", parent),
        ("certificate", certificate),
        ("private key", private_key),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing signed output: {output}")
    manifest = build_manifest(output.name)
    manifest.update(
        {
            "alg": algorithm,
            "private_key": str(private_key.resolve()),
            "sign_cert": str(certificate.resolve()),
        }
    )
    command = [
        c2patool,
        str(asset.resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        "--parent",
        str(parent.resolve()),
        "--output",
        str(output.resolve()),
    ]
    return command, manifest
