"""Identity-space validation shared by training and export tools."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


IDENTITY_CONTRACT = "inswapper-buff2fs-mapped-l2-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_emap(path: str | Path) -> dict[str, str | int]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"identity map does not exist: {source}")
    value = np.load(source, allow_pickle=False)
    if value.shape != (512, 512) or value.dtype != np.float32:
        raise ValueError(
            f"identity map must be float32 [512,512], got {value.shape} {value.dtype}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("identity map contains non-finite values")
    return {
        "contract": IDENTITY_CONTRACT,
        "emap_sha256": file_sha256(source),
        "emap_bytes": source.stat().st_size,
    }


__all__ = ["IDENTITY_CONTRACT", "file_sha256", "validate_emap"]
