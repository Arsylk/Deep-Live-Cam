"""Authorized, identity-disjoint manifest handling for native-256 training.

The loader performs no downloads. Every path must remain below the caller's
data root and every sample must explicitly declare its provenance and usage
authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


VALID_SPLITS = frozenset({"train", "validation", "test"})
IDENTITY_TYPE = "inswapper_mapped_512"
TEACHER_TYPE = "uncomposited-candidate-alpha-rgb-128-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: str
    target: str
    target_identity: str
    source_identity: str
    source_latent: str
    identity_type: str
    identity_map_sha256: str
    source_embedding: str | None
    teacher_candidate: str | None
    teacher_alpha: str | None
    teacher_type: str | None
    face_mask: str | None
    next_target: str | None
    flow: str | None
    flow_valid: str | None
    sequence: str | None
    frame_index: int | None
    teacher_confidence: float
    provenance: str
    license: str
    usage_authorized: bool

    @property
    def same_identity(self) -> bool:
        return self.source_identity == self.target_identity


def _required_text(data: dict[str, Any], key: str, line: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest line {line}: {key} must be non-empty text")
    return value.strip()


def _optional_text(data: dict[str, Any], key: str, line: int) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest line {line}: {key} must be null or non-empty text")
    return value.strip()


def _parse_record(data: object, line: int) -> SampleRecord:
    if not isinstance(data, dict):
        raise ValueError(f"manifest line {line}: record must be a JSON object")
    allowed = {
        "sample_id",
        "split",
        "target",
        "target_identity",
        "source_identity",
        "source_latent",
        "identity_type",
        "identity_map_sha256",
        "source_embedding",
        "teacher_candidate",
        "teacher_alpha",
        "teacher_type",
        "face_mask",
        "next_target",
        "flow",
        "flow_valid",
        "sequence",
        "frame_index",
        "teacher_confidence",
        "provenance",
        "license",
        "usage_authorized",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"manifest line {line}: unknown fields: {', '.join(unknown)}"
        )
    split = _required_text(data, "split", line)
    if split not in VALID_SPLITS:
        raise ValueError(
            f"manifest line {line}: split must be one of {sorted(VALID_SPLITS)}"
        )
    authorization = data.get("usage_authorized")
    if authorization is not True:
        raise ValueError(
            f"manifest line {line}: usage_authorized must explicitly be true"
        )
    confidence = data.get("teacher_confidence", 1.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"manifest line {line}: teacher_confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"manifest line {line}: teacher_confidence must be in [0,1]"
        )
    frame_index = data.get("frame_index")
    if frame_index is not None and (
        isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0
    ):
        raise ValueError(f"manifest line {line}: frame_index must be non-negative")
    temporal = [data.get("next_target"), data.get("flow"), data.get("flow_valid")]
    if any(item is not None for item in temporal) and not all(
        item is not None for item in temporal
    ):
        raise ValueError(
            f"manifest line {line}: next_target, flow and flow_valid are all-or-none"
        )
    identity_type = _required_text(data, "identity_type", line)
    if identity_type != IDENTITY_TYPE:
        raise ValueError(
            f"manifest line {line}: identity_type must be {IDENTITY_TYPE!r}"
        )
    identity_map_sha256 = _required_text(data, "identity_map_sha256", line).lower()
    if not _SHA256.fullmatch(identity_map_sha256):
        raise ValueError(
            f"manifest line {line}: identity_map_sha256 must be 64 hexadecimal digits"
        )
    teacher_candidate = _optional_text(data, "teacher_candidate", line)
    teacher_alpha = _optional_text(data, "teacher_alpha", line)
    teacher_type = _optional_text(data, "teacher_type", line)
    teacher_fields = (teacher_candidate, teacher_alpha, teacher_type)
    if any(value is not None for value in teacher_fields) and not all(
        value is not None for value in teacher_fields
    ):
        raise ValueError(
            f"manifest line {line}: teacher_candidate, teacher_alpha and "
            "teacher_type must be supplied together"
        )
    if teacher_type is not None and teacher_type != TEACHER_TYPE:
        raise ValueError(
            f"manifest line {line}: teacher_type must be {TEACHER_TYPE!r}"
        )
    return SampleRecord(
        sample_id=_required_text(data, "sample_id", line),
        split=split,
        target=_required_text(data, "target", line),
        target_identity=_required_text(data, "target_identity", line),
        source_identity=_required_text(data, "source_identity", line),
        source_latent=_required_text(data, "source_latent", line),
        identity_type=identity_type,
        identity_map_sha256=identity_map_sha256,
        source_embedding=_optional_text(data, "source_embedding", line),
        teacher_candidate=teacher_candidate,
        teacher_alpha=teacher_alpha,
        teacher_type=teacher_type,
        face_mask=_optional_text(data, "face_mask", line),
        next_target=_optional_text(data, "next_target", line),
        flow=_optional_text(data, "flow", line),
        flow_valid=_optional_text(data, "flow_valid", line),
        sequence=_optional_text(data, "sequence", line),
        frame_index=frame_index,
        teacher_confidence=confidence,
        provenance=_required_text(data, "provenance", line),
        license=_required_text(data, "license", line),
        usage_authorized=True,
    )


def read_manifest(path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"manifest line {line_number}: invalid JSON: {error.msg}"
                ) from error
            records.append(_parse_record(value, line_number))
    if not records:
        raise ValueError("manifest contains no samples")
    validate_records(records)
    return records


def validate_records(records: Iterable[SampleRecord]) -> None:
    seen_ids: set[str] = set()
    identities_by_split: dict[str, set[str]] = {
        name: set() for name in VALID_SPLITS
    }
    identity_map_hashes: set[str] = set()
    for record in records:
        if record.sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        seen_ids.add(record.sample_id)
        identities_by_split[record.split].update(
            (record.target_identity, record.source_identity)
        )
        identity_map_hashes.add(record.identity_map_sha256)
        if not record.same_identity and record.teacher_candidate is None:
            raise ValueError(
                f"cross-identity sample {record.sample_id} has no teacher candidate/alpha"
            )
    split_names = sorted(identities_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = identities_by_split[left] & identities_by_split[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"identity leakage between {left} and {right}: {examples}"
                )
    if len(identity_map_hashes) != 1:
        raise ValueError(
            "manifest mixes source latents produced by different identity maps"
        )


def resolve_under(root: str | Path, relative: str) -> Path:
    root_path = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"manifest path must be relative: {relative}")
    resolved = (root_path / candidate).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as error:
        raise ValueError(f"manifest path escapes data root: {relative}") from error
    return resolved


def validate_files(
    records: Iterable[SampleRecord], root: str | Path, *, split: str | None = None
) -> None:
    path_fields = (
        "target",
        "source_latent",
        "source_embedding",
        "teacher_candidate",
        "teacher_alpha",
        "face_mask",
        "next_target",
        "flow",
        "flow_valid",
    )
    for record in records:
        if split is not None and record.split != split:
            continue
        for field in path_fields:
            relative = getattr(record, field)
            if relative is None:
                continue
            path = resolve_under(root, relative)
            if not path.is_file():
                raise FileNotFoundError(
                    f"sample {record.sample_id}: missing {field}: {path}"
                )


def _rgb(path: Path, size: int) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            raise ValueError(f"expected {size}x{size} RGB image, got {image.size}: {path}")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _mask(path: Path) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (256, 256):
            raise ValueError(f"expected 256x256 mask, got {image.size}: {path}")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array[None].copy())


def _teacher_candidate(path: Path) -> Tensor:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (3, 128, 128):
        raise ValueError(
            f"teacher_candidate must be float32 with shape (3,128,128): {path}"
        )
    if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(
            f"teacher_candidate must contain finite values in [0,1]: {path}"
        )
    return torch.from_numpy(np.ascontiguousarray(value))


def _teacher_alpha(path: Path) -> Tensor:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (1, 128, 128):
        raise ValueError(
            f"teacher_alpha must be float32 with shape (1,128,128): {path}"
        )
    if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(
            f"teacher_alpha must contain finite values in [0,1]: {path}"
        )
    return torch.from_numpy(np.ascontiguousarray(value))


def _vector(path: Path, name: str) -> Tensor:
    value = np.load(path, allow_pickle=False)
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.shape != (512,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain 512 finite floats: {path}")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        raise ValueError(f"{name} has zero norm: {path}")
    return torch.from_numpy(np.ascontiguousarray(value / norm))


def _flow(path: Path) -> Tensor:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (2, 256, 256):
        raise ValueError(
            f"flow must be float32 with shape (2,256,256): {path}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"flow contains a non-finite value: {path}")
    return torch.from_numpy(np.ascontiguousarray(value))


def _flow_valid(path: Path) -> Tensor:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (1, 256, 256):
        raise ValueError(
            f"flow_valid must be float32 with shape (1,256,256): {path}"
        )
    if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"flow_valid must contain finite values in [0,1]: {path}")
    return torch.from_numpy(np.ascontiguousarray(value))


class Native256Dataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        split: str = "train",
        *,
        verify_files: bool = True,
        identity_map_sha256: str | None = None,
        identity_map: np.ndarray | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {split}")
        all_records = read_manifest(manifest)
        self.records = [record for record in all_records if record.split == split]
        if not self.records:
            raise ValueError(f"manifest contains no {split} samples")
        self.root = Path(root).resolve()
        self.identity_map = None
        if identity_map is not None:
            if identity_map.dtype != np.float32 or identity_map.shape != (512, 512):
                raise ValueError(
                    "identity_map must be float32 with shape (512,512)"
                )
            if not np.isfinite(identity_map).all():
                raise ValueError("identity_map contains a non-finite value")
            self.identity_map = np.ascontiguousarray(identity_map)
            self.identity_map.setflags(write=False)
        if identity_map_sha256 is not None:
            expected = identity_map_sha256.lower()
            mismatched = [
                record.sample_id
                for record in self.records
                if record.identity_map_sha256 != expected
            ]
            if mismatched:
                raise ValueError(
                    "manifest source latents do not match the requested identity map; "
                    f"first mismatched sample: {mismatched[0]}"
                )
        if verify_files:
            validate_files(self.records, self.root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        target = _rgb(resolve_under(self.root, record.target), 256)
        latent = _vector(
            resolve_under(self.root, record.source_latent), "source_latent"
        )
        teacher_candidate = torch.zeros((3, 128, 128), dtype=torch.float32)
        teacher_alpha = torch.zeros((1, 128, 128), dtype=torch.float32)
        has_teacher = record.teacher_candidate is not None
        if record.teacher_candidate is not None:
            teacher_candidate = _teacher_candidate(
                resolve_under(self.root, record.teacher_candidate)
            )
            teacher_alpha = _teacher_alpha(
                resolve_under(self.root, record.teacher_alpha)
            )
        source_embedding = torch.zeros(512, dtype=torch.float32)
        has_source_embedding = record.source_embedding is not None
        if record.source_embedding is not None:
            source_embedding = _vector(
                resolve_under(self.root, record.source_embedding),
                "source_embedding",
            )
            if self.identity_map is not None:
                expected = (
                    source_embedding.numpy().reshape(1, 512) @ self.identity_map
                ).reshape(512)
                norm = float(np.linalg.norm(expected))
                if not np.isfinite(norm) or norm <= 1e-8:
                    raise ValueError(
                        f"sample {record.sample_id}: mapped source identity has zero norm"
                    )
                expected = np.ascontiguousarray(expected / norm, dtype=np.float32)
                if not np.allclose(
                    latent.numpy(), expected, rtol=1e-5, atol=1e-6
                ):
                    raise ValueError(
                        f"sample {record.sample_id}: source_latent does not equal "
                        "source_embedding @ identity_map"
                    )
        face_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
        has_face_mask = record.face_mask is not None
        if record.face_mask is not None:
            face_mask = _mask(resolve_under(self.root, record.face_mask))
        next_target = target.clone()
        flow = torch.zeros((2, 256, 256), dtype=torch.float32)
        flow_valid = torch.zeros((1, 256, 256), dtype=torch.float32)
        has_temporal = record.next_target is not None
        if has_temporal:
            next_target = _rgb(resolve_under(self.root, record.next_target), 256)
            flow = _flow(resolve_under(self.root, record.flow))
            flow_valid = _flow_valid(resolve_under(self.root, record.flow_valid))
        return {
            "target": target,
            "source_latent": latent,
            "source_embedding": source_embedding,
            "teacher_candidate": teacher_candidate,
            "teacher_alpha": teacher_alpha,
            "face_mask": face_mask,
            "next_target": next_target,
            "flow": flow,
            "flow_valid": flow_valid,
            "has_teacher": torch.tensor(has_teacher, dtype=torch.bool),
            "has_source_embedding": torch.tensor(
                has_source_embedding, dtype=torch.bool
            ),
            "has_face_mask": torch.tensor(has_face_mask, dtype=torch.bool),
            "has_temporal": torch.tensor(has_temporal, dtype=torch.bool),
            "same_identity": torch.tensor(record.same_identity, dtype=torch.bool),
            "teacher_confidence": torch.tensor(
                record.teacher_confidence, dtype=torch.float32
            ),
        }


__all__ = [
    "Native256Dataset",
    "SampleRecord",
    "TEACHER_TYPE",
    "read_manifest",
    "resolve_under",
    "validate_files",
    "validate_records",
]
