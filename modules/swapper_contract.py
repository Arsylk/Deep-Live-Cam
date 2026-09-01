"""Strict runtime contract for aligned face-swap models.

The contract deliberately contains no download or model-discovery behavior.
Every artifact is named by a local JSON manifest and verified before a runtime
is allowed to open it.  This keeps model selection separate from inference and
prevents a requested model from being silently replaced with another one.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = 1
NATIVE_256_SIZE = (256, 256)
QUALITY_STATUSES = frozenset({"development", "experimental", "qualified"})
QUALIFICATION_GATES = frozenset(
    {
        "identity_similarity",
        "pose_landmarks",
        "background_temporal",
        "ncnn_parity",
        "android_qnn_admission",
        "rx570_latency",
        "target_phone_latency",
        "blind_review",
    }
)
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SwapperContractError(ValueError):
    """A local model artifact or inference result violates the contract."""


@dataclass(frozen=True, slots=True)
class OnnxAsset:
    """One hash-pinned ONNX file from a local manifest."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class IdentityConditionerSpec:
    asset: OnnxAsset
    input_name: str
    output_name: str


@dataclass(frozen=True, slots=True)
class SwapperModelSpec:
    asset: OnnxAsset
    target_input_name: str
    style_input_name: str
    candidate_output_name: str
    alpha_output_name: str


@dataclass(frozen=True, slots=True)
class NcnnIdentityConditionerSpec:
    """Hash-pinned ncnn identity-conditioner graph and fixed C-ABI blobs."""

    param: OnnxAsset
    model: OnnxAsset
    input_name: str
    output_name: str


@dataclass(frozen=True, slots=True)
class NcnnSwapperModelSpec:
    """Hash-pinned ncnn generator graph and fixed C-ABI blobs."""

    param: OnnxAsset
    model: OnnxAsset
    target_input_name: str
    style_input_name: str
    candidate_output_name: str
    alpha_output_name: str


@dataclass(frozen=True, slots=True)
class Native256NcnnSpec:
    """Optional local ncnn/Vulkan representation of the ONNX release."""

    backend: str
    fp16_storage: bool
    conditioner: NcnnIdentityConditionerSpec
    swapper: NcnnSwapperModelSpec


@dataclass(frozen=True, slots=True)
class PreprocessSpec:
    """NCHW RGB preprocessing applied to an aligned uint8 BGR crop."""

    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SwapperSpec:
    """Model capabilities consumed by the application-level paste pipeline."""

    model_id: str
    input_size: tuple[int, int]
    embedding_size: int
    identity_type: str
    has_semantic_alpha: bool = True
    face_precomposited: bool = False


@dataclass(frozen=True, slots=True)
class Native256Manifest:
    """Validated schema for the two-stage native-256 ONNX model."""

    path: Path
    quality_status: str
    auto_select_eligible: bool
    spec: SwapperSpec
    preprocess: PreprocessSpec
    candidate_range: tuple[float, float]
    alpha_range: tuple[float, float]
    identity_map: OnnxAsset
    conditioner: IdentityConditionerSpec
    swapper: SwapperModelSpec
    ncnn: Native256NcnnSpec | None
    qualification_report: OnnxAsset | None

    @classmethod
    def load(
        cls, manifest_path: str | Path, *, verify_hashes: bool = True
    ) -> "Native256Manifest":
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise SwapperContractError(f"native-256 manifest does not exist: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SwapperContractError(
                f"could not read native-256 manifest {path}: {error}"
            ) from error
        root = _mapping(document, "manifest")

        schema_version = _integer(root.get("schema_version"), "schema_version")
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise SwapperContractError(
                f"unsupported native-256 manifest schema: {schema_version}"
            )
        if root.get("format") != "onnx":
            raise SwapperContractError("manifest format must be 'onnx'")

        model_id = _name(root.get("model_id"), "model_id", model_id=True)
        quality_status = _name(root.get("quality_status"), "quality_status")
        if quality_status not in QUALITY_STATUSES:
            raise SwapperContractError(
                "quality_status must be development, experimental, or qualified"
            )
        auto_select_eligible = root.get("auto_select_eligible")
        if not isinstance(auto_select_eligible, bool):
            raise SwapperContractError(
                "auto_select_eligible must be a JSON boolean"
            )
        if auto_select_eligible and quality_status != "qualified":
            raise SwapperContractError(
                "auto_select_eligible may be true only for a qualified model"
            )
        input_size = _integer_pair(root.get("input_size"), "input_size")
        if input_size != NATIVE_256_SIZE:
            raise SwapperContractError(
                f"native-256 input_size must be [256, 256], got {list(input_size)}"
            )
        embedding_size = _integer(root.get("embedding_size"), "embedding_size")
        if embedding_size != 512:
            raise SwapperContractError(
                f"native-256 embedding_size must be 512, got {embedding_size}"
            )
        if root.get("layout") != "NCHW":
            raise SwapperContractError("manifest layout must be 'NCHW'")
        if root.get("color_order") != "RGB":
            raise SwapperContractError("manifest color_order must be 'RGB'")
        if root.get("identity_type") != "inswapper_mapped_512":
            raise SwapperContractError(
                "manifest identity_type must be 'inswapper_mapped_512'"
            )
        if root.get("candidate_precomposited") is not False:
            raise SwapperContractError(
                "native-256 candidate RGB must be uncomposited"
            )

        preprocess_doc = _mapping(root.get("preprocess"), "preprocess")
        scale = _finite_number(preprocess_doc.get("scale"), "preprocess.scale")
        if scale <= 0.0:
            raise SwapperContractError("preprocess.scale must be greater than zero")
        mean = _number_triplet(preprocess_doc.get("mean"), "preprocess.mean")
        std = _number_triplet(preprocess_doc.get("std"), "preprocess.std")
        if any(value <= 0.0 for value in std):
            raise SwapperContractError("every preprocess.std value must be positive")

        candidate_range = _number_pair(
            root.get("candidate_range"), "candidate_range"
        )
        alpha_range = _number_pair(root.get("alpha_range"), "alpha_range")
        if alpha_range != (0.0, 1.0):
            raise SwapperContractError("alpha_range must be [0.0, 1.0]")

        conditioner_doc = _mapping(
            root.get("identity_conditioner"), "identity_conditioner"
        )
        identity_map_doc = _mapping(root.get("identity_map"), "identity_map")
        swapper_doc = _mapping(root.get("swapper"), "swapper")
        identity_map = _asset(path.parent, identity_map_doc, "identity_map")
        if identity_map.path.suffix.lower() != ".npy":
            raise SwapperContractError("identity_map.file must be a .npy asset")
        conditioner = IdentityConditionerSpec(
            asset=_asset(path.parent, conditioner_doc, "identity_conditioner"),
            input_name=_name(
                conditioner_doc.get("input"), "identity_conditioner.input"
            ),
            output_name=_name(
                conditioner_doc.get("output"), "identity_conditioner.output"
            ),
        )
        swapper = SwapperModelSpec(
            asset=_asset(path.parent, swapper_doc, "swapper"),
            target_input_name=_name(
                swapper_doc.get("target_input"), "swapper.target_input"
            ),
            style_input_name=_name(
                swapper_doc.get("style_input"), "swapper.style_input"
            ),
            candidate_output_name=_name(
                swapper_doc.get("candidate_output"), "swapper.candidate_output"
            ),
            alpha_output_name=_name(
                swapper_doc.get("alpha_output"), "swapper.alpha_output"
            ),
        )
        if conditioner.asset.path.suffix.lower() != ".onnx":
            raise SwapperContractError(
                "identity_conditioner.file must be an .onnx asset"
            )
        if swapper.asset.path.suffix.lower() != ".onnx":
            raise SwapperContractError("swapper.file must be an .onnx asset")
        if swapper.candidate_output_name == swapper.alpha_output_name:
            raise SwapperContractError(
                "candidate_output and alpha_output must have different names"
            )
        if swapper.target_input_name == swapper.style_input_name:
            raise SwapperContractError(
                "target_input and style_input must have different names"
            )

        ncnn = _optional_ncnn_spec(path.parent, root.get("ncnn"))
        qualification_report = _optional_qualification_report(
            path.parent,
            root.get("qualification_report"),
            model_id=model_id,
            identity_map=identity_map,
            conditioner=conditioner,
            swapper=swapper,
            ncnn=ncnn,
            verify_hash=verify_hashes,
        )
        if auto_select_eligible and qualification_report is None:
            raise SwapperContractError(
                "auto_select_eligible requires a hash-pinned qualification_report"
            )

        if verify_hashes:
            _verify_asset(identity_map, "identity_map")
            _verify_asset(conditioner.asset, "identity_conditioner")
            _verify_asset(swapper.asset, "swapper")
            if ncnn is not None:
                _verify_asset(ncnn.conditioner.param, "ncnn.identity_conditioner.param")
                _verify_asset(ncnn.conditioner.model, "ncnn.identity_conditioner.bin")
                _verify_asset(ncnn.swapper.param, "ncnn.swapper.param")
                _verify_asset(ncnn.swapper.model, "ncnn.swapper.bin")

        return cls(
            path=path,
            quality_status=quality_status,
            auto_select_eligible=auto_select_eligible,
            spec=SwapperSpec(
                model_id=model_id,
                input_size=input_size,
                embedding_size=embedding_size,
                identity_type="inswapper_mapped_512",
                has_semantic_alpha=True,
                face_precomposited=False,
            ),
            preprocess=PreprocessSpec(scale=scale, mean=mean, std=std),
            candidate_range=candidate_range,
            alpha_range=alpha_range,
            identity_map=identity_map,
            conditioner=conditioner,
            swapper=swapper,
            ncnn=ncnn,
            qualification_report=qualification_report,
        )


@dataclass(slots=True)
class SwapResult:
    """One generated aligned face and its semantic over-mask."""

    face_bgr: np.ndarray
    affine: np.ndarray
    alpha: np.ndarray | None
    model_id: str
    backend: str

    def __post_init__(self) -> None:
        if not isinstance(self.face_bgr, np.ndarray):
            raise SwapperContractError("face_bgr must be a numpy array")
        if self.face_bgr.dtype != np.uint8:
            raise SwapperContractError("face_bgr must have dtype uint8")
        if (
            self.face_bgr.ndim != 3
            or self.face_bgr.shape[2] != 3
            or self.face_bgr.shape[0] < 1
            or self.face_bgr.shape[1] < 1
        ):
            raise SwapperContractError(
                f"face_bgr must have shape HxWx3, got {self.face_bgr.shape}"
            )
        if not self.face_bgr.flags.c_contiguous:
            raise SwapperContractError("face_bgr must be C-contiguous")

        if not isinstance(self.affine, np.ndarray):
            raise SwapperContractError("affine must be a numpy array")
        if self.affine.dtype != np.float32 or self.affine.shape != (2, 3):
            raise SwapperContractError(
                f"affine must be float32 with shape (2, 3), got "
                f"{self.affine.dtype} {self.affine.shape}"
            )
        if not np.isfinite(self.affine).all():
            raise SwapperContractError("affine contains a non-finite value")

        if self.alpha is not None:
            if not isinstance(self.alpha, np.ndarray):
                raise SwapperContractError("alpha must be a numpy array or None")
            if self.alpha.dtype != np.uint8:
                raise SwapperContractError("alpha must have dtype uint8")
            if self.alpha.shape != self.face_bgr.shape[:2]:
                raise SwapperContractError(
                    f"alpha shape {self.alpha.shape} does not match face "
                    f"shape {self.face_bgr.shape[:2]}"
                )
            if not self.alpha.flags.c_contiguous:
                raise SwapperContractError("alpha must be C-contiguous")

        _name(self.model_id, "model_id", model_id=True)
        _name(self.backend, "backend")

    @property
    def has_semantic_alpha(self) -> bool:
        return self.alpha is not None


def normalized_embedding(value: Any, expected_size: int = 512) -> np.ndarray:
    """Validate and L2-normalize one source identity embedding."""
    try:
        embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as error:
        raise SwapperContractError(f"invalid source identity embedding: {error}") from error
    if embedding.shape != (expected_size,):
        raise SwapperContractError(
            f"source identity embedding must contain {expected_size} values, "
            f"got {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise SwapperContractError("source identity embedding is not finite")
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise SwapperContractError("source identity embedding has zero norm")
    return np.ascontiguousarray((embedding / norm).reshape(1, expected_size))


def mapped_inswapper_identity(value: Any, identity_map: np.ndarray) -> np.ndarray:
    """Apply INSwapper's identity projection once, then L2-normalize.

    This deliberately does not normalize the source vector before projection.
    It matches InsightFace INSwapper, the ncnn adapter, and the Android bridge:
    ``normed_embedding @ emap`` followed by one normalization of that latent.
    """
    if not isinstance(identity_map, np.ndarray):
        raise SwapperContractError("identity map must be a numpy array")
    if identity_map.dtype != np.float32 or identity_map.shape != (512, 512):
        raise SwapperContractError(
            "identity map must be float32 with shape (512, 512)"
        )
    if not np.isfinite(identity_map).all():
        raise SwapperContractError("identity map contains a non-finite value")
    try:
        embedding = np.asarray(value, dtype=np.float32).reshape(1, -1)
    except (TypeError, ValueError) as error:
        raise SwapperContractError(f"invalid source identity embedding: {error}") from error
    if embedding.shape != (1, 512):
        raise SwapperContractError(
            "source identity embedding must contain 512 values, "
            f"got {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise SwapperContractError("source identity embedding is not finite")
    latent = np.ascontiguousarray(embedding @ identity_map, dtype=np.float32)
    norm = float(np.linalg.norm(latent))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise SwapperContractError("mapped source identity has zero norm")
    latent /= norm
    return np.ascontiguousarray(latent)


def verify_embedded_onnx(asset: OnnxAsset, label: str) -> None:
    """Reject invalid ONNX or any tensor stored outside the hash-pinned file."""
    try:
        import onnx
    except ImportError as error:
        raise SwapperContractError(
            "ONNX validation is required to reject mutable external tensor data"
        ) from error
    try:
        model = onnx.load(str(asset.path), load_external_data=False)
    except Exception as error:
        raise SwapperContractError(f"could not validate {label} ONNX: {error}") from error

    external_names: list[str] = []

    def walk(message: Any) -> None:
        if isinstance(message, onnx.TensorProto) and (
            message.data_location == onnx.TensorProto.EXTERNAL
            or bool(message.external_data)
        ):
            external_names.append(message.name or "<unnamed>")
        for field, field_value in message.ListFields():
            if field.type != field.TYPE_MESSAGE:
                continue
            if field.is_repeated:
                for child in field_value:
                    walk(child)
            else:
                walk(field_value)

    walk(model)
    if external_names:
        raise SwapperContractError(
            f"{label} uses external ONNX tensor data ({', '.join(external_names)}); "
            "every deployable tensor must be embedded in its hash-pinned .onnx file"
        )


def _optional_qualification_report(
    root: Path,
    value: Any,
    *,
    model_id: str,
    identity_map: OnnxAsset,
    conditioner: IdentityConditionerSpec,
    swapper: SwapperModelSpec,
    ncnn: Native256NcnnSpec | None,
    verify_hash: bool,
) -> OnnxAsset | None:
    if value is None:
        return None
    document = _mapping(value, "qualification_report")
    asset = _asset(root, document, "qualification_report")
    if asset.path.suffix.lower() != ".json":
        raise SwapperContractError("qualification_report.file must be a .json asset")
    if verify_hash:
        _verify_asset(asset, "qualification_report")
    try:
        report_value = json.loads(asset.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SwapperContractError(
            f"could not read qualification report {asset.path}: {error}"
        ) from error
    report = _mapping(report_value, "qualification report")
    if _integer(report.get("schema_version"), "qualification.schema_version") != 1:
        raise SwapperContractError("qualification.schema_version must be 1")
    if _name(report.get("model_id"), "qualification.model_id", model_id=True) != model_id:
        raise SwapperContractError(
            "qualification.model_id does not match the model manifest"
        )
    if report.get("verdict") != "qualified":
        raise SwapperContractError("qualification.verdict must be 'qualified'")

    expected_artifacts = {
        "identity_map_sha256": identity_map.sha256,
        "identity_conditioner_sha256": conditioner.asset.sha256,
        "swapper_sha256": swapper.asset.sha256,
    }
    if ncnn is not None:
        expected_artifacts.update(
            {
                "ncnn_identity_conditioner_param_sha256": ncnn.conditioner.param.sha256,
                "ncnn_identity_conditioner_bin_sha256": ncnn.conditioner.model.sha256,
                "ncnn_swapper_param_sha256": ncnn.swapper.param.sha256,
                "ncnn_swapper_bin_sha256": ncnn.swapper.model.sha256,
            }
        )
    artifacts = _mapping(report.get("artifacts"), "qualification.artifacts")
    if set(artifacts) != set(expected_artifacts):
        raise SwapperContractError(
            "qualification.artifacts must bind exactly the deployed model artifacts"
        )
    for name, expected in expected_artifacts.items():
        actual = _name(artifacts.get(name), f"qualification.artifacts.{name}").lower()
        if not _SHA256.fullmatch(actual):
            raise SwapperContractError(
                f"qualification.artifacts.{name} must be a SHA-256 digest"
            )
        if actual != expected:
            raise SwapperContractError(
                f"qualification.artifacts.{name} does not match the model manifest"
            )

    gates = _mapping(report.get("gates"), "qualification.gates")
    if set(gates) != set(QUALIFICATION_GATES):
        raise SwapperContractError(
            "qualification.gates must contain every required release gate exactly once"
        )
    failed = sorted(name for name in QUALIFICATION_GATES if gates.get(name) is not True)
    if failed:
        raise SwapperContractError(
            "qualification report has unpassed release gates: " + ", ".join(failed)
        )
    return asset


def _optional_ncnn_spec(root: Path, value: Any) -> Native256NcnnSpec | None:
    if value is None:
        return None
    document = _mapping(value, "ncnn")
    backend = _name(document.get("backend"), "ncnn.backend")
    if backend != "ncnn-vulkan":
        raise SwapperContractError("ncnn.backend must be 'ncnn-vulkan'")
    fp16_storage = document.get("fp16_storage")
    if not isinstance(fp16_storage, bool):
        raise SwapperContractError("ncnn.fp16_storage must be a JSON boolean")

    conditioner_doc = _mapping(
        document.get("identity_conditioner"), "ncnn.identity_conditioner"
    )
    swapper_doc = _mapping(document.get("swapper"), "ncnn.swapper")
    conditioner = NcnnIdentityConditionerSpec(
        param=_paired_asset(
            root, conditioner_doc, "ncnn.identity_conditioner", "param"
        ),
        model=_paired_asset(
            root, conditioner_doc, "ncnn.identity_conditioner", "bin"
        ),
        input_name=_name(
            conditioner_doc.get("input"), "ncnn.identity_conditioner.input"
        ),
        output_name=_name(
            conditioner_doc.get("output"), "ncnn.identity_conditioner.output"
        ),
    )
    swapper = NcnnSwapperModelSpec(
        param=_paired_asset(root, swapper_doc, "ncnn.swapper", "param"),
        model=_paired_asset(root, swapper_doc, "ncnn.swapper", "bin"),
        target_input_name=_name(
            swapper_doc.get("target_input"), "ncnn.swapper.target_input"
        ),
        style_input_name=_name(
            swapper_doc.get("style_input"), "ncnn.swapper.style_input"
        ),
        candidate_output_name=_name(
            swapper_doc.get("candidate_output"), "ncnn.swapper.candidate_output"
        ),
        alpha_output_name=_name(
            swapper_doc.get("alpha_output"), "ncnn.swapper.alpha_output"
        ),
    )

    # The bridge intentionally exposes a small fixed ABI rather than accepting
    # arbitrary blob names from an editable manifest. The offline converter
    # verifies and records these pnnx names before a bundle can be installed.
    actual_names = (
        conditioner.input_name,
        conditioner.output_name,
        swapper.target_input_name,
        swapper.style_input_name,
        swapper.candidate_output_name,
        swapper.alpha_output_name,
    )
    expected_names = ("in0", "out0", "in0", "in1", "out0", "out1")
    if actual_names != expected_names:
        raise SwapperContractError(
            "ncnn blob names must match the native bridge ABI: "
            "conditioner in0/out0 and swapper in0/in1/out0/out1"
        )
    if conditioner.param.path.suffixes[-2:] != [".ncnn", ".param"]:
        raise SwapperContractError(
            "ncnn.identity_conditioner.param_file must end in .ncnn.param"
        )
    if conditioner.model.path.suffixes[-2:] != [".ncnn", ".bin"]:
        raise SwapperContractError(
            "ncnn.identity_conditioner.bin_file must end in .ncnn.bin"
        )
    if swapper.param.path.suffixes[-2:] != [".ncnn", ".param"]:
        raise SwapperContractError("ncnn.swapper.param_file must end in .ncnn.param")
    if swapper.model.path.suffixes[-2:] != [".ncnn", ".bin"]:
        raise SwapperContractError("ncnn.swapper.bin_file must end in .ncnn.bin")
    return Native256NcnnSpec(
        backend=backend,
        fp16_storage=fp16_storage,
        conditioner=conditioner,
        swapper=swapper,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SwapperContractError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SwapperContractError(f"{label} must be an integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SwapperContractError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SwapperContractError(f"{label} must be finite")
    return result


def _sequence(value: Any, label: str, length: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SwapperContractError(f"{label} must be an array")
    if len(value) != length:
        raise SwapperContractError(f"{label} must contain {length} values")
    return value


def _integer_pair(value: Any, label: str) -> tuple[int, int]:
    items = _sequence(value, label, 2)
    pair = (_integer(items[0], f"{label}[0]"), _integer(items[1], f"{label}[1]"))
    if pair[0] <= 0 or pair[1] <= 0:
        raise SwapperContractError(f"{label} values must be positive")
    return pair


def _number_pair(value: Any, label: str) -> tuple[float, float]:
    items = _sequence(value, label, 2)
    pair = (
        _finite_number(items[0], f"{label}[0]"),
        _finite_number(items[1], f"{label}[1]"),
    )
    if pair[0] >= pair[1]:
        raise SwapperContractError(f"{label}[0] must be less than {label}[1]")
    return pair


def _number_triplet(value: Any, label: str) -> tuple[float, float, float]:
    items = _sequence(value, label, 3)
    return tuple(
        _finite_number(item, f"{label}[{index}]")
        for index, item in enumerate(items)
    )  # type: ignore[return-value]


def _name(value: Any, label: str, *, model_id: bool = False) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SwapperContractError(f"{label} must be a non-empty trimmed string")
    if "\x00" in value:
        raise SwapperContractError(f"{label} must not contain a NUL byte")
    if model_id and not _MODEL_ID.fullmatch(value):
        raise SwapperContractError(f"{label} contains unsupported characters")
    return value


def _asset(root: Path, document: Mapping[str, Any], label: str) -> OnnxAsset:
    filename = _name(document.get("file"), f"{label}.file")
    if "://" in filename or Path(filename).is_absolute():
        raise SwapperContractError(f"{label}.file must be a relative local path")
    resolved = (root / filename).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise SwapperContractError(
            f"{label}.file must remain inside the manifest directory"
        ) from error
    digest = _name(document.get("sha256"), f"{label}.sha256").lower()
    if not _SHA256.fullmatch(digest):
        raise SwapperContractError(f"{label}.sha256 must contain 64 hexadecimal digits")
    return OnnxAsset(path=resolved, sha256=digest)


def _paired_asset(
    root: Path, document: Mapping[str, Any], label: str, kind: str
) -> OnnxAsset:
    """Read one ``param_file``/``bin_file`` pair using the normal asset rules."""
    if kind not in {"param", "bin"}:
        raise AssertionError(f"unsupported paired asset kind: {kind}")
    return _asset(
        root,
        {
            "file": document.get(f"{kind}_file"),
            "sha256": document.get(f"{kind}_sha256"),
        },
        f"{label}.{kind}",
    )


def _verify_asset(asset: OnnxAsset, label: str) -> None:
    if not asset.path.is_file():
        raise SwapperContractError(f"{label} model does not exist: {asset.path}")
    digest = hashlib.sha256()
    try:
        with asset.path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise SwapperContractError(f"could not hash {label} model: {error}") from error
    actual = digest.hexdigest()
    if actual != asset.sha256:
        raise SwapperContractError(
            f"{label} SHA-256 mismatch: expected {asset.sha256}, got {actual}"
        )


__all__ = [
    "IdentityConditionerSpec",
    "MANIFEST_SCHEMA_VERSION",
    "NATIVE_256_SIZE",
    "QUALITY_STATUSES",
    "QUALIFICATION_GATES",
    "Native256Manifest",
    "Native256NcnnSpec",
    "NcnnIdentityConditionerSpec",
    "NcnnSwapperModelSpec",
    "OnnxAsset",
    "PreprocessSpec",
    "SwapResult",
    "SwapperContractError",
    "SwapperModelSpec",
    "SwapperSpec",
    "mapped_inswapper_identity",
    "normalized_embedding",
    "verify_embedded_onnx",
]
