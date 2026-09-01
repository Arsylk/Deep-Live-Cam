"""ONNX Runtime adapter for the local two-stage native-256 swapper.

The source embedding is projected through the bundle's INSwapper identity map
and normalized exactly once.  The identity conditioner is evaluated once for
each distinct mapped latent.  Its cached style tensor is then supplied to the
frame-by-frame swapper, which returns an uncomposited RGB candidate and semantic
alpha mask.

This module never downloads assets and never falls back to another model.  A
missing, corrupt, incompatible, or numerically invalid model raises a specific
error so the caller can leave the current camera frame untouched.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import threading
from typing import Any, Callable, Iterable

import numpy as np
import onnxruntime
from insightface.utils import face_align

from modules.swapper_contract import (
    Native256Manifest,
    SwapResult,
    SwapperContractError,
    mapped_inswapper_identity,
    verify_embedded_onnx,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _ROOT / "models" / "swappers" / "native_256" / "manifest.json"
_RANGE_TOLERANCE = 1e-4


class Native256SwapperError(RuntimeError):
    """Base error for native-256 session setup and execution."""


class Native256LoadError(Native256SwapperError):
    """The explicitly requested native-256 model could not be opened."""


class Native256InferenceError(Native256SwapperError):
    """The native-256 model produced no safe result for the current frame."""


def default_manifest_path() -> Path:
    """Return the configured local manifest path without probing the network."""
    return Path(os.environ.get("DLC_NATIVE256_MANIFEST", _DEFAULT_MANIFEST))


def native256_swapper_available(
    manifest_path: str | Path | None = None,
    *,
    require_qualified: bool = False,
) -> bool:
    """Return whether a complete local bundle meets the requested quality."""
    try:
        manifest = load_native256_manifest(
            manifest_path, require_qualified=require_qualified
        )
        verify_embedded_onnx(
            manifest.conditioner.asset, "identity conditioner"
        )
        verify_embedded_onnx(manifest.swapper.asset, "native-256 swapper")
    except (Native256LoadError, SwapperContractError):
        return False
    return True


def load_native256_manifest(
    manifest_path: str | Path | None = None,
    *,
    require_qualified: bool = False,
) -> Native256Manifest:
    """Load a local manifest and optionally enforce auto-selection quality.

    Explicit native-256 selection should use the default ``False`` value, while
    an automatic model resolver must pass ``require_qualified=True``.
    """
    try:
        manifest = Native256Manifest.load(manifest_path or default_manifest_path())
    except SwapperContractError as error:
        raise Native256LoadError(str(error)) from error
    if require_qualified and not (
        manifest.quality_status == "qualified"
        and manifest.auto_select_eligible
    ):
        raise Native256LoadError(
            "native-256 model is not eligible for automatic selection "
            f"(quality_status={manifest.quality_status}, "
            f"auto_select_eligible={manifest.auto_select_eligible})"
        )
    return manifest


def native256_swapper_quality_status(
    manifest_path: str | Path | None = None,
) -> str | None:
    """Return a valid local bundle's quality status, or ``None`` if invalid."""
    try:
        return load_native256_manifest(manifest_path).quality_status
    except Native256LoadError:
        return None


class Native256Swapper:
    """Two-session, semantic-mask native-256 swapper backed by ORT."""

    backend = "ort"

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        providers: Iterable[Any] | None = None,
        style_cache_entries: int = 8,
        session_factory: Callable[..., Any] | None = None,
        require_qualified: bool = False,
    ) -> None:
        if isinstance(style_cache_entries, bool) or not isinstance(
            style_cache_entries, int
        ):
            raise ValueError("style_cache_entries must be an integer")
        if not 1 <= style_cache_entries <= 64:
            raise ValueError("style_cache_entries must be between 1 and 64")

        self.manifest = load_native256_manifest(
            manifest_path, require_qualified=require_qualified
        )
        if session_factory is None:
            try:
                verify_embedded_onnx(
                    self.manifest.conditioner.asset, "identity conditioner"
                )
                verify_embedded_onnx(self.manifest.swapper.asset, "native-256 swapper")
            except SwapperContractError as error:
                raise Native256LoadError(str(error)) from error

        self.spec = self.manifest.spec
        self.quality_status = self.manifest.quality_status
        self.input_size = self.spec.input_size
        self._style_cache_entries = style_cache_entries
        self._style_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._style_lock = threading.Lock()
        self._closed = False

        try:
            identity_map = np.load(
                self.manifest.identity_map.path,
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            raise Native256LoadError(f"could not load identity map: {error}") from error
        if identity_map.dtype != np.float32 or identity_map.shape != (512, 512):
            raise Native256LoadError(
                "identity map must be float32 with shape (512, 512), got "
                f"{identity_map.dtype} {identity_map.shape}"
            )
        if not np.isfinite(identity_map).all():
            raise Native256LoadError("identity map contains a non-finite value")
        self._identity_map = np.ascontiguousarray(identity_map)
        self._identity_map.setflags(write=False)

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        provider_config = None if providers is None else list(providers)
        factory = session_factory or onnxruntime.InferenceSession
        try:
            self._conditioner_session = factory(
                str(self.manifest.conditioner.asset.path),
                sess_options=options,
                providers=provider_config,
            )
            self._swapper_session = factory(
                str(self.manifest.swapper.asset.path),
                sess_options=options,
                providers=provider_config,
            )
            (
                self._conditioner_input_shape,
                self._style_shape,
            ) = self._validate_session_contracts()
        except Exception as error:
            self._closed = True
            self._conditioner_session = None
            self._swapper_session = None
            if isinstance(error, Native256LoadError):
                raise
            raise Native256LoadError(
                f"could not initialize native-256 ONNX sessions: {error}"
            ) from error

        active_providers = self._active_providers(self._swapper_session)
        self.device_name = ", ".join(active_providers) or "ONNX Runtime"

    @staticmethod
    def _active_providers(session: Any) -> list[str]:
        getter = getattr(session, "get_providers", None)
        if not callable(getter):
            return []
        try:
            return [str(provider) for provider in getter()]
        except Exception:
            return []

    def _validate_session_contracts(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        conditioner = self.manifest.conditioner
        swapper = self.manifest.swapper
        embedding_shape = (1, self.spec.embedding_size)
        target_shape = (1, 3, self.input_size[1], self.input_size[0])
        alpha_shape = (1, 1, self.input_size[1], self.input_size[0])

        conditioner_inputs = _node_map(
            self._conditioner_session.get_inputs(), "identity conditioner inputs"
        )
        conditioner_outputs = _node_map(
            self._conditioner_session.get_outputs(), "identity conditioner outputs"
        )
        if set(conditioner_inputs) != {conditioner.input_name}:
            raise Native256LoadError(
                "identity conditioner inputs differ from the manifest: "
                f"{sorted(conditioner_inputs)}"
            )
        if set(conditioner_outputs) != {conditioner.output_name}:
            raise Native256LoadError(
                "identity conditioner outputs differ from the manifest: "
                f"{sorted(conditioner_outputs)}"
            )
        conditioner_input_shape = _static_shape(
            conditioner_inputs[conditioner.input_name],
            "identity conditioner input",
        )
        if conditioner_input_shape not in {
            embedding_shape,
            (1, self.spec.embedding_size, 1, 1),
        }:
            raise Native256LoadError(
                "identity conditioner input must have shape "
                f"{embedding_shape} or "
                f"{(1, self.spec.embedding_size, 1, 1)}, got "
                f"{conditioner_input_shape}"
            )
        _require_float32_node(
            conditioner_inputs[conditioner.input_name],
            "identity conditioner input",
            conditioner_input_shape,
        )
        style_shape = _static_shape(
            conditioner_outputs[conditioner.output_name],
            "identity conditioner output",
        )
        _require_float32_node(
            conditioner_outputs[conditioner.output_name],
            "identity conditioner output",
            style_shape,
        )
        if len(style_shape) < 2 or style_shape[0] != 1:
            raise Native256LoadError(
                f"conditioned style must have a static batch-one shape, got {style_shape}"
            )

        swapper_inputs = _node_map(
            self._swapper_session.get_inputs(), "native-256 swapper inputs"
        )
        swapper_outputs = _node_map(
            self._swapper_session.get_outputs(), "native-256 swapper outputs"
        )
        expected_inputs = {swapper.target_input_name, swapper.style_input_name}
        expected_outputs = {
            swapper.candidate_output_name,
            swapper.alpha_output_name,
        }
        if set(swapper_inputs) != expected_inputs:
            raise Native256LoadError(
                "native-256 swapper inputs differ from the manifest: "
                f"{sorted(swapper_inputs)}"
            )
        if set(swapper_outputs) != expected_outputs:
            raise Native256LoadError(
                "native-256 swapper outputs differ from the manifest: "
                f"{sorted(swapper_outputs)}"
            )
        _require_float32_node(
            swapper_inputs[swapper.target_input_name],
            "native-256 target input",
            target_shape,
        )
        _require_float32_node(
            swapper_inputs[swapper.style_input_name],
            "native-256 style input",
            style_shape,
        )
        _require_float32_node(
            swapper_outputs[swapper.candidate_output_name],
            "native-256 candidate output",
            target_shape,
        )
        _require_float32_node(
            swapper_outputs[swapper.alpha_output_name],
            "native-256 alpha output",
            alpha_shape,
        )
        return conditioner_input_shape, style_shape

    def close(self) -> None:
        """Release both sessions and all source-conditioned style tensors."""
        with self._style_lock:
            self._style_cache.clear()
            self._closed = True
            self._conditioner_session = None
            self._swapper_session = None

    def clear_style_cache(self) -> None:
        with self._style_lock:
            self._style_cache.clear()

    @property
    def style_cache_size(self) -> int:
        with self._style_lock:
            return len(self._style_cache)

    def _ensure_open(self) -> None:
        if self._closed:
            raise Native256InferenceError("native-256 swapper is closed")

    def _style_for_embedding(self, embedding: np.ndarray) -> np.ndarray:
        key = hashlib.sha256(embedding.tobytes(order="C")).digest()
        with self._style_lock:
            self._ensure_open()
            cached = self._style_cache.get(key)
            if cached is not None:
                self._style_cache.move_to_end(key)
                return cached
            try:
                conditioner_input = np.ascontiguousarray(
                    embedding.reshape(self._conditioner_input_shape)
                )
                outputs = self._conditioner_session.run(
                    [self.manifest.conditioner.output_name],
                    {self.manifest.conditioner.input_name: conditioner_input},
                )
            except Exception as error:
                raise Native256InferenceError(
                    f"identity conditioner failed: {error}"
                ) from error
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
                raise Native256InferenceError(
                    "identity conditioner did not return its one named output"
                )
            style = _runtime_tensor(
                outputs[0], self._style_shape, "conditioned style"
            )
            style.setflags(write=False)
            self._style_cache[key] = style
            self._style_cache.move_to_end(key)
            while len(self._style_cache) > self._style_cache_entries:
                self._style_cache.popitem(last=False)
            return style

    def prepare_source(self, source_face: Any) -> np.ndarray:
        """Condition and cache the style tensor for one source identity."""
        self._ensure_open()
        if source_face is None or getattr(source_face, "normed_embedding", None) is None:
            raise Native256InferenceError(
                "source face does not contain a normalized identity embedding"
            )
        try:
            embedding = mapped_inswapper_identity(
                source_face.normed_embedding, self._identity_map
            )
        except SwapperContractError as error:
            raise Native256InferenceError(str(error)) from error
        return self._style_for_embedding(embedding)

    def _prepare_target(self, image: np.ndarray, target_face: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(image, np.ndarray):
            raise Native256InferenceError("target image must be a numpy array")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise Native256InferenceError(
                f"target image must be uint8 HxWx3 BGR, got {image.dtype} {image.shape}"
            )
        if target_face is None or getattr(target_face, "kps", None) is None:
            raise Native256InferenceError("target face does not contain five keypoints")
        try:
            keypoints = np.asarray(target_face.kps, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise Native256InferenceError(f"invalid target keypoints: {error}") from error
        if keypoints.shape != (5, 2) or not np.isfinite(keypoints).all():
            raise Native256InferenceError(
                f"target keypoints must be finite with shape (5, 2), got {keypoints.shape}"
            )
        try:
            aligned, affine = face_align.norm_crop2(
                image, keypoints, self.input_size[0]
            )
        except Exception as error:
            raise Native256InferenceError(f"256px face alignment failed: {error}") from error
        if (
            not isinstance(aligned, np.ndarray)
            or aligned.dtype != np.uint8
            or aligned.shape != (self.input_size[1], self.input_size[0], 3)
        ):
            shape = getattr(aligned, "shape", None)
            dtype = getattr(aligned, "dtype", None)
            raise Native256InferenceError(
                f"alignment returned an invalid crop: {dtype} {shape}"
            )
        affine = np.ascontiguousarray(affine, dtype=np.float32)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise Native256InferenceError(
                f"alignment returned an invalid affine matrix: {affine.shape}"
            )

        rgb = aligned[:, :, ::-1].astype(np.float32)
        preprocess = self.manifest.preprocess
        rgb *= np.float32(preprocess.scale)
        mean = np.asarray(preprocess.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(preprocess.std, dtype=np.float32).reshape(1, 1, 3)
        rgb = (rgb - mean) / std
        target = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
        return target, affine

    def infer(
        self,
        image: np.ndarray,
        target_face: Any,
        source_face: Any,
        *,
        paste_back: bool = False,
    ) -> SwapResult:
        """Generate one aligned RGB candidate and semantic alpha mask."""
        if paste_back:
            raise ValueError(
                "Native256Swapper returns aligned output for the application's paste path"
            )
        self._ensure_open()
        target, affine = self._prepare_target(image, target_face)
        style = self.prepare_source(source_face)
        swapper = self.manifest.swapper
        try:
            outputs = self._swapper_session.run(
                [swapper.candidate_output_name, swapper.alpha_output_name],
                {
                    swapper.target_input_name: target,
                    swapper.style_input_name: style,
                },
            )
        except Exception as error:
            raise Native256InferenceError(f"native-256 inference failed: {error}") from error
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
            raise Native256InferenceError(
                "native-256 swapper did not return candidate and alpha outputs"
            )

        candidate = _runtime_tensor(
            outputs[0],
            (1, 3, self.input_size[1], self.input_size[0]),
            "native-256 candidate",
        )
        alpha = _runtime_tensor(
            outputs[1],
            (1, 1, self.input_size[1], self.input_size[0]),
            "native-256 alpha",
        )
        candidate = _normalize_range(
            candidate, self.manifest.candidate_range, "native-256 candidate"
        )
        alpha = _normalize_range(
            alpha, self.manifest.alpha_range, "native-256 alpha"
        )

        candidate_rgb = candidate[0].transpose(1, 2, 0)
        face_bgr = np.ascontiguousarray(
            np.clip(candidate_rgb[:, :, ::-1] * 255.0, 0.0, 255.0).astype(
                np.uint8
            )
        )
        alpha_u8 = np.ascontiguousarray(
            np.rint(np.clip(alpha[0, 0] * 255.0, 0.0, 255.0)).astype(np.uint8)
        )
        try:
            return SwapResult(
                face_bgr=face_bgr,
                affine=affine,
                alpha=alpha_u8,
                model_id=self.spec.model_id,
                backend=self.backend,
            )
        except SwapperContractError as error:
            raise Native256InferenceError(str(error)) from error

    def get(
        self,
        image: np.ndarray,
        target_face: Any,
        source_face: Any,
        paste_back: bool = False,
    ) -> SwapResult:
        """Compatibility spelling for callers that currently invoke ``get``."""
        return self.infer(
            image,
            target_face,
            source_face,
            paste_back=paste_back,
        )


def _node_map(nodes: Iterable[Any], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in nodes:
        name = getattr(node, "name", None)
        if not isinstance(name, str) or not name:
            raise Native256LoadError(f"{label} contains an unnamed tensor")
        if name in result:
            raise Native256LoadError(f"{label} contains duplicate tensor {name!r}")
        result[name] = node
    return result


def _static_shape(node: Any, label: str) -> tuple[int, ...]:
    shape = getattr(node, "shape", None)
    if not isinstance(shape, (list, tuple)) or not shape:
        raise Native256LoadError(f"{label} does not have a static shape")
    dimensions: list[int] = []
    for value in shape:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Native256LoadError(
                f"{label} must have positive static dimensions, got {shape}"
            )
        dimensions.append(value)
    return tuple(dimensions)


def _require_float32_node(
    node: Any, label: str, expected_shape: tuple[int, ...]
) -> None:
    tensor_type = getattr(node, "type", None)
    if tensor_type != "tensor(float)":
        raise Native256LoadError(
            f"{label} must be tensor(float), got {tensor_type!r}"
        )
    actual_shape = _static_shape(node, label)
    if actual_shape != expected_shape:
        raise Native256LoadError(
            f"{label} must have shape {expected_shape}, got {actual_shape}"
        )


def _runtime_tensor(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise Native256InferenceError(f"{label} is not a numpy array")
    if value.dtype != np.float32 or value.shape != shape:
        raise Native256InferenceError(
            f"{label} must be float32 with shape {shape}, got "
            f"{value.dtype} {value.shape}"
        )
    if not np.isfinite(value).all():
        raise Native256InferenceError(f"{label} contains a non-finite value")
    return np.ascontiguousarray(value)


def _normalize_range(
    value: np.ndarray, value_range: tuple[float, float], label: str
) -> np.ndarray:
    low, high = value_range
    minimum = float(value.min())
    maximum = float(value.max())
    if minimum < low - _RANGE_TOLERANCE or maximum > high + _RANGE_TOLERANCE:
        raise Native256InferenceError(
            f"{label} is outside its declared range {value_range}: "
            f"[{minimum}, {maximum}]"
        )
    normalized = (np.clip(value, low, high) - low) / (high - low)
    return np.ascontiguousarray(normalized, dtype=np.float32)


__all__ = [
    "Native256InferenceError",
    "Native256LoadError",
    "Native256Swapper",
    "Native256SwapperError",
    "default_manifest_path",
    "load_native256_manifest",
    "native256_swapper_available",
    "native256_swapper_quality_status",
]
