"""Offline adapter for the hash-pinned InStyleSwapper256 Version B model.

Version B is the balanced native-256 checkpoint: unlike a resized 128px
INSwapper result it generates all 256x256 pixels directly, while avoiding the
visible over-sharpened grid of Version C and the heavy smoothing of Version A.
The model embeds the INSwapper identity map as an unused initializer; it is
read once at startup so the identity conditioning exactly matches training.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import threading
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime

from modules.swapper_contract import SwapResult, SwapperContractError


_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _ROOT / "models" / "InStyleSwapper256_Version_B.fp16.onnx"

INSTYLE256_SHA256 = "0870b6c75eaea239bdd72b6c6d0910cb285310736e356c17a2cd67a961738116"
INSTYLE256_SIZE = 277_295_431
_INPUT_SIZE = (256, 256)
_TARGET_TEMPLATE = np.array(
    [
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453],
    ],
    dtype=np.float32,
)


class InStyle256Error(RuntimeError):
    """Base exception for the native-256 InStyle backend."""


class InStyle256LoadError(InStyle256Error):
    """The local model or ONNX contract is invalid."""


class InStyle256InferenceError(InStyle256Error):
    """A source identity or target frame could not be processed safely."""


def model_path() -> Path:
    return Path(os.environ.get("DLC_INSTYLE256_MODEL", _MODEL_PATH))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_asset(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise InStyle256LoadError(f"InStyleSwapper256 is unavailable: {error}") from error
    if size != INSTYLE256_SIZE:
        raise InStyle256LoadError(
            f"InStyleSwapper256 size mismatch: expected {INSTYLE256_SIZE}, got {size}"
        )
    try:
        digest = _sha256(path)
    except OSError as error:
        raise InStyle256LoadError(f"could not hash InStyleSwapper256: {error}") from error
    if digest != INSTYLE256_SHA256:
        raise InStyle256LoadError(
            f"InStyleSwapper256 SHA-256 mismatch: expected {INSTYLE256_SHA256}, got {digest}"
        )


def instyle256_available() -> bool:
    """Return whether the exact local model is ready; never use the network."""
    try:
        _verify_asset(model_path())
    except InStyle256LoadError:
        return False
    return True


def _node_shape(node: Any, label: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(value) for value in node.shape)
    except (TypeError, ValueError) as error:
        raise InStyle256LoadError(f"{label} must have a static shape") from error
    if not shape or any(value <= 0 for value in shape):
        raise InStyle256LoadError(f"{label} must have a static positive shape")
    if getattr(node, "type", "tensor(float)") != "tensor(float)":
        raise InStyle256LoadError(f"{label} must be float32")
    return shape


def _load_embedded_identity_map(path: Path) -> np.ndarray:
    try:
        graph = onnx.load(str(path), load_external_data=False)
        initializer = next(
            value for value in graph.graph.initializer if value.name == "emap"
        )
        identity_map = numpy_helper.to_array(initializer).astype(np.float32)
    except (OSError, StopIteration, ValueError, TypeError) as error:
        raise InStyle256LoadError(
            f"could not read the embedded InStyle identity map: {error}"
        ) from error
    finally:
        graph = None
    if identity_map.shape != (512, 512) or not np.isfinite(identity_map).all():
        raise InStyle256LoadError(
            f"embedded identity map must be finite [512,512], got {identity_map.shape}"
        )
    identity_map = np.ascontiguousarray(identity_map)
    identity_map.setflags(write=False)
    return identity_map


class InStyle256Swapper:
    """Native 256x256 InStyle generator backed by ONNX Runtime."""

    backend = "ort"
    model_id = "instyle-swapper-256-b"
    input_size = _INPUT_SIZE
    native_resolution = 256

    def __init__(
        self,
        *,
        providers: Iterable[Any] | None = None,
        session_factory: Callable[..., Any] | None = None,
        verify_assets: bool = True,
        identity_map: np.ndarray | None = None,
        identity_cache_entries: int = 8,
    ) -> None:
        if isinstance(identity_cache_entries, bool) or not isinstance(identity_cache_entries, int):
            raise ValueError("identity_cache_entries must be an integer")
        if not 1 <= identity_cache_entries <= 64:
            raise ValueError("identity_cache_entries must be between 1 and 64")
        self._model_path = model_path()
        if verify_assets:
            _verify_asset(self._model_path)

        if identity_map is None:
            identity_map = _load_embedded_identity_map(self._model_path)
        else:
            identity_map = np.asarray(identity_map, dtype=np.float32)
            if identity_map.shape != (512, 512) or not np.isfinite(identity_map).all():
                raise InStyle256LoadError("identity_map must be finite float32 [512,512]")
            identity_map = np.ascontiguousarray(identity_map)
            identity_map.setflags(write=False)
        self._identity_map = identity_map

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        factory = session_factory or onnxruntime.InferenceSession
        provider_config = None if providers is None else list(providers)
        try:
            self._session = factory(
                str(self._model_path), sess_options=options, providers=provider_config
            )
            self._validate_contract()
        except Exception as error:
            self._session = None
            if isinstance(error, InStyle256LoadError):
                raise
            raise InStyle256LoadError(
                f"could not initialize InStyleSwapper256: {error}"
            ) from error

        active = getattr(self._session, "get_providers", lambda: [])()
        self.device_name = ", ".join(str(value) for value in active) or "ONNX Runtime"
        self._identity_cache_entries = identity_cache_entries
        self._identity_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._identity_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._closed = False

    def _validate_contract(self) -> None:
        inputs = {node.name: node for node in self._session.get_inputs()}
        outputs = list(self._session.get_outputs())
        if set(inputs) != {"source", "target"}:
            raise InStyle256LoadError(
                f"InStyleSwapper256 inputs are incompatible: {sorted(inputs)}"
            )
        if _node_shape(inputs["source"], "InStyle source input") != (1, 512):
            raise InStyle256LoadError("InStyle source input must be [1,512]")
        if _node_shape(inputs["target"], "InStyle target input") != (1, 3, 256, 256):
            raise InStyle256LoadError("InStyle target input must be [1,3,256,256]")
        if len(outputs) != 1 or _node_shape(outputs[0], "InStyle output") != (1, 3, 256, 256):
            raise InStyle256LoadError("InStyle output must be [1,3,256,256]")
        self._output_name = outputs[0].name

    def close(self) -> None:
        with self._identity_lock:
            self._closed = True
            self._identity_cache.clear()
            self._session = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise InStyle256InferenceError("InStyleSwapper256 is closed")

    def prepare_source(self, source_face: Any) -> np.ndarray:
        self._ensure_open()
        raw = getattr(source_face, "normed_embedding", None)
        if raw is None:
            raise InStyle256InferenceError("source face has no normalized embedding")
        try:
            embedding = np.ascontiguousarray(np.asarray(raw, dtype=np.float32).reshape(1, 512))
        except (TypeError, ValueError) as error:
            raise InStyle256InferenceError(f"invalid source embedding: {error}") from error
        if not np.isfinite(embedding).all():
            raise InStyle256InferenceError("source embedding contains a non-finite value")
        key = hashlib.sha256(embedding.tobytes(order="C")).digest()
        with self._identity_lock:
            cached = self._identity_cache.get(key)
            if cached is not None:
                self._identity_cache.move_to_end(key)
                return cached
            latent = np.dot(embedding, self._identity_map)
            norm = float(np.linalg.norm(latent))
            if not np.isfinite(latent).all() or not np.isfinite(norm) or norm < 1e-8:
                raise InStyle256InferenceError("InStyle identity map produced an invalid latent")
            latent = np.ascontiguousarray(latent / np.float32(norm), dtype=np.float32)
            latent.setflags(write=False)
            self._identity_cache[key] = latent
            while len(self._identity_cache) > self._identity_cache_entries:
                self._identity_cache.popitem(last=False)
            return latent

    def _prepare_target(self, image: np.ndarray, target_face: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise InStyle256InferenceError("target image must be a uint8 array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InStyle256InferenceError("target image must be HxWx3 BGR")
        try:
            points = np.asarray(target_face.kps, dtype=np.float32)
        except (AttributeError, TypeError, ValueError) as error:
            raise InStyle256InferenceError(f"invalid target keypoints: {error}") from error
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise InStyle256InferenceError(
                f"target keypoints must be finite [5,2], got {points.shape}"
            )
        destination = _TARGET_TEMPLATE * np.asarray(_INPUT_SIZE, dtype=np.float32)
        affine = cv2.estimateAffinePartial2D(
            points, destination, method=cv2.RANSAC, ransacReprojThreshold=100
        )[0]
        if affine is None or affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise InStyle256InferenceError("256px face alignment failed")
        aligned = cv2.warpAffine(
            image,
            affine,
            _INPUT_SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rgb = aligned[:, :, ::-1].astype(np.float32) * np.float32(1.0 / 255.0)
        target = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
        return target, np.ascontiguousarray(affine, dtype=np.float32)

    def infer(
        self,
        image: np.ndarray,
        target_face: Any,
        source_face: Any,
        *,
        paste_back: bool = False,
    ) -> SwapResult:
        if paste_back:
            raise ValueError("InStyle256Swapper returns aligned output for the common paste path")
        self._ensure_open()
        target, affine = self._prepare_target(image, target_face)
        source = self.prepare_source(source_face)
        try:
            with self._inference_lock:
                candidate = self._session.run(
                    [self._output_name], {"target": target, "source": source}
                )[0]
        except Exception as error:
            raise InStyle256InferenceError(f"InStyleSwapper256 inference failed: {error}") from error
        candidate = np.asarray(candidate, dtype=np.float32)
        if candidate.shape != (1, 3, 256, 256) or not np.isfinite(candidate).all():
            raise InStyle256InferenceError(
                f"InStyleSwapper256 returned an invalid candidate: {candidate.shape}"
            )
        face_bgr = np.ascontiguousarray(
            np.clip(candidate[0].transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
        )
        try:
            return SwapResult(
                face_bgr=face_bgr,
                affine=affine,
                alpha=None,
                model_id=self.model_id,
                backend=self.backend,
            )
        except SwapperContractError as error:
            raise InStyle256InferenceError(str(error)) from error

    def get(
        self,
        image: np.ndarray,
        target_face: Any,
        source_face: Any,
        paste_back: bool = False,
    ) -> SwapResult:
        return self.infer(image, target_face, source_face, paste_back=paste_back)


__all__ = [
    "INSTYLE256_SHA256",
    "INSTYLE256_SIZE",
    "InStyle256Error",
    "InStyle256InferenceError",
    "InStyle256LoadError",
    "InStyle256Swapper",
    "instyle256_available",
    "model_path",
]
