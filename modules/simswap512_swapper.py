"""Offline, hash-pinned native-512 SimSwap adapter.

This backend uses the model pair published by VisoMaster Fusion: the 512px
generator and the dedicated SimSwap ArcFace recognizer it was trained with.
Using the regular INSwapper/InsightFace embedding (or converting it after the
fact) produces visibly malformed identities, so the source image is aligned
and embedded by the matching recognizer exactly once and then cached.

The runtime never downloads assets.  Both ONNX files must already be present
in ``models`` and match the exact sizes and SHA-256 digests below.
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
import onnxruntime

from modules.swapper_contract import SwapResult, SwapperContractError


_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _ROOT / "models" / "simswap_512_unoff.onnx"
_ARCFACE_PATH = _ROOT / "models" / "simswap_arcface_model.onnx"

SIMSWAP512_SHA256 = "08c6ca9c0a65eff119bea42686a4574337141de304b9d26e2f9d11e78d9e8e86"
SIMSWAP_ARCFACE_SHA256 = "58949c864ab4a89012aaefc117f1ab8548c5f470bbc3889474bca13a412fc843"
SIMSWAP512_SIZE = 239_249_146
SIMSWAP_ARCFACE_SIZE = 208_969_332

_INPUT_SIZE = (512, 512)
_ARCFACE_SIZE = (112, 112)

# The generator is trained on an ArcFace-128 crop enlarged to 512px.  This is
# deliberately not the ArcFace-112-v1 template used by another unofficial
# export of SimSwap-512.
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

_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Pose-aware templates used by the paired recognizer.  They prevent a strongly
# angled source photo from being flattened into a corrupt identity embedding.
_POSE_TEMPLATES = np.array(
    [
        [[51.642, 50.115], [57.617, 49.990], [35.740, 69.007], [51.157, 89.050], [57.025, 89.702]],
        [[45.031, 50.118], [65.568, 50.872], [39.677, 68.111], [45.177, 86.190], [64.246, 86.758]],
        [[39.730, 51.138], [72.270, 51.138], [56.000, 68.493], [42.463, 87.010], [69.537, 87.010]],
        [[46.845, 50.872], [67.382, 50.118], [72.737, 68.111], [48.167, 86.758], [67.236, 86.190]],
        [[54.796, 49.990], [60.771, 50.115], [76.673, 69.007], [55.388, 89.702], [61.257, 89.050]],
        [[39.730, 55.000], [72.270, 55.000], [56.000, 64.000], [42.463, 78.000], [69.537, 78.000]],
        [[39.730, 45.000], [72.270, 45.000], [56.000, 75.000], [42.463, 95.000], [69.537, 95.000]],
    ],
    dtype=np.float32,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SimSwap512Error(RuntimeError):
    """Base exception for the native-512 backend."""


class SimSwap512LoadError(SimSwap512Error):
    """The local asset or ONNX session contract is invalid."""


class SimSwap512InferenceError(SimSwap512Error):
    """A frame could not be processed safely."""


def model_path() -> Path:
    return Path(os.environ.get("DLC_SIMSWAP512_MODEL", _MODEL_PATH))


def arcface_path() -> Path:
    return Path(os.environ.get("DLC_SIMSWAP512_ARCFACE", _ARCFACE_PATH))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_asset(path: Path, size: int, digest: str, label: str) -> None:
    try:
        stat = path.stat()
    except OSError as error:
        raise SimSwap512LoadError(f"{label} is unavailable: {error}") from error
    if stat.st_size != size:
        raise SimSwap512LoadError(
            f"{label} size mismatch: expected {size}, got {stat.st_size}"
        )
    try:
        actual = _sha256(path)
    except OSError as error:
        raise SimSwap512LoadError(f"could not hash {label}: {error}") from error
    if actual != digest:
        raise SimSwap512LoadError(
            f"{label} SHA-256 mismatch: expected {digest}, got {actual}"
        )


def simswap512_available() -> bool:
    """Return whether the exact local model pair is ready; never download."""
    try:
        _verify_asset(model_path(), SIMSWAP512_SIZE, SIMSWAP512_SHA256, "SimSwap-512")
        _verify_asset(
            arcface_path(),
            SIMSWAP_ARCFACE_SIZE,
            SIMSWAP_ARCFACE_SHA256,
            "SimSwap ArcFace recognizer",
        )
    except SimSwap512LoadError:
        return False
    return True


def _node_shape(node: Any, label: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(value) for value in node.shape)
    except (TypeError, ValueError) as error:
        raise SimSwap512LoadError(f"{label} must have a static shape") from error
    if not shape or any(value <= 0 for value in shape):
        raise SimSwap512LoadError(f"{label} must have a static positive shape")
    if getattr(node, "type", "tensor(float)") != "tensor(float)":
        raise SimSwap512LoadError(f"{label} must be float32")
    return shape


def _face_keypoints(face: Any, label: str) -> np.ndarray:
    try:
        points = np.asarray(face.kps, dtype=np.float32)
    except (AttributeError, TypeError, ValueError) as error:
        raise SimSwap512InferenceError(f"invalid {label} keypoints: {error}") from error
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise SimSwap512InferenceError(
            f"{label} keypoints must be finite [5,2], got {points.shape}"
        )
    return points


def _estimate_affine(points: np.ndarray, destination: np.ndarray, label: str) -> np.ndarray:
    matrix = cv2.estimateAffinePartial2D(
        points,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )[0]
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise SimSwap512InferenceError(f"{label} face alignment failed")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _yaw_pitch(points: np.ndarray) -> tuple[float, float]:
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    left_distance = float(nose[0] - left_eye[0])
    right_distance = float(right_eye[0] - nose[0])
    yaw = (left_distance - right_distance) / (left_distance + right_distance + 1e-6) * 90.0
    eye_y = float((left_eye[1] + right_eye[1]) * 0.5)
    mouth_y = float((left_mouth[1] + right_mouth[1]) * 0.5)
    ratio = float(nose[1] - eye_y) / (mouth_y - float(nose[1]) + 1e-6)
    return yaw, (1.0 - ratio) * 90.0


def _source_template(points: np.ndarray) -> np.ndarray:
    yaw, pitch = _yaw_pitch(points)
    if abs(yaw) <= 30.0 and abs(pitch) <= 30.0:
        return _ARCFACE_TEMPLATE
    best_template = _POSE_TEMPLATES[0]
    best_error = float("inf")
    for template in _POSE_TEMPLATES:
        try:
            matrix = _estimate_affine(points, template, "source")
        except SimSwap512InferenceError:
            continue
        transformed = cv2.transform(points[None, :, :], matrix)[0]
        error = float(np.linalg.norm(transformed - template, axis=1).sum())
        if error < best_error:
            best_error = error
            best_template = template
    return best_template


class SimSwap512Swapper:
    """Native 512x512 generator with its matching source recognizer."""

    backend = "ort"
    model_id = "simswap-512-visomaster"
    input_size = _INPUT_SIZE
    native_resolution = 512

    def __init__(
        self,
        *,
        providers: Iterable[Any] | None = None,
        session_factory: Callable[..., Any] | None = None,
        verify_assets: bool = True,
        identity_cache_entries: int = 8,
    ) -> None:
        if isinstance(identity_cache_entries, bool) or not isinstance(identity_cache_entries, int):
            raise ValueError("identity_cache_entries must be an integer")
        if not 1 <= identity_cache_entries <= 64:
            raise ValueError("identity_cache_entries must be between 1 and 64")

        self._model_path = model_path()
        self._arcface_path = arcface_path()
        if verify_assets:
            _verify_asset(self._model_path, SIMSWAP512_SIZE, SIMSWAP512_SHA256, "SimSwap-512")
            _verify_asset(
                self._arcface_path,
                SIMSWAP_ARCFACE_SIZE,
                SIMSWAP_ARCFACE_SHA256,
                "SimSwap ArcFace recognizer",
            )

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        factory = session_factory or onnxruntime.InferenceSession
        provider_config = None if providers is None else list(providers)
        try:
            self._arcface = factory(
                str(self._arcface_path), sess_options=options, providers=provider_config
            )
            self._swapper = factory(
                str(self._model_path), sess_options=options, providers=provider_config
            )
            self._validate_contract()
        except Exception as error:
            self._arcface = None
            self._swapper = None
            if isinstance(error, SimSwap512LoadError):
                raise
            raise SimSwap512LoadError(
                f"could not initialize SimSwap-512 ONNX sessions: {error}"
            ) from error

        active = getattr(self._swapper, "get_providers", lambda: [])()
        self.device_name = ", ".join(str(value) for value in active) or "ONNX Runtime"
        self._identity_cache_entries = identity_cache_entries
        self._identity_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._identity_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._closed = False

    def _validate_contract(self) -> None:
        arcface_inputs = list(self._arcface.get_inputs())
        arcface_outputs = list(self._arcface.get_outputs())
        if len(arcface_inputs) != 1 or len(arcface_outputs) != 1:
            raise SimSwap512LoadError("SimSwap ArcFace must expose one input and one output")
        if _node_shape(arcface_inputs[0], "SimSwap ArcFace input") != (1, 3, 112, 112):
            raise SimSwap512LoadError("SimSwap ArcFace input must be [1,3,112,112]")
        if _node_shape(arcface_outputs[0], "SimSwap ArcFace output") != (1, 512):
            raise SimSwap512LoadError("SimSwap ArcFace output must be [1,512]")
        self._arcface_input = arcface_inputs[0].name
        self._arcface_output = arcface_outputs[0].name

        inputs = {node.name: node for node in self._swapper.get_inputs()}
        outputs = list(self._swapper.get_outputs())
        if set(inputs) != {"input", "onnx::Gemm_1"}:
            raise SimSwap512LoadError(
                f"SimSwap-512 inputs are incompatible: {sorted(inputs)}"
            )
        if _node_shape(inputs["input"], "SimSwap target input") != (1, 3, 512, 512):
            raise SimSwap512LoadError("SimSwap target input must be [1,3,512,512]")
        if _node_shape(inputs["onnx::Gemm_1"], "SimSwap source input") != (1, 512):
            raise SimSwap512LoadError("SimSwap source input must be [1,512]")
        if len(outputs) != 1 or _node_shape(outputs[0], "SimSwap output") != (1, 3, 512, 512):
            raise SimSwap512LoadError("SimSwap output must be [1,3,512,512]")
        self._swapper_output = outputs[0].name

    def close(self) -> None:
        with self._identity_lock:
            self._closed = True
            self._identity_cache.clear()
            self._arcface = None
            self._swapper = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise SimSwap512InferenceError("SimSwap-512 is closed")

    @staticmethod
    def _source_frame(source_face: Any) -> np.ndarray:
        for attribute in ("_dlc_source_frame", "source_frame"):
            frame = getattr(source_face, attribute, None)
            if isinstance(frame, np.ndarray):
                return frame
        try:
            import modules.globals
            from modules import imread_unicode

            path = getattr(modules.globals, "source_path", None)
            frame = imread_unicode(path) if path else None
        except (ImportError, OSError, TypeError, ValueError):
            frame = None
        if not isinstance(frame, np.ndarray):
            raise SimSwap512InferenceError(
                "the matching SimSwap recognizer needs the selected source image"
            )
        return frame

    def prepare_source(self, source_face: Any) -> np.ndarray:
        self._ensure_open()
        points = _face_keypoints(source_face, "source")
        existing = getattr(source_face, "embedding", None)
        key_material = points.tobytes(order="C")
        if existing is not None:
            try:
                key_material += np.asarray(existing, dtype=np.float32).reshape(512).tobytes(order="C")
            except (TypeError, ValueError):
                pass
        key = hashlib.sha256(key_material).digest()
        with self._identity_lock:
            cached = self._identity_cache.get(key)
            if cached is not None:
                self._identity_cache.move_to_end(key)
                return cached

            frame = self._source_frame(source_face)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise SimSwap512InferenceError("source image must be HxWx3 uint8 BGR")
            matrix = _estimate_affine(points, _source_template(points), "source")
            crop = cv2.warpAffine(
                frame,
                matrix,
                _ARCFACE_SIZE,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            rgb = crop[:, :, ::-1].astype(np.float32) * np.float32(1.0 / 255.0)
            rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            recognizer_input = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
            try:
                embedding = self._arcface.run(
                    [self._arcface_output], {self._arcface_input: recognizer_input}
                )[0]
            except Exception as error:
                raise SimSwap512InferenceError(
                    f"SimSwap source recognition failed: {error}"
                ) from error
            embedding = np.asarray(embedding, dtype=np.float32).reshape(1, 512)
            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(embedding).all() or not np.isfinite(norm) or norm < 1e-8:
                raise SimSwap512InferenceError("SimSwap recognizer produced an invalid embedding")
            embedding = np.ascontiguousarray(embedding / np.float32(norm))
            embedding.setflags(write=False)
            self._identity_cache[key] = embedding
            while len(self._identity_cache) > self._identity_cache_entries:
                self._identity_cache.popitem(last=False)
            return embedding

    def _prepare_target(self, image: np.ndarray, target_face: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise SimSwap512InferenceError("target image must be a uint8 array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise SimSwap512InferenceError("target image must be HxWx3 BGR")
        points = _face_keypoints(target_face, "target")
        destination = _TARGET_TEMPLATE * np.asarray(_INPUT_SIZE, dtype=np.float32)
        affine = _estimate_affine(points, destination, "target")
        aligned = cv2.warpAffine(
            image,
            affine,
            _INPUT_SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rgb = aligned[:, :, ::-1].astype(np.float32) * np.float32(1.0 / 255.0)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...]), affine

    def infer(
        self,
        image: np.ndarray,
        target_face: Any,
        source_face: Any,
        *,
        paste_back: bool = False,
    ) -> SwapResult:
        if paste_back:
            raise ValueError("SimSwap512Swapper returns aligned output for the common paste path")
        self._ensure_open()
        target, affine = self._prepare_target(image, target_face)
        source = self.prepare_source(source_face)
        try:
            with self._inference_lock:
                candidate = self._swapper.run(
                    [self._swapper_output],
                    {"input": target, "onnx::Gemm_1": source},
                )[0]
        except Exception as error:
            raise SimSwap512InferenceError(f"SimSwap-512 inference failed: {error}") from error
        candidate = np.asarray(candidate, dtype=np.float32)
        if candidate.shape != (1, 3, 512, 512) or not np.isfinite(candidate).all():
            raise SimSwap512InferenceError(
                f"SimSwap-512 returned an invalid candidate: {candidate.shape}"
            )
        candidate_rgb = candidate[0].transpose(1, 2, 0)
        face_bgr = np.ascontiguousarray(
            np.clip(candidate_rgb[:, :, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
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
            raise SimSwap512InferenceError(str(error)) from error

    def get(
        self,
        image: np.ndarray,
        target_face: Any,
        source_face: Any,
        paste_back: bool = False,
    ) -> SwapResult:
        return self.infer(image, target_face, source_face, paste_back=paste_back)


__all__ = [
    "SIMSWAP512_SHA256",
    "SIMSWAP512_SIZE",
    "SIMSWAP_ARCFACE_SHA256",
    "SIMSWAP_ARCFACE_SIZE",
    "SimSwap512Error",
    "SimSwap512InferenceError",
    "SimSwap512LoadError",
    "SimSwap512Swapper",
    "arcface_path",
    "model_path",
    "simswap512_available",
]
