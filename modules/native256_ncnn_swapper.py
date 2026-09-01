"""ncnn/Vulkan runtime for a hash-pinned native-256 model bundle.

The bridge executes the split identity conditioner and per-frame generator as
one serialized operation.  It caches the conditioned style internally until
the exact mapped source identity changes, and returns the model's uncomposited
RGB candidate plus semantic alpha.  This module performs no downloads and has
no model fallback behavior.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import threading
from typing import Any
import weakref

import numpy as np
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
_DEFAULT_LIBRARY = _ROOT / "arch-linux" / "ncnn" / "libdeep_live_cam_ncnn.so"
_RANGE_TOLERANCE = 1e-4
_ABI_VERSION = 2


class Native256NcnnError(RuntimeError):
    """Base native-256 ncnn runtime error."""


class Native256NcnnLoadError(Native256NcnnError):
    """The local ncnn bundle or bridge could not be opened safely."""


class Native256NcnnInferenceError(Native256NcnnError):
    """The ncnn bundle produced no safe result for the current frame."""


def default_manifest_path() -> Path:
    return Path(os.environ.get("DLC_NATIVE256_MANIFEST", _DEFAULT_MANIFEST))


def default_library_path() -> Path:
    return Path(os.environ.get("DLC_NCNN_LIBRARY", _DEFAULT_LIBRARY))


def _load_manifest(
    manifest_path: str | Path | None, *, require_qualified: bool
) -> Native256Manifest:
    try:
        manifest = Native256Manifest.load(manifest_path or default_manifest_path())
    except SwapperContractError as error:
        raise Native256NcnnLoadError(str(error)) from error
    if require_qualified and not (
        manifest.quality_status == "qualified" and manifest.auto_select_eligible
    ):
        raise Native256NcnnLoadError(
            "native-256 model is not eligible for automatic selection "
            f"(quality_status={manifest.quality_status}, "
            f"auto_select_eligible={manifest.auto_select_eligible})"
        )
    if manifest.ncnn is None:
        raise Native256NcnnLoadError(
            "native-256 manifest does not contain a hash-pinned ncnn bundle"
        )
    return manifest


def native256_ncnn_available(
    manifest_path: str | Path | None = None,
    *,
    library_path: str | Path | None = None,
    require_qualified: bool = False,
) -> bool:
    """Return whether the complete, local ncnn representation is usable."""
    library = Path(library_path or default_library_path())
    if not library.is_file():
        return False
    try:
        manifest = _load_manifest(
            manifest_path, require_qualified=require_qualified
        )
        verify_embedded_onnx(
            manifest.conditioner.asset, "identity conditioner"
        )
        verify_embedded_onnx(manifest.swapper.asset, "native-256 swapper")
    except (Native256NcnnLoadError, SwapperContractError):
        return False
    return True


class Native256NcnnSwapper:
    """Application adapter for the native-256 ncnn C ABI."""

    backend = "ncnn"

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        library_path: str | Path | None = None,
        device_index: int = 0,
        require_qualified: bool = False,
        library_loader: Any = ctypes.CDLL,
    ) -> None:
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise ValueError("device_index must be an integer")
        if device_index < 0:
            raise ValueError("device_index must not be negative")
        self.manifest = _load_manifest(
            manifest_path, require_qualified=require_qualified
        )
        if library_loader is ctypes.CDLL:
            try:
                verify_embedded_onnx(
                    self.manifest.conditioner.asset, "identity conditioner"
                )
                verify_embedded_onnx(self.manifest.swapper.asset, "native-256 swapper")
            except SwapperContractError as error:
                raise Native256NcnnLoadError(str(error)) from error
        self.spec = self.manifest.spec
        self.quality_status = self.manifest.quality_status
        self.input_size = self.spec.input_size
        self._run_lock = threading.RLock()
        self._closed = False
        self._source_key: bytes | None = None

        try:
            identity_map = np.load(self.manifest.identity_map.path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise Native256NcnnLoadError(
                f"could not load identity map: {error}"
            ) from error
        if identity_map.dtype != np.float32 or identity_map.shape != (512, 512):
            raise Native256NcnnLoadError(
                "identity map must be float32 with shape (512, 512), got "
                f"{identity_map.dtype} {identity_map.shape}"
            )
        if not np.isfinite(identity_map).all():
            raise Native256NcnnLoadError("identity map contains a non-finite value")
        self._identity_map = np.ascontiguousarray(identity_map)
        self._identity_map.setflags(write=False)

        library = Path(library_path or default_library_path())
        if not library.is_file():
            raise Native256NcnnLoadError(f"ncnn bridge does not exist: {library}")
        try:
            self._library = library_loader(str(library))
            self._configure_abi(self._library)
            if self._library.dlc_ncnn_abi_version() != _ABI_VERSION:
                raise Native256NcnnLoadError(
                    "ncnn bridge ABI is incompatible; rebuild arch-linux/ncnn"
                )
            ncnn = self.manifest.ncnn
            assert ncnn is not None
            handle = self._library.dlc_ncnn_native256_create(
                os.fsencode(ncnn.conditioner.param.path),
                os.fsencode(ncnn.conditioner.model.path),
                os.fsencode(ncnn.swapper.param.path),
                os.fsencode(ncnn.swapper.model.path),
                device_index,
                int(ncnn.fp16_storage),
            )
        except Native256NcnnLoadError:
            raise
        except Exception as error:
            raise Native256NcnnLoadError(
                f"could not load native-256 ncnn bridge: {error}"
            ) from error
        if not handle:
            raise Native256NcnnLoadError(self._native_error())
        self._handle = ctypes.c_void_p(handle)
        self._finalizer = weakref.finalize(
            self, self._destroy, self._library, self._handle
        )
        name = self._library.dlc_ncnn_native256_device_name(self._handle)
        self.device_name = (
            name.decode("utf-8", errors="replace") if name else "Vulkan"
        )

    @staticmethod
    def _configure_abi(library: Any) -> None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        library.dlc_ncnn_abi_version.argtypes = []
        library.dlc_ncnn_abi_version.restype = ctypes.c_int
        library.dlc_ncnn_last_error.argtypes = []
        library.dlc_ncnn_last_error.restype = ctypes.c_char_p
        library.dlc_ncnn_native256_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.dlc_ncnn_native256_create.restype = ctypes.c_void_p
        library.dlc_ncnn_native256_destroy.argtypes = [ctypes.c_void_p]
        library.dlc_ncnn_native256_destroy.restype = None
        library.dlc_ncnn_native256_device_name.argtypes = [ctypes.c_void_p]
        library.dlc_ncnn_native256_device_name.restype = ctypes.c_char_p
        library.dlc_ncnn_native256_clear_style.argtypes = [ctypes.c_void_p]
        library.dlc_ncnn_native256_clear_style.restype = None
        library.dlc_ncnn_native256_run.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
        ]
        library.dlc_ncnn_native256_run.restype = ctypes.c_int

    @staticmethod
    def _destroy(library: Any, handle: ctypes.c_void_p) -> None:
        library.dlc_ncnn_native256_destroy(handle)

    def _native_error(self) -> str:
        message = self._library.dlc_ncnn_last_error()
        if not message:
            return "unknown native-256 ncnn error"
        return message.decode("utf-8", errors="replace")

    def _ensure_open(self) -> None:
        if self._closed:
            raise Native256NcnnInferenceError("native-256 ncnn swapper is closed")

    def close(self) -> None:
        with self._run_lock:
            if self._closed:
                return
            self._closed = True
            self._source_key = None
            if self._finalizer.alive:
                self._finalizer()

    def clear_style_cache(self) -> None:
        with self._run_lock:
            self._ensure_open()
            self._library.dlc_ncnn_native256_clear_style(self._handle)
            self._source_key = None

    @property
    def style_cache_size(self) -> int:
        with self._run_lock:
            return int(self._source_key is not None)

    def prepare_source(self, source_face: Any) -> np.ndarray:
        """Validate and project the source; conditioning is lazy in ``infer``."""
        self._ensure_open()
        if source_face is None or getattr(source_face, "normed_embedding", None) is None:
            raise Native256NcnnInferenceError(
                "source face does not contain a normalized identity embedding"
            )
        try:
            return mapped_inswapper_identity(
                source_face.normed_embedding, self._identity_map
            )
        except SwapperContractError as error:
            raise Native256NcnnInferenceError(str(error)) from error

    def _prepare_target(
        self, image: np.ndarray, target_face: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(image, np.ndarray):
            raise Native256NcnnInferenceError("target image must be a numpy array")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise Native256NcnnInferenceError(
                f"target image must be uint8 HxWx3 BGR, got {image.dtype} {image.shape}"
            )
        if target_face is None or getattr(target_face, "kps", None) is None:
            raise Native256NcnnInferenceError(
                "target face does not contain five keypoints"
            )
        try:
            keypoints = np.asarray(target_face.kps, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise Native256NcnnInferenceError(
                f"invalid target keypoints: {error}"
            ) from error
        if keypoints.shape != (5, 2) or not np.isfinite(keypoints).all():
            raise Native256NcnnInferenceError(
                "target keypoints must be finite with shape (5, 2), got "
                f"{keypoints.shape}"
            )
        try:
            aligned, affine = face_align.norm_crop2(
                image, keypoints, self.input_size[0]
            )
        except Exception as error:
            raise Native256NcnnInferenceError(
                f"256px face alignment failed: {error}"
            ) from error
        if (
            not isinstance(aligned, np.ndarray)
            or aligned.dtype != np.uint8
            or aligned.shape != (self.input_size[1], self.input_size[0], 3)
        ):
            raise Native256NcnnInferenceError(
                "alignment returned an invalid crop: "
                f"{getattr(aligned, 'dtype', None)} {getattr(aligned, 'shape', None)}"
            )
        affine = np.ascontiguousarray(affine, dtype=np.float32)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise Native256NcnnInferenceError(
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
        if paste_back:
            raise ValueError(
                "Native256NcnnSwapper returns aligned output for the application paste path"
            )
        target, affine = self._prepare_target(image, target_face)
        mapped_identity = self.prepare_source(source_face)
        candidate = np.empty((1, 3, 256, 256), dtype=np.float32)
        alpha = np.empty((1, 1, 256, 256), dtype=np.float32)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        with self._run_lock:
            self._ensure_open()
            status = self._library.dlc_ncnn_native256_run(
                self._handle,
                target.ctypes.data_as(float_pointer),
                mapped_identity.ctypes.data_as(float_pointer),
                candidate.ctypes.data_as(float_pointer),
                alpha.ctypes.data_as(float_pointer),
            )
            if status != 0:
                raise Native256NcnnInferenceError(self._native_error())
            self._source_key = hashlib.sha256(
                mapped_identity.tobytes(order="C")
            ).digest()

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
            np.rint(alpha[0, 0] * 255.0).astype(np.uint8)
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
            raise Native256NcnnInferenceError(str(error)) from error

    def get(
        self,
        image: np.ndarray,
        target_face: Any,
        source_face: Any,
        paste_back: bool = False,
    ) -> SwapResult:
        return self.infer(
            image, target_face, source_face, paste_back=paste_back
        )


def _normalize_range(
    value: np.ndarray, value_range: tuple[float, float], label: str
) -> np.ndarray:
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise Native256NcnnInferenceError(
            f"{label} must contain finite float32 values"
        )
    low, high = value_range
    minimum = float(value.min())
    maximum = float(value.max())
    if minimum < low - _RANGE_TOLERANCE or maximum > high + _RANGE_TOLERANCE:
        raise Native256NcnnInferenceError(
            f"{label} is outside its declared range {value_range}: "
            f"[{minimum}, {maximum}]"
        )
    normalized = (np.clip(value, low, high) - low) / (high - low)
    return np.ascontiguousarray(normalized, dtype=np.float32)


__all__ = [
    "Native256NcnnError",
    "Native256NcnnInferenceError",
    "Native256NcnnLoadError",
    "Native256NcnnSwapper",
    "default_library_path",
    "default_manifest_path",
    "native256_ncnn_available",
]
