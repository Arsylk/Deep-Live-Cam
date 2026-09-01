"""ncnn/Vulkan adapter for INSwapper.

The native bridge only replaces the expensive ONNX inference call. Alignment,
identity projection, color conversion, tracking, masking, and paste-back stay
in the existing Python pipeline.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import weakref

import cv2
import numpy as np
from insightface.utils import face_align


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_DIR = _ROOT / "models" / "ncnn"
_DEFAULT_LIBRARY = _ROOT / "arch-linux" / "ncnn" / "libdeep_live_cam_ncnn.so"


def _asset_paths() -> tuple[Path, Path, Path, Path]:
    model_dir = Path(os.environ.get("DLC_NCNN_MODEL_DIR", _DEFAULT_MODEL_DIR))
    library = Path(os.environ.get("DLC_NCNN_LIBRARY", _DEFAULT_LIBRARY))
    return (
        library,
        model_dir / "inswapper_128.ncnn.param",
        model_dir / "inswapper_128.ncnn.bin",
        model_dir / "inswapper_128_emap.npy",
    )


def ncnn_swapper_available() -> bool:
    """Return whether all local, offline runtime assets are present."""
    return all(path.is_file() for path in _asset_paths())


class NcnnSwapper:
    """InsightFace-compatible INSwapper backed by ncnn Vulkan."""

    input_size = (128, 128)
    input_mean = 0.0
    input_std = 255.0

    def __init__(self, device_index: int = 0) -> None:
        library_path, param_path, model_path, emap_path = _asset_paths()
        missing = [
            str(path)
            for path in (library_path, param_path, model_path, emap_path)
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError("missing ncnn swapper assets: " + ", ".join(missing))

        self._library = ctypes.CDLL(str(library_path))
        self._configure_abi(self._library)
        handle = self._library.dlc_ncnn_swapper_create(
            os.fsencode(param_path), os.fsencode(model_path), int(device_index)
        )
        if not handle:
            raise RuntimeError(self._native_error())
        self._handle = ctypes.c_void_p(handle)
        self._finalizer = weakref.finalize(
            self, self._destroy, self._library, self._handle
        )

        self.emap = np.load(emap_path, allow_pickle=False)
        if self.emap.shape != (512, 512) or self.emap.dtype != np.float32:
            self.close()
            raise RuntimeError(
                f"invalid ncnn identity map: {self.emap.shape} {self.emap.dtype}"
            )
        self.emap = np.ascontiguousarray(self.emap)
        name = self._library.dlc_ncnn_swapper_device_name(self._handle)
        self.device_name = name.decode("utf-8", errors="replace") if name else "Vulkan"

    @staticmethod
    def _configure_abi(library: ctypes.CDLL) -> None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        library.dlc_ncnn_last_error.argtypes = []
        library.dlc_ncnn_last_error.restype = ctypes.c_char_p
        library.dlc_ncnn_swapper_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        library.dlc_ncnn_swapper_create.restype = ctypes.c_void_p
        library.dlc_ncnn_swapper_destroy.argtypes = [ctypes.c_void_p]
        library.dlc_ncnn_swapper_destroy.restype = None
        library.dlc_ncnn_swapper_device_name.argtypes = [ctypes.c_void_p]
        library.dlc_ncnn_swapper_device_name.restype = ctypes.c_char_p
        library.dlc_ncnn_swapper_run.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            float_pointer,
            float_pointer,
        ]
        library.dlc_ncnn_swapper_run.restype = ctypes.c_int

    @staticmethod
    def _destroy(library: ctypes.CDLL, handle: ctypes.c_void_p) -> None:
        library.dlc_ncnn_swapper_destroy(handle)

    def _native_error(self) -> str:
        message = self._library.dlc_ncnn_last_error()
        if not message:
            return "unknown ncnn Vulkan error"
        return message.decode("utf-8", errors="replace")

    def close(self) -> None:
        if self._finalizer.alive:
            self._finalizer()

    def forward(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and not finalizer.alive:
            raise RuntimeError("ncnn swapper is closed")
        target = np.ascontiguousarray(target, dtype=np.float32)
        source = np.ascontiguousarray(source, dtype=np.float32)
        if target.shape != (1, 3, 128, 128):
            raise ValueError(f"target must have shape (1,3,128,128), got {target.shape}")
        if source.shape != (1, 512):
            raise ValueError(f"source must have shape (1,512), got {source.shape}")

        output = np.empty((1, 3, 128, 128), dtype=np.float32)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        status = self._library.dlc_ncnn_swapper_run(
            self._handle,
            target.ctypes.data_as(float_pointer),
            source.ctypes.data_as(float_pointer),
            output.ctypes.data_as(float_pointer),
        )
        if status != 0:
            raise RuntimeError(self._native_error())
        return output

    def get(
        self,
        image: np.ndarray,
        target_face: object,
        source_face: object,
        paste_back: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if paste_back:
            raise ValueError(
                "NcnnSwapper expects the application's optimized paste-back path"
            )
        aligned, matrix = face_align.norm_crop2(
            image, target_face.kps, self.input_size[0]
        )
        target = cv2.dnn.blobFromImage(
            aligned,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        latent = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)
        latent = np.ascontiguousarray(latent @ self.emap, dtype=np.float32)
        norm = float(np.linalg.norm(latent))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("source identity embedding has zero or invalid norm")
        latent /= norm

        prediction = self.forward(target, latent)
        rgb = prediction.transpose((0, 2, 3, 1))[0]
        bgr = np.clip(255.0 * rgb, 0.0, 255.0).astype(np.uint8)[:, :, ::-1]
        return bgr, matrix


__all__ = ["NcnnSwapper", "ncnn_swapper_available"]
