#!/usr/bin/env python3
"""One durable user intent shared by every processor and output.

The manager never treats a remote processor's last response as the source of
truth.  A click updates this document first, then each available processor is
reconciled to it.  An offline node therefore comes back to the same settings
instead of silently restoring an older per-machine configuration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping


STATE_VERSION = 1

PROCESSOR_WINDOWS = "windows"
PROCESSOR_ARCH = "arch"

INPUT_ARCH_WEBCAM = "arch-webcam"
INPUT_ANDROID_FRONT = "android-front"
INPUT_ANDROID_BACK = "android-back"
INPUT_PRERECORDED = "prerecorded"
INPUT_ASSEMBLER = "assembler"
DEFAULT_ASSEMBLER_LIB = "/var/lib/deep-live-cam/puppet_lib"

OUTPUT_ARCH_CAMERA = "arch-camera"
OUTPUT_ANDROID_PHONE = "android-phone"


@dataclass(frozen=True)
class ProcessorSpec:
    key: str
    label: str
    model: str
    backend: str
    detail: str


PROCESSOR_SPECS: dict[str, ProcessorSpec] = {
    PROCESSOR_WINDOWS: ProcessorSpec(
        PROCESSOR_WINDOWS,
        "Windows 11 remote",
        "InStyleSwapper-256 B",
        "CUDA",
        "RTX-class remote processor · native 256×256 face generation",
    ),
    PROCESSOR_ARCH: ProcessorSpec(
        PROCESSOR_ARCH,
        "This Arch workstation",
        "Native-256 semantic",
        "NCNN/Vulkan",
        "Local RX 570 processor · development checkpoint (not yet qualified)",
    ),
}


@dataclass(frozen=True)
class InputSpec:
    key: str
    label: str
    device_id: str
    stack: str
    lens_facing: str | None
    detail: str


INPUT_SPECS: dict[str, InputSpec] = {
    INPUT_ARCH_WEBCAM: InputSpec(
        INPUT_ARCH_WEBCAM,
        "Arch USB webcam",
        "arch-webcam",
        "arch-v4l2",
        None,
        "MJPEG 1280×720 at 30 FPS · measured natural-indoor controls",
    ),
    INPUT_ANDROID_FRONT: InputSpec(
        INPUT_ANDROID_FRONT,
        "Phone front camera",
        "android-phone",
        "android-camera2",
        "front",
        "Front Camera2 native portrait stream (720x1280 @ 30, hardware H.264) "
        "carried unmodified end-to-end; fitted into the locked system-camera "
        "output at delivery.",
    ),
    INPUT_ANDROID_BACK: InputSpec(
        INPUT_ANDROID_BACK,
        "Phone back camera",
        "android-phone",
        "android-camera2",
        "back",
        "Back Camera2 native portrait stream (720x1280 @ 30, hardware H.264) "
        "carried unmodified end-to-end; fitted into the locked system-camera "
        "output at delivery.",
    ),
    INPUT_PRERECORDED: InputSpec(
        INPUT_PRERECORDED,
        "Prerecorded video",
        "local-prerecorded",
        "prerecorded-relay",
        None,
        "Loop-replay a recorded or rendered MP4 through the receiver as a camera source.",
    ),
    INPUT_ASSEMBLER: InputSpec(
        INPUT_ASSEMBLER,
        "Assembler",
        "local-prerecorded",
        "prerecorded-relay",
        None,
        "Compose a prompt sequence from the pre-rendered puppet library and replay it as a camera source.",
    ),
}


# The intersection intentionally excludes model/backend (fixed by the chosen
# processor) and mirror/rotation (delivery properties owned by Output).
DEFAULT_PROCESSING_SETTINGS: dict[str, Any] = {
    "processing_mode": "face_swap",
    "opacity": 1.0,
    "sharpness": 0.2,
    "mouth_mask_size": 8.0,
    "color_match_strength": 0.35,
    "interpolation_weight": 0.75,
    "many_faces": False,
    "enable_interpolation": False,
    "tracking_enabled": True,
    "detection_interval": 1,
    "tracking_smoothing": 0.65,
    "tracking_grace_frames": 5,
    "minimum_detection_score": 0.45,
    "minimum_face_size": 64,
    "enhancer": "none",
    "show_fps": False,
    "quality_mode": "balanced",
    "quality_auto_correct": True,
    # The measured balanced repair profile used by the qualified Windows
    # baseline.  These fields are supported by both processor stacks and must
    # live in the shared document or the same-looking form can produce visibly
    # different seams/detail on each host.
    "repair_hf_strength": 0.3,
    "repair_checkerboard": 0.4,
    "repair_wavelet": 0.5,
    "repair_boundary_mask": True,
    "repair_boundary_strength": 0.35,
    "repair_camera_detail": 0.0,
}

# Orientation belongs exclusively to the Output stage. These compatibility
# fields are sent to both processors but deliberately have no second UI copy.
PROCESSOR_FIXED_SETTINGS: dict[str, Any] = {
    "live_mirror": False,
    "processing_off_output": "passthrough",
}


FLOAT_LIMITS = {
    "opacity": (0.0, 1.0),
    "sharpness": (0.0, 5.0),
    "mouth_mask_size": (0.0, 100.0),
    "color_match_strength": (0.0, 1.0),
    "interpolation_weight": (0.0, 1.0),
    "tracking_smoothing": (0.0, 0.95),
    "minimum_detection_score": (0.1, 0.95),
    "repair_hf_strength": (0.0, 0.5),
    "repair_checkerboard": (0.0, 1.0),
    "repair_wavelet": (0.0, 1.0),
    "repair_boundary_strength": (0.0, 1.0),
    "repair_camera_detail": (0.0, 4.0),
}
INTEGER_LIMITS = {
    "detection_interval": (1, 5),
    "tracking_grace_frames": (0, 15),
    "minimum_face_size": (32, 512),
}
BOOLEAN_FIELDS = {
    "many_faces",
    "enable_interpolation",
    "tracking_enabled",
    "show_fps",
    "quality_auto_correct",
    "repair_boundary_mask",
}
CHOICES = {
    "processing_mode": {"face_swap", "passthrough"},
    "enhancer": {"none", "gfpgan"},
    "quality_mode": {"monitor", "balanced", "strict"},
}


def default_state_path() -> Path:
    base = Path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    ).expanduser()
    return base / "deep-live-cam" / "manager-state.json"


def normalize_processing_field(field: str, value: Any) -> Any:
    """Validate one shared setting and return its canonical representation."""
    if field not in DEFAULT_PROCESSING_SETTINGS:
        raise ValueError(f"unsupported shared processing setting: {field}")
    if field in FLOAT_LIMITS:
        low, high = FLOAT_LIMITS[field]
        number = float(value)
        if not low <= number <= high:
            raise ValueError(f"{field} must be between {low} and {high}")
        return number
    if field in INTEGER_LIMITS:
        low, high = INTEGER_LIMITS[field]
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
        number = int(value)
        if not low <= number <= high:
            raise ValueError(f"{field} must be between {low} and {high}")
        return number
    if field in BOOLEAN_FIELDS:
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be true or false")
        return value
    allowed = CHOICES[field]
    selected = str(value)
    if selected not in allowed:
        raise ValueError(f"{field} must be one of {', '.join(sorted(allowed))}")
    return selected


def normalize_processing(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(DEFAULT_PROCESSING_SETTINGS)
    for field, value in values.items():
        if field in DEFAULT_PROCESSING_SETTINGS:
            normalized[field] = normalize_processing_field(field, value)
    # Existing strict-profile documents predate the final-resolution repair.
    # Migrate only that measured profile; balanced/monitor remain unchanged.
    if (
        "repair_camera_detail" not in values
        and normalized.get("quality_mode") == "strict"
    ):
        normalized["repair_camera_detail"] = 3.5
    if (
        "repair_boundary_strength" not in values
        and normalized.get("quality_mode") == "strict"
    ):
        normalized["repair_boundary_strength"] = 0.5
    return normalized


def default_document() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "revision": 0,
        "updated_at": 0.0,
        "processor": PROCESSOR_WINDOWS,
        "input": INPUT_ANDROID_FRONT,
        "outputs": {
            OUTPUT_ARCH_CAMERA: True,
            OUTPUT_ANDROID_PHONE: True,
        },
        "output_transform": {"mirror": False, "rotation": 0},
        "processing": deepcopy(DEFAULT_PROCESSING_SETTINGS),
        "source_identifier": None,
        "prerecorded_path": None,
        "prerecorded_mode": "loop",
        # Framing of the prerecorded video inside the locked output box.
        # offset_x/offset_y shift the image in pixels (positive = right/down),
        # exposing black where there is no longer any source.  zoom scales the
        # image about its centre (1.0 = fit-to-cover, >1.0 zooms in).
        "prerecorded_adjust": {"offset_x": 0, "offset_y": 0, "zoom": 1.0},
        # Assembler input: the puppet segment library directory and the
        # composed prompt sequence (token strings, e.g. "turn_left",
        # "say 4-7-2", "neutral 1s").
        "assembler_lib": DEFAULT_ASSEMBLER_LIB,
        "assembler_tokens": [],
    }


def normalize_document(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = default_document()
    document = dict(value or {})
    processor = str(document.get("processor", ""))
    if processor in PROCESSOR_SPECS:
        result["processor"] = processor
    selected_input = str(document.get("input", ""))
    if selected_input in INPUT_SPECS:
        result["input"] = selected_input
    outputs = document.get("outputs")
    if isinstance(outputs, Mapping):
        for key in (OUTPUT_ARCH_CAMERA, OUTPUT_ANDROID_PHONE):
            if isinstance(outputs.get(key), bool):
                result["outputs"][key] = outputs[key]
    transform = document.get("output_transform")
    if isinstance(transform, Mapping):
        if isinstance(transform.get("mirror"), bool):
            result["output_transform"]["mirror"] = transform["mirror"]
        try:
            rotation = int(transform.get("rotation", 0))
        except (TypeError, ValueError):
            rotation = 0
        if rotation in (0, 90, 180, 270):
            result["output_transform"]["rotation"] = rotation
    processing = document.get("processing")
    if isinstance(processing, Mapping):
        result["processing"] = normalize_processing(processing)
    identifier = document.get("source_identifier")
    result["source_identifier"] = str(identifier) if identifier else None
    prerecorded_path = document.get("prerecorded_path")
    result["prerecorded_path"] = str(prerecorded_path) if prerecorded_path else None
    mode = str(document.get("prerecorded_mode", "loop"))
    result["prerecorded_mode"] = mode if mode in ("loop", "once", "freeze") else "loop"
    adjust = document.get("prerecorded_adjust")
    if isinstance(adjust, Mapping):
        for key in ("offset_x", "offset_y"):
            try:
                result["prerecorded_adjust"][key] = int(adjust.get(key, 0))
            except (TypeError, ValueError):
                result["prerecorded_adjust"][key] = 0
        try:
            zoom = float(adjust.get("zoom", 1.0))
        except (TypeError, ValueError):
            zoom = 1.0
        result["prerecorded_adjust"]["zoom"] = min(4.0, max(0.25, zoom))
    if "assembler_lib" in document:
        lib = document.get("assembler_lib")
        # An empty selection always falls back to the default library; there
        # is no meaningful "no library" state.
        result["assembler_lib"] = str(lib) if lib else DEFAULT_ASSEMBLER_LIB
    if "assembler_tokens" in document:
        raw_tokens = document.get("assembler_tokens")
        tokens: list[str] = []
        if isinstance(raw_tokens, list):
            for token in raw_tokens:
                text = str(token).strip()
                if text:
                    tokens.append(text)
        result["assembler_tokens"] = tokens[:32]
    try:
        result["revision"] = max(0, int(document.get("revision", 0)))
    except (TypeError, ValueError):
        pass
    try:
        result["updated_at"] = max(0.0, float(document.get("updated_at", 0.0)))
    except (TypeError, ValueError):
        pass
    return result


def processor_payload(
    document: Mapping[str, Any], target: str | None = None
) -> dict[str, Any]:
    """Return one host's payload while keeping the inactive host in bypass.

    Quality controls remain shared and synchronized. ``processing_mode`` is
    the one host-owned effective field: the non-selected endpoint must receive
    passthrough so two machines can never claim to be processing at once.
    Omitting ``target`` means the selected host and is retained for callers
    that only need the active processor contract.
    """
    normalized = normalize_document(document)
    selected_target = normalized["processor"] if target is None else str(target)
    if selected_target not in PROCESSOR_SPECS:
        raise ValueError(f"unknown processor: {selected_target}")
    processing = dict(normalized["processing"])
    if selected_target != normalized["processor"]:
        processing["processing_mode"] = "passthrough"
    return {**processing, **PROCESSOR_FIXED_SETTINGS}


def local_processor_payload(
    document: Mapping[str, Any], health: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a safe payload for the installed Arch worker generation.

    A rolling deployment can leave an already-running *inactive* standby on
    the previous control schema until its privileged system service is next
    restarted.  Its advertised processing fields are authoritative in that
    narrow case: omit newer quality fields so the OFF request remains atomic.
    Once Arch is selected, return the complete contract and require an
    upgraded worker rather than silently activating with partial settings.
    """
    normalized = normalize_document(document)
    payload = processor_payload(normalized, PROCESSOR_ARCH)
    if normalized["processor"] == PROCESSOR_ARCH:
        return payload
    control = health.get("control")
    capabilities = (
        control.get("capabilities") if isinstance(control, Mapping) else None
    )
    advertised = (
        capabilities.get("processing_fields")
        if isinstance(capabilities, Mapping)
        else None
    )
    if not isinstance(advertised, (list, tuple, set, frozenset)):
        return payload
    supported = {str(field) for field in advertised}
    if not supported:
        return payload
    return {key: value for key, value in payload.items() if key in supported}


def local_processor_matches(
    document: Mapping[str, Any],
    health: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> bool:
    """Return whether the restarted Arch worker reflects manager intent.

    The worker's ``in_sync`` bit refers to its own last control document, so it
    is not sufficient after a process restart or an out-of-band update.  Check
    the effective input, activation owner, fixed model, every shared setting,
    and (when available) the exact content-addressed source path.
    """
    normalized = normalize_document(document)
    if health.get("state") != "running":
        return False
    control = health.get("control")
    if not isinstance(control, Mapping):
        return False
    control_in_sync = control.get("in_sync") is True
    arch_active = normalized["processor"] == PROCESSOR_ARCH
    # An inactive endpoint may honestly clamp an unavailable, non-essential
    # enhancer to ``none``.  Treat that one advertised capability difference
    # as safe standby convergence; all transport, ownership, model and other
    # processing fields are still compared below.  An active Arch processor
    # must remain strictly synchronized.
    capabilities = control.get("capabilities")
    supported_enhancers = (
        set(capabilities.get("enhancers", ()))
        if isinstance(capabilities, Mapping)
        else set()
    )
    advertised_processing = (
        capabilities.get("processing_fields")
        if isinstance(capabilities, Mapping)
        else None
    )
    supported_processing = (
        {str(field) for field in advertised_processing}
        if isinstance(advertised_processing, (list, tuple, set, frozenset))
        else set()
    )
    allowed_inactive_enhancer_clamp = False
    allowed_inactive_capability_omission = False
    effective = control.get("effective")
    if not isinstance(effective, Mapping):
        return False
    if effective.get("active") is not arch_active:
        return False
    if effective.get("input") != normalized["input"]:
        return False
    model = effective.get("model")
    if not isinstance(model, Mapping):
        return False
    if (
        model.get("swapper_model") != "native-256"
        or model.get("swapper_backend") != "ncnn"
        or model.get("configured") is not True
    ):
        return False
    model_must_be_ready = bool(
        normalized["processor"] == PROCESSOR_ARCH
        and normalized["processing"]["processing_mode"] == "face_swap"
    )
    if model_must_be_ready and model.get("ready") is not True:
        return False
    processing = effective.get("processing")
    if not isinstance(processing, Mapping):
        return False
    for key, wanted in processor_payload(normalized, PROCESSOR_ARCH).items():
        if not arch_active and supported_processing and key not in supported_processing:
            allowed_inactive_capability_omission = True
            continue
        current = processing.get(key)
        if (
            key == "enhancer"
            and not arch_active
            and current == "none"
            and wanted not in supported_enhancers
            and "none" in supported_enhancers
        ):
            allowed_inactive_enhancer_clamp = True
            continue
        if isinstance(wanted, float):
            try:
                if abs(float(current) - wanted) >= 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif current != wanted:
            return False
    if source_path is not None and processing.get("source_path") != str(source_path):
        return False
    return bool(
        control_in_sync
        or allowed_inactive_enhancer_clamp
        or allowed_inactive_capability_omission
    )


def reconciliation_payload(
    document: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    target: str = PROCESSOR_WINDOWS,
) -> dict[str, Any]:
    """Only send settings that differ from an endpoint's reported state.

    Older Windows deployments omit newer fields.  Shared fields are still
    sent because the legacy endpoint accepts and persists this established
    contract; response-only model fields are deliberately outside it.
    """
    desired = processor_payload(document, target)
    differences: dict[str, Any] = {}
    for key, wanted in desired.items():
        current = actual.get(key)
        if isinstance(wanted, float):
            try:
                equal = abs(float(current) - wanted) < 1e-6
            except (TypeError, ValueError):
                equal = False
        else:
            equal = current == wanted
        if not equal:
            differences[key] = wanted
    return differences


class DesiredStateStore:
    """Atomic, small JSON store used by both the GUI and local processor."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or default_state_path()).expanduser()
        self.loaded_from_disk = False
        self.document = self._read()

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("state is not a JSON object")
            normalized = normalize_document(value)
            self.loaded_from_disk = True
            return normalized
        except (OSError, json.JSONDecodeError, ValueError):
            return default_document()

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.document)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def update(self, **changes: Any) -> dict[str, Any]:
        candidate = self.snapshot()
        candidate.update(changes)
        normalized = normalize_document(candidate)
        normalized["revision"] = int(self.document.get("revision", 0)) + 1
        normalized["updated_at"] = time.time()
        self.document = normalized
        self._save()
        return self.snapshot()

    def set_processing(self, field: str, value: Any) -> dict[str, Any]:
        processing = dict(self.document["processing"])
        processing[field] = normalize_processing_field(field, value)
        return self.update(processing=processing)

    def set_processing_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        processing = dict(self.document["processing"])
        for field, value in values.items():
            processing[field] = normalize_processing_field(field, value)
        return self.update(processing=processing)

    def set_processor(self, processor: str) -> dict[str, Any]:
        if processor not in PROCESSOR_SPECS:
            raise ValueError(f"unknown processor: {processor}")
        return self.update(processor=processor)

    def set_processor_processing(
        self, processor: str, enabled: bool
    ) -> dict[str, Any]:
        """Atomically select/enable one host or disable the selected host.

        An unchecked non-selected host is already off and cannot disable the
        processor that currently owns the route. This guard also makes a stale
        queued UI event harmless.
        """
        if processor not in PROCESSOR_SPECS:
            raise ValueError(f"unknown processor: {processor}")
        if not isinstance(enabled, bool):
            raise ValueError("processor processing state must be true or false")
        if not enabled and self.document["processor"] != processor:
            return self.snapshot()
        processing = dict(self.document["processing"])
        processing["processing_mode"] = "face_swap" if enabled else "passthrough"
        changes: dict[str, Any] = {"processing": processing}
        if enabled:
            changes["processor"] = processor
        return self.update(**changes)

    def set_input(self, input_key: str) -> dict[str, Any]:
        if input_key not in INPUT_SPECS:
            raise ValueError(f"unknown input: {input_key}")
        return self.update(input=input_key)

    def set_prerecorded_path(self, path: str | None) -> dict[str, Any]:
        return self.update(prerecorded_path=str(path) if path else None)

    def set_prerecorded_mode(self, mode: str) -> dict[str, Any]:
        if mode not in ("loop", "once", "freeze"):
            raise ValueError("prerecorded_mode must be loop, once, or freeze")
        return self.update(prerecorded_mode=mode)

    def set_assembler_lib(self, path: str | None) -> dict[str, Any]:
        return self.update(assembler_lib=str(path) if path else None)

    def set_assembler_tokens(self, tokens: list[str]) -> dict[str, Any]:
        clean = [str(t).strip() for t in tokens if str(t).strip()]
        return self.update(assembler_tokens=clean[:32])

    def set_prerecorded_adjust(
        self,
        *,
        offset_x: int | None = None,
        offset_y: int | None = None,
        zoom: float | None = None,
    ) -> dict[str, Any]:
        """Update the prerecorded framing (offset in px, zoom about centre)."""
        adjust = dict(self.document["prerecorded_adjust"])
        if offset_x is not None:
            adjust["offset_x"] = int(offset_x)
        if offset_y is not None:
            adjust["offset_y"] = int(offset_y)
        if zoom is not None:
            adjust["zoom"] = min(4.0, max(0.25, float(zoom)))
        return self.update(prerecorded_adjust=adjust)

    def set_output(self, output: str, enabled: bool) -> dict[str, Any]:
        if output not in (OUTPUT_ARCH_CAMERA, OUTPUT_ANDROID_PHONE):
            raise ValueError(f"unknown output: {output}")
        if not isinstance(enabled, bool):
            raise ValueError("output state must be true or false")
        outputs = dict(self.document["outputs"])
        outputs[output] = enabled
        return self.update(outputs=outputs)

    def set_transform(self, *, mirror: bool, rotation: int) -> dict[str, Any]:
        if not isinstance(mirror, bool):
            raise ValueError("mirror must be true or false")
        if int(rotation) not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0, 90, 180, or 270")
        return self.update(
            output_transform={"mirror": mirror, "rotation": int(rotation)}
        )
