#!/usr/bin/env python3
"""Evaluate visually motivated post-processing variants with DeepfakeBench.

This is a diagnostic A/B harness, not an official DeepfakeBench evaluation.
It expects paired, identically aligned ``real`` and ``fake`` PNG crops.  The
post-processing variants only restore camera-domain properties (tone and
high-frequency texture) from each real target frame; no detector gradients or
detector-specific perturbations are used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ImagePairTransform = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _paired_images(
    corpus: Path, real_subdir: str = "real", fake_subdir: str = "fake"
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    real_paths = sorted((corpus / real_subdir).glob("*.png"))
    fake_paths = sorted((corpus / fake_subdir).glob("*.png"))
    if not real_paths or len(real_paths) != len(fake_paths):
        raise ValueError("corpus must contain equally sized real/ and fake/ PNG sets")
    if [path.name for path in real_paths] != [path.name for path in fake_paths]:
        raise ValueError("real/fake filenames must match exactly")

    real = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in real_paths]
    fake = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in fake_paths]
    if any(image is None for image in real + fake):
        raise ValueError("one or more corpus images could not be decoded")
    return real, fake, [path.name for path in real_paths]


def _change_mask(real: np.ndarray, fake: np.ndarray) -> np.ndarray:
    """Return a soft mask around genuinely changed pixels in an aligned pair."""
    delta = np.mean(np.abs(real.astype(np.float32) - fake.astype(np.float32)), axis=2)
    hard = (delta > 2.0).astype(np.uint8) * 255
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    hard = cv2.dilate(hard, np.ones((7, 7), np.uint8), iterations=1)
    soft = cv2.GaussianBlur(hard.astype(np.float32) / 255.0, (0, 0), 3.0)
    return np.clip(soft, 0.0, 1.0)[..., None]


def _tone_match(real: np.ndarray, fake: np.ndarray, amount: float) -> np.ndarray:
    """Match bounded local LAB statistics inside the changed face region."""
    if amount <= 0.0:
        return fake.copy()
    mask = _change_mask(real, fake)
    core = mask[..., 0] > 0.5
    if int(core.sum()) < 64:
        return fake.copy()

    real_lab = cv2.cvtColor(real, cv2.COLOR_BGR2LAB).astype(np.float32)
    fake_lab = cv2.cvtColor(fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    adjusted = fake_lab.copy()
    for channel in range(3):
        real_values = real_lab[..., channel][core]
        fake_values = fake_lab[..., channel][core]
        real_mean, fake_mean = float(real_values.mean()), float(fake_values.mean())
        real_std = float(real_values.std()) + 1e-6
        fake_std = float(fake_values.std()) + 1e-6
        scale = float(np.clip(real_std / fake_std, 0.90, 1.10))
        offset_limit = 8.0 if channel == 0 else 4.0
        offset = float(np.clip(real_mean - fake_mean, -offset_limit, offset_limit))
        target = (fake_lab[..., channel] - fake_mean) * scale + fake_mean + offset
        adjusted[..., channel] += (target - fake_lab[..., channel]) * float(amount)

    matched = cv2.cvtColor(np.clip(adjusted, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    result = fake.astype(np.float32) * (1.0 - mask) + matched.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _restore_detail(
    real: np.ndarray,
    fake: np.ndarray,
    amount: float,
    transfer: float,
    ceiling: float = 1.7,
) -> np.ndarray:
    """Restore target-camera detail without changing low-frequency identity.

    ``amount`` scales the swap's own high-pass residual toward the target
    camera's energy. ``transfer`` blends in the spatially aligned target-frame
    residual. Both are bounded and restricted to the changed face region.
    """
    if amount <= 0.0 and transfer <= 0.0:
        return fake.copy()
    mask = _change_mask(real, fake)
    core = mask[..., 0] > 0.5
    if int(core.sum()) < 64:
        return fake.copy()

    real_f = real.astype(np.float32)
    fake_f = fake.astype(np.float32)
    real_low = cv2.GaussianBlur(real_f, (0, 0), 1.0)
    fake_low = cv2.GaussianBlur(fake_f, (0, 0), 1.0)
    real_high = real_f - real_low
    fake_high = fake_f - fake_low

    real_energy = float(np.sqrt(np.mean(np.square(real_high[core])))) + 1e-6
    fake_energy = float(np.sqrt(np.mean(np.square(fake_high[core])))) + 1e-6
    raw_scale = real_energy / fake_energy
    target_scale = float(np.clip(raw_scale, 0.85, 1.75))
    effective_scale = 1.0 + (target_scale - 1.0) * float(np.clip(amount, 0.0, 4.0))
    minimum_scale = max(0.5, raw_scale * 0.8)
    maximum_scale = min(2.5, max(minimum_scale, raw_scale * float(ceiling)))
    effective_scale = float(np.clip(effective_scale, minimum_scale, maximum_scale))
    transfer = float(np.clip(transfer, 0.0, 1.0))
    repaired_high = fake_high * effective_scale
    repaired_high = repaired_high * (1.0 - transfer) + real_high * transfer
    repaired = fake_low + repaired_high
    result = fake_f * (1.0 - mask) + repaired * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _masked_tonal_adjustment(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Apply a small, feathered camera-control-like adjustment to the face."""
    mask = _change_mask(real, fake)
    value = fake.astype(np.float32)
    if contrast != 1.0 or brightness != 0.0:
        value = (value - 127.5) * float(contrast) + 127.5 + float(brightness)
    if saturation != 1.0:
        gray = cv2.cvtColor(
            np.clip(value, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)[..., None]
        value = gray + (value - gray) * float(saturation)
    result = fake.astype(np.float32) * (1.0 - mask) + value * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _masked_unsharp(
    real: np.ndarray, fake: np.ndarray, *, strength: float, sigma: float = 1.0
) -> np.ndarray:
    mask = _change_mask(real, fake)
    value = fake.astype(np.float32)
    blurred = cv2.GaussianBlur(value, (0, 0), float(sigma))
    sharpened = value + (value - blurred) * float(strength)
    result = value * (1.0 - mask) + sharpened * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_aligned_camera_residual(
    real: np.ndarray, fake: np.ndarray, *, strength: float
) -> np.ndarray:
    """Add a bounded amount of target-camera high-pass grain, not RGB content."""
    mask = _change_mask(real, fake)
    real_f = real.astype(np.float32)
    residual = real_f - cv2.GaussianBlur(real_f, (0, 0), 0.8)
    value = fake.astype(np.float32) + residual * float(strength)
    result = fake.astype(np.float32) * (1.0 - mask) + value * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _camera_consistency(real: np.ndarray, fake: np.ndarray, strength: float) -> np.ndarray:
    toned = _tone_match(real, fake, amount=min(0.65, strength * 0.65))
    return _restore_detail(real, toned, amount=strength, transfer=strength * 0.50)


def _reference_blend(real: np.ndarray, fake: np.ndarray, real_fraction: float) -> np.ndarray:
    """Diagnostic lower bound; never eligible for deployment."""
    mask = _change_mask(real, fake)
    alpha = mask * float(real_fraction)
    result = fake.astype(np.float32) * (1.0 - alpha) + real.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _variants() -> dict[str, ImagePairTransform]:
    variants: dict[str, ImagePairTransform] = {
        "baseline": lambda real, fake: fake.copy(),
        "tone-25": lambda real, fake: _tone_match(real, fake, 0.25),
        "tone-50": lambda real, fake: _tone_match(real, fake, 0.50),
        "tone-75": lambda real, fake: _tone_match(real, fake, 0.75),
        "detail-25": lambda real, fake: _restore_detail(real, fake, 0.25, 0.0),
        "detail-50": lambda real, fake: _restore_detail(real, fake, 0.50, 0.0),
        "detail-75": lambda real, fake: _restore_detail(real, fake, 0.75, 0.0),
        "detail-100": lambda real, fake: _restore_detail(real, fake, 1.00, 0.0),
        "detail-125": lambda real, fake: _restore_detail(real, fake, 1.25, 0.0),
        "detail-150": lambda real, fake: _restore_detail(real, fake, 1.50, 0.0),
        "detail-200": lambda real, fake: _restore_detail(real, fake, 2.00, 0.0),
        "detail-250": lambda real, fake: _restore_detail(real, fake, 2.50, 0.0),
        "detail-300": lambda real, fake: _restore_detail(real, fake, 3.00, 0.0),
        "detail-350": lambda real, fake: _restore_detail(real, fake, 3.50, 0.0),
        "detail-400": lambda real, fake: _restore_detail(real, fake, 4.00, 0.0),
        "detail350-cap130": lambda real, fake: _restore_detail(
            real, fake, 3.50, 0.0, 1.30
        ),
        "detail350-cap140": lambda real, fake: _restore_detail(
            real, fake, 3.50, 0.0, 1.40
        ),
        "detail350-cap150": lambda real, fake: _restore_detail(
            real, fake, 3.50, 0.0, 1.50
        ),
        "detail350-cap160": lambda real, fake: _restore_detail(
            real, fake, 3.50, 0.0, 1.60
        ),
        "detail350-cap170": lambda real, fake: _restore_detail(
            real, fake, 3.50, 0.0, 1.70
        ),
        "detail350-cap180": lambda real, fake: _restore_detail(
            real, fake, 3.50, 0.0, 1.80
        ),
        "detail350-cap170-unsharp15": lambda real, fake: _masked_unsharp(
            real,
            _restore_detail(real, fake, 3.50, 0.0, 1.70),
            strength=0.15,
        ),
        "detail350-cap170-unsharp15-contrast15": lambda real, fake: (
            _masked_tonal_adjustment(
                real,
                _masked_unsharp(
                    real,
                    _restore_detail(real, fake, 3.50, 0.0, 1.70),
                    strength=0.15,
                ),
                contrast=0.85,
            )
        ),
        "texture-25": lambda real, fake: _restore_detail(real, fake, 0.50, 0.25),
        "texture-50": lambda real, fake: _restore_detail(real, fake, 0.75, 0.50),
        "texture-75": lambda real, fake: _restore_detail(real, fake, 1.00, 0.75),
        "consistent-25": lambda real, fake: _camera_consistency(real, fake, 0.25),
        "consistent-50": lambda real, fake: _camera_consistency(real, fake, 0.50),
        "consistent-75": lambda real, fake: _camera_consistency(real, fake, 0.75),
        "consistent-100": lambda real, fake: _camera_consistency(real, fake, 1.00),
        "bright+1": lambda real, fake: _masked_tonal_adjustment(real, fake, brightness=1.0),
        "bright+2": lambda real, fake: _masked_tonal_adjustment(real, fake, brightness=2.0),
        "bright+4": lambda real, fake: _masked_tonal_adjustment(real, fake, brightness=4.0),
        "bright-1": lambda real, fake: _masked_tonal_adjustment(real, fake, brightness=-1.0),
        "bright-2": lambda real, fake: _masked_tonal_adjustment(real, fake, brightness=-2.0),
        "bright-4": lambda real, fake: _masked_tonal_adjustment(real, fake, brightness=-4.0),
        "contrast-2": lambda real, fake: _masked_tonal_adjustment(real, fake, contrast=0.98),
        "contrast-5": lambda real, fake: _masked_tonal_adjustment(real, fake, contrast=0.95),
        "contrast-10": lambda real, fake: _masked_tonal_adjustment(real, fake, contrast=0.90),
        "contrast-15": lambda real, fake: _masked_tonal_adjustment(real, fake, contrast=0.85),
        "contrast-20": lambda real, fake: _masked_tonal_adjustment(real, fake, contrast=0.80),
        "contrast-10-bright-2": lambda real, fake: _masked_tonal_adjustment(
            real, fake, brightness=-2.0, contrast=0.90
        ),
        "contrast+2": lambda real, fake: _masked_tonal_adjustment(real, fake, contrast=1.02),
        "contrast+5": lambda real, fake: _masked_tonal_adjustment(real, fake, contrast=1.05),
        "saturation-5": lambda real, fake: _masked_tonal_adjustment(real, fake, saturation=0.95),
        "saturation+5": lambda real, fake: _masked_tonal_adjustment(real, fake, saturation=1.05),
        "unsharp-15": lambda real, fake: _masked_unsharp(real, fake, strength=0.15),
        "unsharp-30": lambda real, fake: _masked_unsharp(real, fake, strength=0.30),
        "unsharp-50": lambda real, fake: _masked_unsharp(real, fake, strength=0.50),
        "unsharp-75": lambda real, fake: _masked_unsharp(real, fake, strength=0.75),
        "unsharp-100": lambda real, fake: _masked_unsharp(real, fake, strength=1.00),
        "unsharp-125": lambda real, fake: _masked_unsharp(real, fake, strength=1.25),
        "unsharp-150": lambda real, fake: _masked_unsharp(real, fake, strength=1.50),
        "unsharp-175": lambda real, fake: _masked_unsharp(real, fake, strength=1.75),
        "unsharp-200": lambda real, fake: _masked_unsharp(real, fake, strength=2.00),
        "camera-grain-10": lambda real, fake: _add_aligned_camera_residual(real, fake, strength=0.10),
        "camera-grain-20": lambda real, fake: _add_aligned_camera_residual(real, fake, strength=0.20),
        "camera-grain-30": lambda real, fake: _add_aligned_camera_residual(real, fake, strength=0.30),
        "detail100-bright2": lambda real, fake: _masked_tonal_adjustment(
            real, _restore_detail(real, fake, 1.0, 0.0), brightness=2.0
        ),
        "unsharp50-bright2": lambda real, fake: _masked_tonal_adjustment(
            real, _masked_unsharp(real, fake, strength=0.50), brightness=2.0
        ),
        "reference-blend-10": lambda real, fake: _reference_blend(real, fake, 0.10),
        "reference-blend-20": lambda real, fake: _reference_blend(real, fake, 0.20),
    }
    return variants


def _load_network_classes(training_root: Path):
    """Load only Meso4/Xception without executing DeepfakeBench's broad imports."""
    package = types.ModuleType("networks")
    package.__path__ = [str(training_root / "networks")]
    sys.modules["networks"] = package
    sys.path.insert(0, str(training_root))
    from networks.mesonet import Meso4  # type: ignore
    from networks.xception import Xception  # type: ignore

    return Meso4, Xception


def _load_detector(name: str, root: Path, device: torch.device):
    Meso4, Xception = _load_network_classes(root / "training")
    if name == "meso4":
        model = Meso4({"num_classes": 2, "inc": 3})
    elif name == "xception":
        model = Xception(
            {"mode": "original", "num_classes": 2, "inc": 3, "dropout": False}
        )
    else:
        raise ValueError(f"unsupported detector: {name}")

    checkpoint = root / f"{name}_best.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    model.load_state_dict(backbone, strict=True)
    return model.eval().to(device)


def _tensor_batch(images: list[np.ndarray]) -> torch.Tensor:
    tensors = []
    for image in images:
        resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        tensors.append(tensor.div_(127.5).sub_(1.0))
    return torch.stack(tensors)


@torch.inference_mode()
def _score(model, images: list[np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = _tensor_batch(images[start : start + batch_size]).to(device)
        logits, _ = model(batch)
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64)


def _eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[index] + fpr[index]) * 0.5), float(thresholds[index])


def _detector_metrics(real_scores: np.ndarray, fake_scores: np.ndarray) -> dict:
    labels = np.concatenate([np.zeros_like(real_scores), np.ones_like(fake_scores)])
    scores = np.concatenate([real_scores, fake_scores])
    eer, eer_threshold = _eer(labels, scores)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "equal_error_rate": eer,
        "eer_threshold": eer_threshold,
        "real_mean": float(real_scores.mean()),
        "fake_mean": float(fake_scores.mean()),
        "mean_fake_minus_real": float(np.mean(fake_scores - real_scores)),
        "median_fake_minus_real": float(np.median(fake_scores - real_scores)),
        "fake_higher_fraction": float(np.mean(fake_scores > real_scores)),
    }


def _highpass_energy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    return float(residual.std())


def _quality_metrics(
    real: list[np.ndarray], baseline: list[np.ndarray], candidate: list[np.ndarray]
) -> dict:
    maes = []
    psnrs = []
    luma_deltas = []
    detail_ratios = []
    for reference, original_fake, changed_fake in zip(real, baseline, candidate):
        difference = changed_fake.astype(np.float32) - original_fake.astype(np.float32)
        mae = float(np.mean(np.abs(difference)))
        mse = float(np.mean(np.square(difference)))
        maes.append(mae)
        psnrs.append(99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse)))
        real_y = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
        fake_y = cv2.cvtColor(changed_fake, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
        luma_deltas.append(float(fake_y.mean() - real_y.mean()))
        detail_ratios.append(_highpass_energy(changed_fake) / (_highpass_energy(reference) + 1e-6))
    return {
        "mae_from_baseline": float(np.mean(maes)),
        "psnr_from_baseline_db": float(np.mean(psnrs)),
        "luma_delta_from_real": float(np.mean(luma_deltas)),
        "detail_ratio_to_real": float(np.mean(detail_ratios)),
    }


def _write_variant(path: Path, names: list[str], images: list[np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, image in zip(names, images):
        if not cv2.imwrite(str(path / name), image):
            raise OSError(f"failed to write {path / name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="benchmark runtime root")
    parser.add_argument("--corpus", type=Path, required=True, help="paired crop corpus")
    parser.add_argument("--real-subdir", default="real")
    parser.add_argument("--fake-subdir", default="fake")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", choices=("meso4", "xception"), default=["meso4", "xception"])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-variants", action="store_true")
    args = parser.parse_args()

    available = _variants()
    selected = args.variants or list(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        parser.error(f"unknown variants: {', '.join(unknown)}")

    real, fake, names = _paired_images(
        args.corpus, real_subdir=args.real_subdir, fake_subdir=args.fake_subdir
    )
    candidates = {
        name: [available[name](reference, generated) for reference, generated in zip(real, fake)]
        for name in selected
    }
    result = {
        "schema_version": "1.0",
        "classification": "diagnostic paired-corpus ablation; not a leaderboard result",
        "corpus": str(args.corpus),
        "pairs": len(real),
        "variants": {
            name: {"quality": _quality_metrics(real, fake, images), "detectors": {}}
            for name, images in candidates.items()
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result["device"] = str(device)
    for detector_name in args.detectors:
        started = time.perf_counter()
        model = _load_detector(detector_name, args.root, device)
        real_scores = _score(model, real, device, args.batch_size)
        for variant_name, images in candidates.items():
            fake_scores = _score(model, images, device, args.batch_size)
            result["variants"][variant_name]["detectors"][detector_name] = _detector_metrics(
                real_scores, fake_scores
            )
        result.setdefault("runtime_seconds", {})[detector_name] = time.perf_counter() - started
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_variants:
        output_root = args.output.with_suffix("")
        for name, images in candidates.items():
            _write_variant(output_root / name, names, images)

    compact = {
        name: {
            "quality": metrics["quality"],
            **{
                detector: detector_metrics["roc_auc"]
                for detector, detector_metrics in metrics["detectors"].items()
            },
        }
        for name, metrics in result["variants"].items()
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
