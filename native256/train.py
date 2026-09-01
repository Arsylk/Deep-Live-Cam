"""Minimal, reproducible training entry point for DLC-Swap256-M."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import Native256Config, load_config
from .dataset import Native256Dataset
from .losses import LossInputs, Native256Objective, temporal_losses
from .identity import validate_emap
from .model import DlcSwap256M, GENERATOR_MACS, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--identity-map",
        dest="identity_map",
        type=Path,
        required=True,
        help="exact float32 [512,512] buff2fs map used to prepare source_latent",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, help="override configured step count")
    parser.add_argument(
        "--allow-missing-identity-loss",
        action="store_true",
        help="structural smoke training only; not suitable for a quality checkpoint",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite(loader: DataLoader[dict[str, Tensor]]) -> Iterator[dict[str, Tensor]]:
    while True:
        yield from loader


def build_training_binding(
    args: argparse.Namespace,
    config: Native256Config,
    dataset: Native256Dataset,
    identity_metadata: dict[str, str | int],
) -> dict[str, object]:
    """Bind a checkpoint to the exact authorized manifest and configuration."""
    manifest_bytes = Path(args.manifest).read_bytes()
    config_bytes = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    inventory_bytes = json.dumps(
        [asdict(record) for record in dataset.records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "sample_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "sample_count": len(dataset.records),
        "data_root": str(Path(args.data_root).resolve()),
        "identity": identity_metadata,
    }


def load_identity_model(
    config: Native256Config, device: torch.device, allow_missing: bool
) -> nn.Module | None:
    path_text = config.training.identity_model
    if path_text is None:
        if config.losses.identity > 0.0 and not allow_missing:
            raise RuntimeError(
                "identity loss is enabled but training.identity_model is null; "
                "supply a differentiable TorchScript ArcFace model or use "
                "--allow-missing-identity-loss for a structural smoke run"
            )
        return None
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"identity model does not exist: {path}")
    model = torch.jit.load(str(path), map_location=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    source = model.state_dict()
    for name, target in ema.state_dict().items():
        current = source[name]
        if target.is_floating_point():
            target.mul_(decay).add_(current, alpha=1.0 - decay)
        else:
            target.copy_(current)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: DlcSwap256M,
    ema: DlcSwap256M,
    optimizer: torch.optim.Optimizer,
    config: Native256Config,
    identity_metadata: dict[str, str | int],
    training_binding: dict[str, object],
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "dlc-swap256m-v1",
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "identity": identity_metadata,
            "training_binding": training_binding,
            "scaler": scaler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "loader": loader_generator.get_state(),
            },
            "quality_status": "development",
            "auto_select_eligible": False,
        },
        temporary,
    )
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    identity_metadata = validate_emap(args.identity_map)
    total_steps = args.steps if args.steps is not None else config.training.steps
    if total_steps < 1:
        raise ValueError("steps must be positive")
    seed_everything(config.training.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = Native256Dataset(
        args.manifest,
        args.data_root,
        "train",
        identity_map_sha256=str(identity_metadata["emap_sha256"]),
        identity_map=np.load(args.identity_map, allow_pickle=False),
    )
    training_binding = build_training_binding(
        args, config, dataset, identity_metadata
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= config.training.batch_size,
        persistent_workers=config.training.workers > 0,
        generator=generator,
    )
    batches = infinite(loader)

    model = DlcSwap256M(config.model).to(device)
    ema = copy.deepcopy(model).eval()
    identity_model = load_identity_model(
        config, device, args.allow_missing_identity_loss
    )
    if identity_model is not None:
        missing_identity = [
            record.sample_id
            for record in dataset.records
            if record.source_embedding is None
        ]
        if missing_identity:
            raise RuntimeError(
                "identity loss is active but training samples lack "
                "source_embedding; first missing sample: "
                f"{missing_identity[0]}"
            )
    temporal_active = bool(
        config.losses.temporal_residual > 0.0
        or config.losses.temporal_mask > 0.0
    )
    temporal_samples = sum(
        record.next_target is not None for record in dataset.records
    )
    if temporal_active and temporal_samples == 0:
        raise RuntimeError(
            "temporal loss is enabled but the training split has no "
            "next_target/flow/flow_valid samples"
        )
    objective = Native256Objective(config.losses, identity_model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.5, 0.99),
    )
    use_amp = bool(config.training.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("format") != "dlc-swap256m-v1":
            raise RuntimeError("resume checkpoint has an incompatible format")
        if checkpoint.get("identity") != identity_metadata:
            raise RuntimeError(
                "resume checkpoint identity contract or emap hash does not match "
                "--identity-map"
            )
        if checkpoint.get("training_binding") != training_binding:
            raise RuntimeError(
                "resume checkpoint was created from a different manifest, "
                "configuration, data root, or sample inventory"
            )
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint["step"])
        rng = checkpoint.get("rng")
        if not isinstance(rng, dict):
            raise RuntimeError("resume checkpoint has no reproducible RNG state")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        generator.set_state(rng["loader"])

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "dlc-swap256m-training-v1",
        "parameters": parameter_count(model),
        "generator_macs": GENERATOR_MACS,
        "config": config.to_dict(),
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "identity_loss_active": identity_model is not None,
        "identity_embedding_coverage": sum(
            record.source_embedding is not None for record in dataset.records
        )
        / len(dataset.records),
        "temporal_loss_active": temporal_active,
        "temporal_sample_count": temporal_samples,
        "identity": identity_metadata,
        "training_binding": training_binding,
        "quality_status": "development",
        "auto_select_eligible": False,
    }
    (args.output / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    model.train()
    last_log = time.perf_counter()
    for step in range(start_step + 1, total_steps + 1):
        batch = move_batch(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with context:
            style = model.condition(batch["source_latent"])
            candidate, alpha = model.forward_with_style(batch["target"], style)
            result = objective(
                LossInputs(
                    target=batch["target"],
                    candidate=candidate,
                    alpha=alpha,
                    teacher_candidate=batch["teacher_candidate"],
                    teacher_alpha=batch["teacher_alpha"],
                    teacher_confidence=batch["teacher_confidence"],
                    has_teacher=batch["has_teacher"],
                    same_identity=batch["same_identity"],
                    face_mask=batch["face_mask"],
                    has_face_mask=batch["has_face_mask"],
                    source_embedding=batch["source_embedding"],
                    has_source_embedding=batch["has_source_embedding"],
                )
            )
            temporal_residual = result["output"].sum() * 0.0
            temporal_mask = result["output"].sum() * 0.0
            temporal_selector = batch["has_temporal"].to(torch.bool)
            if temporal_active and bool(temporal_selector.any()):
                next_candidate, next_alpha = model.forward_with_style(
                    batch["next_target"][temporal_selector],
                    style[temporal_selector],
                )
                next_output = model.composite(
                    batch["next_target"][temporal_selector],
                    next_candidate,
                    next_alpha,
                )
                temporal_residual, temporal_mask = temporal_losses(
                    result["output"][temporal_selector],
                    batch["target"][temporal_selector],
                    alpha[temporal_selector],
                    next_output,
                    batch["next_target"][temporal_selector],
                    next_alpha,
                    batch["flow"][temporal_selector],
                    batch["flow_valid"][temporal_selector],
                )
            result["temporal_residual"] = temporal_residual
            result["temporal_mask"] = temporal_mask
            result["total"] = (
                result["total"]
                + temporal_residual * config.losses.temporal_residual
                + temporal_mask * config.losses.temporal_mask
            )
        scaler.scale(result["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, config.training.ema_decay)

        if step == 1 or step % config.training.log_interval == 0:
            now = time.perf_counter()
            values = {
                name: round(float(value.detach()), 6)
                for name, value in result.items()
                if name != "output"
            }
            values.update(
                {
                    "step": step,
                    "seconds_per_log_window": round(now - last_log, 3),
                }
            )
            print(json.dumps(values), flush=True)
            last_log = now
        if step % config.training.checkpoint_interval == 0 or step == total_steps:
            save_checkpoint(
                args.output / f"checkpoint-{step:08d}.pt",
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                config=config,
                identity_metadata=identity_metadata,
                training_binding=training_binding,
                scaler=scaler,
                loader_generator=generator,
            )


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
