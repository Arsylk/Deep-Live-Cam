"""Strict JSON configuration for native-256 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 256
    identity_size: int = 512
    conditioner_hidden: int = 128
    channels: tuple[int, int, int, int, int] = (32, 48, 72, 112, 160)
    safety_inner_radius: float = 0.82
    safety_outer_radius: float = 1.0

    def validate(self) -> None:
        if self.input_size != 256:
            raise ValueError("DLC-Swap256-M has a fixed 256px input contract")
        if self.identity_size != 512:
            raise ValueError("DLC-Swap256-M requires a 512-value identity latent")
        if tuple(self.channels) != (32, 48, 72, 112, 160):
            raise ValueError("channel widths are part of the DLC-Swap256-M contract")
        if not 0.0 < self.safety_inner_radius < self.safety_outer_radius:
            raise ValueError("safety radii must satisfy 0 < inner < outer")


@dataclass(frozen=True)
class LossConfig:
    teacher_pixel: float = 20.0
    identity: float = 8.0
    self_pixel: float = 20.0
    self_gradient: float = 2.0
    background: float = 10.0
    mask_background: float = 2.0
    mask_foreground: float = 2.0
    mask_area: float = 0.02
    mask_tv: float = 0.1
    # Still-image training is the default. Set these explicitly for a manifest
    # containing next_target/flow/flow_valid tuples.
    temporal_residual: float = 0.0
    temporal_mask: float = 0.0

    def validate(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if value < 0.0:
                raise ValueError(f"loss weight {item.name} cannot be negative")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260814
    batch_size: int = 8
    workers: int = 4
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    steps: int = 300000
    checkpoint_interval: int = 5000
    log_interval: int = 50
    amp: bool = True
    ema_decay: float = 0.999
    identity_model: str | None = None

    def validate(self) -> None:
        if self.batch_size < 1 or self.workers < 0:
            raise ValueError("batch_size must be positive and workers cannot be negative")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer values are invalid")
        if self.steps < 1 or self.checkpoint_interval < 1 or self.log_interval < 1:
            raise ValueError("training intervals must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")


@dataclass(frozen=True)
class Native256Config:
    model: ModelConfig = ModelConfig()
    losses: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()

    def validate(self) -> None:
        self.model.validate()
        self.losses.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model"]["channels"] = list(self.model.channels)
        return value


T = TypeVar("T")


def _strict_dataclass(cls: type[T], value: object, label: str) -> T:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} keys: {', '.join(unknown)}")
    data = dict(value)
    if cls is ModelConfig and "channels" in data:
        channels = data["channels"]
        if not isinstance(channels, list) or len(channels) != 5:
            raise ValueError("model.channels must contain five integers")
        data["channels"] = tuple(int(item) for item in channels)
    return cls(**data)


def load_config(path: str | Path) -> Native256Config:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a JSON object")
    unknown = sorted(set(data) - {"model", "losses", "training"})
    if unknown:
        raise ValueError(f"unknown configuration sections: {', '.join(unknown)}")
    config = Native256Config(
        model=_strict_dataclass(ModelConfig, data.get("model", {}), "model"),
        losses=_strict_dataclass(LossConfig, data.get("losses", {}), "losses"),
        training=_strict_dataclass(
            TrainingConfig, data.get("training", {}), "training"
        ),
    )
    config.validate()
    return config
