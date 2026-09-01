"""DLC-Swap256-M teacher-student architecture.

The deployed form is deliberately split into an identity conditioner, which
runs when the source changes, and a per-frame generator. Both graphs use only
fixed-shape, conventional convolutional operations that map cleanly to ONNX,
ncnn and mobile execution providers.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig
from .identity import IDENTITY_CONTRACT


INPUT_SIZE = 256
IDENTITY_SIZE = 512
CHANNELS = (32, 48, 72, 112, 160)
CONDITION_CHANNELS = (72, 112, 112, 160, 160, 160, 112, 72, 48)
STYLE_SIZE = 2 * sum(CONDITION_CHANNELS)

# Analytic multiply-accumulate count at batch one. This includes the RGB and
# semantic-mask paths but excludes the source conditioner, which is cached.
GENERATOR_MACS = 970_907_648


def _check_image(value: Tensor, name: str) -> None:
    if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
        return
    if value.ndim != 4 or value.shape[1:] != (3, INPUT_SIZE, INPUT_SIZE):
        raise ValueError(
            f"{name} must have shape [batch,3,{INPUT_SIZE},{INPUT_SIZE}], "
            f"got {tuple(value.shape)}"
        )


def _check_matrix(value: Tensor, width: int, name: str) -> None:
    if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
        return
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [batch,{width}], got {tuple(value.shape)}")


def _check_style(value: Tensor) -> None:
    if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
        return
    if value.ndim != 4 or value.shape[1:] != (STYLE_SIZE, 1, 1):
        raise ValueError(
            f"style must have shape [batch,{STYLE_SIZE},1,1], got "
            f"{tuple(value.shape)}"
        )


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.LeakyReLU(0.1, inplace=False))
        super().__init__(*layers)


class DepthwiseSeparable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(
            in_channels, out_channels, 1, activate=activate
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(value))


class ResidualDS(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            channels, channels, 3, groups=channels
        )
        self.pointwise = ConvBNAct(channels, channels, 1, activate=False)
        self.activation = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.pointwise(self.depthwise(value)))


class ConditionedResidualDS(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.depthwise = ConvBNAct(
            channels, channels, 3, groups=channels
        )
        self.pointwise = ConvBNAct(channels, channels, 1, activate=False)
        self.activation = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, value: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
            expected = (value.shape[0], self.channels, 1, 1)
            if gamma.shape != expected or beta.shape != expected:
                raise ValueError(
                    f"conditioning must have shape {expected}, got "
                    f"{tuple(gamma.shape)} and {tuple(beta.shape)}"
                )
        update = self.pointwise(self.depthwise(value))
        # The cached style is already NCHW. Keeping these broadcast tensors
        # explicitly three-dimensional after ncnn drops the fixed batch axis
        # avoids pnnx interpreting C as tensor depth in a four-dimensional
        # Reshape, which corrupts subsequent BinaryOp shapes.
        scale = 1.0 + 0.5 * torch.tanh(gamma)
        bias = 0.25 * torch.tanh(beta)
        return self.activation(value + update * scale + bias)


class UpFusion(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = DepthwiseSeparable(in_channels + skip_channels, out_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        return self.project(torch.cat((value, skip), dim=1))


class IdentityConditioner(nn.Module):
    """Turn a mapped+normalized buff2fs latent into cached FiLM data.

    This class must never be fed the recognizer's raw ArcFace embedding. The
    required contract is ``inswapper-buff2fs-mapped-l2-v1``.
    """

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.project = nn.Linear(IDENTITY_SIZE, hidden)
        self.activation = nn.LeakyReLU(0.1, inplace=False)
        self.to_style = nn.Linear(hidden, STYLE_SIZE)
        # Begin as an identity-independent U-Net. This avoids unstable random
        # per-channel scaling at the start of reconstruction warm-up.
        nn.init.zeros_(self.to_style.weight)
        nn.init.zeros_(self.to_style.bias)

    def forward(self, mapped_identity: Tensor) -> Tensor:
        _check_matrix(mapped_identity, IDENTITY_SIZE, "mapped_identity")
        style = self.to_style(self.activation(self.project(mapped_identity)))
        return style.unsqueeze(2).unsqueeze(3)


def split_style(style: Tensor) -> tuple[tuple[Tensor, Tensor], ...]:
    _check_style(style)
    result: list[tuple[Tensor, Tensor]] = []
    offset = 0
    for channels in CONDITION_CHANNELS:
        gamma = style[:, offset : offset + channels]
        offset += channels
        beta = style[:, offset : offset + channels]
        offset += channels
        result.append((gamma, beta))
    return tuple(result)


def make_safety_prior(inner: float = 0.82, outer: float = 1.0) -> Tensor:
    if not 0.0 < inner < outer:
        raise ValueError("safety prior radii must satisfy 0 < inner < outer")
    axis = torch.linspace(-1.0, 1.0, INPUT_SIZE, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    # The aligned crop is wider than the useful facial oval. These radii make
    # the corners exactly zero while leaving mask selection inside learned.
    radius = torch.sqrt((xx / 0.90) ** 2 + (yy / 0.98) ** 2)
    phase = torch.clamp((radius - inner) / (outer - inner), 0.0, 1.0)
    transition = 0.5 * (1.0 + torch.cos(math.pi * phase))
    prior = torch.where(radius <= inner, torch.ones_like(radius), transition)
    prior = torch.where(radius >= outer, torch.zeros_like(radius), prior)
    return prior.unsqueeze(0).unsqueeze(0)


class MaskFusionStage(nn.Module):
    """Fuse one target-encoder scale into the semantic mask path."""

    def __init__(self, skip_channels: int, mask_channels: int = 8) -> None:
        super().__init__()
        self.reduce_skip = ConvBNAct(skip_channels, 4, 1)
        self.refine = DepthwiseSeparable(mask_channels + 4, mask_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        return self.refine(torch.cat((value, self.reduce_skip(skip)), dim=1))


class SemanticMaskDecoder(nn.Module):
    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: tuple[int, int, int, int],
        inner: float,
        outer: float,
    ) -> None:
        super().__init__()
        self.project = ConvBNAct(bottleneck_channels, 8, 1)
        self.stages = nn.ModuleList(
            MaskFusionStage(channels) for channels in skip_channels
        )
        self.logits = nn.Conv2d(8, 1, 3, padding=1)
        nn.init.constant_(self.logits.bias, -1.5)
        self.register_buffer("safety_prior", make_safety_prior(inner, outer))

    def forward(
        self,
        bottleneck: Tensor,
        e3: Tensor,
        e2: Tensor,
        e1: Tensor,
        e0: Tensor,
    ) -> Tensor:
        value = self.project(bottleneck)
        for stage, skip in zip(self.stages, (e3, e2, e1, e0)):
            value = stage(value, skip)
        return torch.sigmoid(self.logits(value)) * self.safety_prior


class Swap256Generator(nn.Module):
    """Per-frame target+cached-style graph."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        config.validate()
        c0, c1, c2, c3, c4 = config.channels

        self.stem = ConvBNAct(3, c0, 3)
        self.encoder0 = ResidualDS(c0)
        self.down1 = DepthwiseSeparable(c0, c1, stride=2)
        self.encoder1 = ResidualDS(c1)
        self.down2 = DepthwiseSeparable(c1, c2, stride=2)
        self.encoder2 = ConditionedResidualDS(c2)
        self.down3 = DepthwiseSeparable(c2, c3, stride=2)
        self.encoder3a = ConditionedResidualDS(c3)
        self.encoder3b = ConditionedResidualDS(c3)
        self.down4 = DepthwiseSeparable(c3, c4, stride=2)
        self.bottleneck0 = ConditionedResidualDS(c4)
        self.bottleneck1 = ConditionedResidualDS(c4)
        self.bottleneck2 = ConditionedResidualDS(c4)

        self.up3 = UpFusion(c4, c3, c3)
        self.decoder3 = ConditionedResidualDS(c3)
        self.up2 = UpFusion(c3, c2, c2)
        self.decoder2 = ConditionedResidualDS(c2)
        self.up1 = UpFusion(c2, c1, c1)
        self.decoder1 = ConditionedResidualDS(c1)
        self.up0 = UpFusion(c1, c0, c0)
        self.decoder0 = ResidualDS(c0)
        self.rgb_head = DepthwiseSeparable(c0, 3, activate=False)
        self.mask_head = SemanticMaskDecoder(
            c4,
            (c3, c2, c1, c0),
            config.safety_inner_radius,
            config.safety_outer_radius,
        )

    def forward(self, target: Tensor, style: Tensor) -> tuple[Tensor, Tensor]:
        _check_image(target, "target")
        conditions = split_style(style)
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
            if target.shape[0] != style.shape[0]:
                raise ValueError("target and style batch sizes differ")

        e0 = self.encoder0(self.stem(target))
        e1 = self.encoder1(self.down1(e0))
        e2 = self.encoder2(self.down2(e1), *conditions[0])
        e3 = self.down3(e2)
        e3 = self.encoder3a(e3, *conditions[1])
        e3 = self.encoder3b(e3, *conditions[2])
        bottleneck = self.down4(e3)
        bottleneck = self.bottleneck0(bottleneck, *conditions[3])
        bottleneck = self.bottleneck1(bottleneck, *conditions[4])
        bottleneck = self.bottleneck2(bottleneck, *conditions[5])

        value = self.decoder3(self.up3(bottleneck, e3), *conditions[6])
        value = self.decoder2(self.up2(value, e2), *conditions[7])
        value = self.decoder1(self.up1(value, e1), *conditions[8])
        value = self.decoder0(self.up0(value, e0))
        candidate = torch.sigmoid(self.rgb_head(value))
        alpha = self.mask_head(bottleneck, e3, e2, e1, e0)
        return candidate, alpha


class DlcSwap256M(nn.Module):
    """Training graph accepting target crop and current mapped identity."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        config.validate()
        self.conditioner = IdentityConditioner(config.conditioner_hidden)
        self.generator = Swap256Generator(config)

    def condition(self, mapped_identity: Tensor) -> Tensor:
        return self.conditioner(mapped_identity)

    def forward_with_style(
        self, target: Tensor, style: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.generator(target, style)

    def forward(
        self, target: Tensor, mapped_identity: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.generator(target, self.conditioner(mapped_identity))

    @staticmethod
    def composite(target: Tensor, candidate: Tensor, alpha: Tensor) -> Tensor:
        return alpha * candidate + (1.0 - alpha) * target


def parameter_count(modules: nn.Module | Iterable[nn.Module]) -> int:
    if isinstance(modules, nn.Module):
        modules = (modules,)
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


__all__ = [
    "CHANNELS",
    "CONDITION_CHANNELS",
    "DlcSwap256M",
    "GENERATOR_MACS",
    "IDENTITY_CONTRACT",
    "IDENTITY_SIZE",
    "INPUT_SIZE",
    "IdentityConditioner",
    "STYLE_SIZE",
    "Swap256Generator",
    "make_safety_prior",
    "parameter_count",
    "split_style",
]
