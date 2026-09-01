"""Training-only reconstruction, identity, mask and temporal objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import LossConfig


class FeatureExtractor(Protocol):
    def __call__(self, value: Tensor) -> Tensor: ...


def masked_mean(value: Tensor, sample_mask: Tensor | None = None) -> Tensor:
    if sample_mask is None:
        return value.mean()
    weights = sample_mask.to(dtype=value.dtype, device=value.device)
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.expand_as(value).sum().clamp_min(1.0)
    return (value * weights).sum() / denominator


def charbonnier(
    prediction: Tensor,
    target: Tensor,
    sample_mask: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    return masked_mean(
        torch.sqrt((prediction - target) ** 2 + epsilon**2), sample_mask
    )


def image_gradients(value: Tensor) -> tuple[Tensor, Tensor]:
    return value[:, :, :, 1:] - value[:, :, :, :-1], value[:, :, 1:, :] - value[:, :, :-1, :]


def gradient_loss(
    prediction: Tensor, target: Tensor, sample_mask: Tensor | None = None
) -> Tensor:
    px, py = image_gradients(prediction)
    tx, ty = image_gradients(target)
    return masked_mean((px - tx).abs(), sample_mask) + masked_mean(
        (py - ty).abs(), sample_mask
    )


def total_variation(value: Tensor) -> Tensor:
    dx, dy = image_gradients(value)
    return dx.abs().mean() + dy.abs().mean()


def normalized_embedding(value: Tensor) -> Tensor:
    return F.normalize(value.flatten(1), dim=1, eps=1e-8)


def arcface_input(value: Tensor) -> Tensor:
    value = F.interpolate(value, size=(112, 112), mode="bilinear", align_corners=False)
    return (value - 0.5) / 0.5


def identity_loss(
    output: Tensor,
    source_embedding: Tensor,
    extractor: FeatureExtractor,
    sample_mask: Tensor | None = None,
) -> Tensor:
    generated = normalized_embedding(extractor(arcface_input(output)))
    source = normalized_embedding(source_embedding)
    distance = 1.0 - (generated * source).sum(dim=1)
    return masked_mean(distance, sample_mask)


def background_preservation(
    output: Tensor,
    target: Tensor,
    face_mask: Tensor,
    sample_mask: Tensor | None = None,
) -> Tensor:
    background = 1.0 - face_mask.clamp(0.0, 1.0)
    return masked_mean((output - target).abs() * background, sample_mask)


def mask_background_loss(
    alpha: Tensor,
    face_mask: Tensor,
    sample_mask: Tensor | None = None,
) -> Tensor:
    return masked_mean(
        alpha * (1.0 - face_mask.clamp(0.0, 1.0)), sample_mask
    )


def mask_foreground_loss(
    alpha: Tensor,
    face_mask: Tensor,
    sample_mask: Tensor | None = None,
) -> Tensor:
    """Penalize false negatives inside a supervised face foreground."""
    return masked_mean(
        (1.0 - alpha) * face_mask.clamp(0.0, 1.0), sample_mask
    )


def flow_grid(flow: Tensor) -> Tensor:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [batch,2,height,width]")
    batch, _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=flow.dtype, device=flow.device),
        torch.arange(width, dtype=flow.dtype, device=flow.device),
        indexing="ij",
    )
    base_x = xx.unsqueeze(0).expand(batch, -1, -1) + flow[:, 0]
    base_y = yy.unsqueeze(0).expand(batch, -1, -1) + flow[:, 1]
    if width > 1:
        base_x = base_x * (2.0 / (width - 1)) - 1.0
    else:
        base_x = torch.zeros_like(base_x)
    if height > 1:
        base_y = base_y * (2.0 / (height - 1)) - 1.0
    else:
        base_y = torch.zeros_like(base_y)
    return torch.stack((base_x, base_y), dim=-1)


def warp(value: Tensor, flow: Tensor) -> Tensor:
    return F.grid_sample(
        value,
        flow_grid(flow),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def temporal_losses(
    output: Tensor,
    target: Tensor,
    alpha: Tensor,
    next_output: Tensor,
    next_target: Tensor,
    next_alpha: Tensor,
    flow: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor]:
    valid = valid.clamp(0.0, 1.0)
    residual = output - target
    next_residual = next_output - next_target
    denominator = valid.sum().clamp_min(1.0)
    residual_loss = (
        (warp(residual, flow) - next_residual).abs() * valid
    ).sum() / (denominator * output.shape[1])
    alpha_loss = ((warp(alpha, flow) - next_alpha).abs() * valid).sum() / denominator
    return residual_loss, alpha_loss


class PatchDiscriminator(nn.Module):
    """Small training-only PatchGAN; it is never exported with the student."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = (3, base, base * 2, base * 4, base * 8)
        for index, (in_channels, out_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            layers.append(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
            )
            if index:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        layers.append(nn.Conv2d(channels[-1], 1, 3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


def discriminator_hinge(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge(fake_logits: Tensor) -> Tensor:
    return -fake_logits.mean()


@dataclass
class LossInputs:
    target: Tensor
    candidate: Tensor
    alpha: Tensor
    teacher_candidate: Tensor
    teacher_alpha: Tensor
    teacher_confidence: Tensor
    has_teacher: Tensor
    same_identity: Tensor
    face_mask: Tensor
    has_face_mask: Tensor
    source_embedding: Tensor
    has_source_embedding: Tensor


class Native256Objective(nn.Module):
    """Core still-image objective; temporal and GAN losses are added by trainers."""

    def __init__(
        self,
        weights: LossConfig | None = None,
        identity_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossConfig()
        self.weights.validate()
        self.identity_extractor = identity_extractor
        if self.identity_extractor is not None:
            self.identity_extractor.eval()
            for parameter in self.identity_extractor.parameters():
                parameter.requires_grad_(False)

    def forward(self, inputs: LossInputs) -> dict[str, Tensor]:
        output = inputs.alpha * inputs.candidate + (1.0 - inputs.alpha) * inputs.target
        student_output = F.interpolate(output, size=(128, 128), mode="area")
        teacher_target = F.interpolate(
            inputs.target, size=(128, 128), mode="area"
        )
        teacher_alpha = inputs.teacher_alpha.clamp(0.0, 1.0)
        teacher_output = (
            teacher_alpha * inputs.teacher_candidate
            + (1.0 - teacher_alpha) * teacher_target
        )
        teacher_mask = inputs.has_teacher.to(torch.float32) * inputs.teacher_confidence
        teacher = charbonnier(student_output, teacher_output, teacher_mask)
        self_pixel = charbonnier(output, inputs.target, inputs.same_identity)
        self_grad = gradient_loss(output, inputs.target, inputs.same_identity)
        background = background_preservation(
            output, inputs.target, inputs.face_mask, inputs.has_face_mask
        )
        mask_background = mask_background_loss(
            inputs.alpha, inputs.face_mask, inputs.has_face_mask
        )
        mask_foreground = mask_foreground_loss(
            inputs.alpha, inputs.face_mask, inputs.has_face_mask
        )
        mask_area = inputs.alpha.mean()
        mask_tv = total_variation(inputs.alpha)

        zero = output.sum() * 0.0
        identity = zero
        if self.identity_extractor is not None:
            identity = identity_loss(
                output,
                inputs.source_embedding,
                self.identity_extractor,
                inputs.has_source_embedding,
            )
        losses = {
            "teacher_pixel": teacher,
            "identity": identity,
            "self_pixel": self_pixel,
            "self_gradient": self_grad,
            "background": background,
            "mask_background": mask_background,
            "mask_foreground": mask_foreground,
            "mask_area": mask_area,
            "mask_tv": mask_tv,
        }
        total = sum(
            losses[name] * float(getattr(self.weights, name)) for name in losses
        )
        losses["total"] = total
        losses["output"] = output
        return losses


__all__ = [
    "LossInputs",
    "Native256Objective",
    "PatchDiscriminator",
    "arcface_input",
    "background_preservation",
    "charbonnier",
    "discriminator_hinge",
    "generator_hinge",
    "gradient_loss",
    "identity_loss",
    "mask_background_loss",
    "mask_foreground_loss",
    "temporal_losses",
    "total_variation",
    "warp",
]
