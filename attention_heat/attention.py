"""Attention extraction and heatmap utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from .model import TaskSpec

LOGGER = logging.getLogger(__name__)


def normalize_map(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a tensor to [0, 1] range per sample."""

    min_vals = tensor.amin(dim=(-2, -1), keepdim=True)
    tensor = tensor - min_vals
    max_vals = tensor.amax(dim=(-2, -1), keepdim=True)
    tensor = tensor / (max_vals + eps)
    return tensor.clamp_(0.0, 1.0)


class LastLayerAttentionRecorder:
    """Hook utility that records the attention map of the final Swin block."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        # Locate the final attention module of the Swin backbone.
        try:
            layers = backbone.layers  # type: ignore[attr-defined]
        except AttributeError as exc:  # pragma: no cover - defensive programming
            raise AttributeError(
                "Expected Swin Transformer backbone with a 'layers' attribute"
            ) from exc

        if not layers:
            raise ValueError("Backbone does not expose any Swin layers")

        last_layer = layers[-1]
        if not getattr(last_layer, "blocks", None):
            raise ValueError("Last Swin layer does not expose transformer blocks")

        self.block = last_layer.blocks[-1]
        self.attention_module = self.block.attn
        # Ensure we follow the code path that exposes the softmax tensor.
        if getattr(self.attention_module, "fused_attn", False):
            LOGGER.debug("Disabling fused attention for heatmap extraction")
            self.attention_module.fused_attn = False

        self.attention: torch.Tensor | None = None
        self.handle = self.attention_module.softmax.register_forward_hook(self._hook)  # type: ignore[arg-type]

    def _hook(self, _module: torch.nn.Module, _inputs, output: torch.Tensor) -> None:
        self.attention = output.detach()

    def clear(self) -> None:
        self.attention = None

    def remove(self) -> None:
        self.handle.remove()

    @property
    def window_size(self) -> Tuple[int, int]:
        size = getattr(self.attention_module, "window_size", None)
        if size is None:
            raise AttributeError("Attention module does not expose window size")
        if isinstance(size, int):
            return (size, size)
        return tuple(size)

    def build_attention_map(
        self,
        batch_size: int,
        feature_resolution: Tuple[int, int],
        image_resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Stitch the recorded window attentions into a full-resolution heatmap."""

        if self.attention is None:
            raise RuntimeError("No attention map recorded. Run a forward pass first.")

        window_h, window_w = self.window_size
        tokens_per_window = window_h * window_w
        feature_h, feature_w = feature_resolution
        num_windows = (feature_h // window_h) * (feature_w // window_w)
        heads = self.attention.shape[1]

        attn = self.attention.view(
            batch_size,
            num_windows,
            heads,
            tokens_per_window,
            tokens_per_window,
        )
        attn = attn.mean(dim=2)  # average across heads
        attn = attn.mean(dim=-2)  # average across query positions
        attn = attn.view(batch_size, num_windows, window_h, window_w)

        full_map = attn.new_zeros((batch_size, 1, feature_h, feature_w))

        window_index = 0
        for y in range(0, feature_h, window_h):
            for x in range(0, feature_w, window_w):
                full_map[:, :, y : y + window_h, x : x + window_w] = attn[:, window_index].unsqueeze(1)
                window_index += 1

        full_map = normalize_map(full_map)
        full_map = F.interpolate(
            full_map,
            size=image_resolution,
            mode="bicubic",
            align_corners=False,
        )
        full_map = normalize_map(full_map)
        return full_map.squeeze(1)


def compute_task_heatmaps(
    predictions: Dict[str, torch.Tensor],
    attention_map: torch.Tensor,
    specs: Iterable[TaskSpec],
    image_resolution: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """Combine attention map and task activations into task-specific heatmaps."""

    spec_map = {spec.name: spec for spec in specs}
    batch = attention_map.shape[0]
    heatmaps: Dict[str, torch.Tensor] = {}

    for name, prediction in predictions.items():
        spec = spec_map.get(name)
        if spec is None:
            LOGGER.warning("No TaskSpec provided for task '%s'; skipping heatmap", name)
            continue

        activation = spec.activation_map(prediction)
        activation = F.interpolate(
            activation,
            size=image_resolution,
            mode="bilinear",
            align_corners=False,
        )
        activation = normalize_map(activation)

        combined = normalize_map(activation.squeeze(1) * attention_map)
        heatmaps[name] = combined.view(batch, *image_resolution)

    return heatmaps