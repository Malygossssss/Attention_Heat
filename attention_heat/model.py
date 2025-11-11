"""Model utilities for running multi-task inference with Swin Transformer backbones."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - helpful runtime error message
    raise ImportError(
        "The 'timm' package is required to instantiate Swin Transformer models. "
        "Install it with `pip install timm`."
    ) from exc


@dataclass(frozen=True)
class TaskSpec:
    """Configuration describing the decoder head for a downstream task."""

    name: str
    output_channels: int
    activation: Optional[str] = None
    upsample_to_input: bool = True

    def activation_map(self, prediction: torch.Tensor) -> torch.Tensor:
        """Derive a single-channel activation map from a task prediction."""

        if prediction.ndim != 4:
            raise ValueError(
                f"Expected prediction tensor with 4 dimensions (B, C, H, W); "
                f"received shape {tuple(prediction.shape)}"
            )

        if self.activation == "softmax":
            activation = torch.softmax(prediction, dim=1).max(dim=1, keepdim=True).values
        elif self.activation == "sigmoid":
            activation = torch.sigmoid(prediction)
        elif self.activation == "tanh":
            activation = torch.tanh(prediction).norm(p=2, dim=1, keepdim=True)
        else:
            activation = prediction.abs().mean(dim=1, keepdim=True)

        return activation


class TaskHead(nn.Module):
    """A lightweight decoder head operating on the Swin feature map."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(in_channels // 2, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiTaskSwin(nn.Module):
    """Wrapper combining a Swin backbone with task-specific decoder heads."""

    def __init__(
        self,
        tasks: Iterable[TaskSpec],
        backbone: str = "swin_tiny_patch4_window7_224",
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained_backbone,
            features_only=False,
        )
        # Drop the classification head – downstream tasks provide their own decoders.
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)

        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            raise AttributeError("Unable to determine embedding dimension for Swin backbone")

        self.task_specs: Dict[str, TaskSpec] = {spec.name: spec for spec in tasks}
        self.task_heads = nn.ModuleDict(
            {name: TaskHead(embed_dim, spec.output_channels) for name, spec in self.task_specs.items()}
        )

        # Register recorder for the last attention module. Import placed here to avoid cycle.
        from .attention import LastLayerAttentionRecorder

        self.attention_recorder = LastLayerAttentionRecorder(self.backbone)

    def forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Run the Swin backbone and return the final feature map and its resolution."""

        features = self.backbone.forward_features(x)
        if features.ndim != 3:
            raise RuntimeError(
                "Unexpected backbone output shape. Expected (B, L, C) from Swin transformer."
            )

        b, num_tokens, channels = features.shape
        spatial_dim = int(math.sqrt(num_tokens))
        if spatial_dim * spatial_dim != num_tokens:
            raise RuntimeError(
                "Backbone output tokens cannot be reshaped into a square feature map."
            )

        features = features.transpose(1, 2).reshape(b, channels, spatial_dim, spatial_dim)
        return features, (spatial_dim, spatial_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs, _ = self.forward_with_backbone(x)
        return outputs

    def forward_with_backbone(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Tuple[int, int]]:
        """Run inference and return predictions, feature map, and feature resolution."""

        feature_map, feature_resolution = self.forward_backbone(x)
        predictions = {name: head(feature_map) for name, head in self.task_heads.items()}
        return predictions, feature_map, feature_resolution

    @property
    def tasks(self) -> Iterable[TaskSpec]:
        return self.task_specs.values()

    def set_task_specs(self, specs: Iterable[TaskSpec]) -> None:
        """Replace task heads with a new specification list."""

        specs = list(specs)
        self.task_specs = {spec.name: spec for spec in specs}
        embed_dim = next(iter(self.task_heads.values())).layers[0].in_channels
        self.task_heads = nn.ModuleDict(
            {spec.name: TaskHead(embed_dim, spec.output_channels) for spec in specs}
        )