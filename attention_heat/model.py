"""Model utilities for running multi-task inference with Swin Transformer backbones."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn

try:
    import timm
    from timm.models.swin_transformer import SwinTransformer
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
        backbone: Optional[str] = "swin_tiny_patch4_window7_224",
        pretrained_backbone: bool = False,
        backbone_config: Optional[SwinBackboneConfig] = None,
    ) -> None:
        super().__init__()

        if backbone_config is not None:
            if pretrained_backbone:
                raise ValueError("Cannot combine pretrained_backbone with a custom backbone_config")
            self.backbone = SwinTransformer(**backbone_config.to_kwargs())
        else:
            if backbone is None:
                backbone = "swin_tiny_patch4_window7_224"
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
@dataclass(frozen=True)
class SwinBackboneConfig:
    """Configuration describing a Swin Transformer backbone."""

    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    ape: bool = False
    patch_norm: bool = True
    drop_path_rate: float = 0.0
    use_checkpoint: bool = False
    pretrained_window_sizes: Optional[Tuple[int, ...]] = None

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert the configuration into keyword arguments for ``SwinTransformer``."""

        kwargs: Dict[str, Any] = {
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "drop_path_rate": self.drop_path_rate,
            "ape": self.ape,
            "patch_norm": self.patch_norm,
            "use_checkpoint": self.use_checkpoint,
        }
        if self.qk_scale is not None:
            kwargs["qk_scale"] = self.qk_scale
        if self.pretrained_window_sizes is not None:
            kwargs["pretrained_window_sizes"] = self.pretrained_window_sizes
        return kwargs

    @staticmethod
    def _get_nested(config: Any, *keys: str) -> Any:
        current = config
        for key in keys:
            if current is None:
                return None
            if isinstance(current, Mapping):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> Optional["SwinBackboneConfig"]:
        """Infer a configuration from checkpoint metadata or its state dict."""

        for key in ("config", "cfg", "CONFIG"):
            cfg = checkpoint.get(key)
            if cfg is None:
                continue
            swin_cfg = cls._get_nested(cfg, "MODEL", "SWIN")
            if swin_cfg is None:
                continue
            depths = tuple(int(x) for x in cls._get_nested(swin_cfg, "DEPTHS") or [])
            num_heads = tuple(int(x) for x in cls._get_nested(swin_cfg, "NUM_HEADS") or [])
            if not depths or not num_heads:
                continue
            window_size = cls._get_nested(swin_cfg, "WINDOW_SIZE")
            mlp_ratio = cls._get_nested(swin_cfg, "MLP_RATIO")
            pretrained_window_sizes = cls._get_nested(swin_cfg, "PRETRAINED_WINDOW_SIZES")
            if pretrained_window_sizes is not None:
                pretrained_window_sizes = tuple(int(x) for x in pretrained_window_sizes)
            patch_norm_value = cls._get_nested(swin_cfg, "PATCH_NORM")
            return cls(
                patch_size=int(cls._get_nested(swin_cfg, "PATCH_SIZE") or 4),
                in_chans=int(cls._get_nested(swin_cfg, "IN_CHANS") or 3),
                embed_dim=int(cls._get_nested(swin_cfg, "EMBED_DIM") or 96),
                depths=depths,
                num_heads=num_heads,
                window_size=int(window_size or 7),
                mlp_ratio=float(mlp_ratio or 4.0),
                qkv_bias=bool(cls._get_nested(swin_cfg, "QKV_BIAS") if cls._get_nested(swin_cfg, "QKV_BIAS") is not None else True),
                qk_scale=cls._get_nested(swin_cfg, "QK_SCALE"),
                ape=bool(cls._get_nested(swin_cfg, "APE") or False),
                patch_norm=bool(patch_norm_value if patch_norm_value is not None else True),
                drop_path_rate=float(cls._get_nested(swin_cfg, "DROP_PATH_RATE") or 0.0),
                use_checkpoint=bool(cls._get_nested(swin_cfg, "USE_CHECKPOINT") or False),
                pretrained_window_sizes=pretrained_window_sizes,
            )

        state_dict = checkpoint.get("model", checkpoint)
        return cls.from_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any]) -> Optional["SwinBackboneConfig"]:
        """Infer the configuration by inspecting a checkpoint's state dict."""

        if not state_dict:
            return None

        keys = list(state_dict.keys())
        prefix = "backbone." if any(k.startswith("backbone.") for k in keys) else ""

        def get_tensor(name: str):
            return state_dict.get(prefix + name)

        patch_weight = get_tensor("patch_embed.proj.weight")
        if patch_weight is None:
            return None

        patch_size = int(patch_weight.shape[-1])
        in_chans = int(patch_weight.shape[1])
        embed_dim = int(patch_weight.shape[0])

        depths = []
        num_heads = []
        window_size: Optional[int] = None
        mlp_ratio: Optional[float] = None

        for layer_idx in range(4):
            block_prefix = f"{prefix}layers.{layer_idx}.blocks."
            block_ids = {
                int(key[len(block_prefix):].split(".")[0])
                for key in keys
                if key.startswith(block_prefix)
            }
            if not block_ids:
                break

            depth = max(block_ids) + 1
            depths.append(depth)

            bias_table = get_tensor(f"layers.{layer_idx}.blocks.0.attn.relative_position_bias_table")
            if bias_table is not None:
                num_heads.append(int(bias_table.shape[1]))
                if window_size is None:
                    table_len = int(bias_table.shape[0])
                    win = int(round((table_len ** 0.5 + 1) / 2))
                    window_size = max(win, 1)

            fc1_weight = get_tensor(f"layers.{layer_idx}.blocks.0.mlp.fc1.weight")
            if fc1_weight is not None and mlp_ratio is None:
                mlp_ratio = float(fc1_weight.shape[0] / fc1_weight.shape[1])

        if not depths:
            return None

        if len(num_heads) < len(depths):
            # Fallback to the canonical Swin head schedule.
            num_heads = [embed_dim // 32 * (2 ** i) for i in range(len(depths))]

        qkv_bias = get_tensor("layers.0.blocks.0.attn.qkv.bias") is not None
        ape = get_tensor("absolute_pos_embed") is not None
        patch_norm = get_tensor("patch_embed.norm.weight") is not None

        return cls(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=tuple(depths),
            num_heads=tuple(int(h) for h in num_heads),
            window_size=window_size or 7,
            mlp_ratio=mlp_ratio or 4.0,
            qkv_bias=qkv_bias,
            ape=ape,
            patch_norm=patch_norm,
        )