"""Utility package for generating attention heatmaps from Swin models."""

from .model import MultiTaskSwin, TaskSpec
from .attention import (
    LastLayerAttentionRecorder,
    compute_task_heatmaps,
    normalize_map,
)
from .visualization import save_attention_heatmaps
from .utils import load_checkpoint

__all__ = [
    "MultiTaskSwin",
    "TaskSpec",
    "LastLayerAttentionRecorder",
    "compute_task_heatmaps",
    "normalize_map",
    "save_attention_heatmaps",
    "load_checkpoint",
]