"""Visualization helpers for attention heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

try:
    from matplotlib import cm
except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError(
        "matplotlib is required for colormap generation. Install it with `pip install matplotlib`."
    ) from exc


def _apply_colormap(heatmap: np.ndarray, cmap: str = "magma") -> np.ndarray:
    cmap_fn = cm.get_cmap(cmap)
    colored = cmap_fn(heatmap)
    rgb = (colored[..., :3] * 255).astype(np.uint8)
    return rgb


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, cmap: str = "magma") -> Image.Image:
    """Overlay a heatmap on top of an RGB image."""

    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be a 2D array")

    normalized = np.clip(heatmap, 0.0, 1.0)
    heatmap_rgb = _apply_colormap(normalized, cmap)
    heatmap_img = Image.fromarray(heatmap_rgb).convert("RGBA")
    image_rgba = image.convert("RGBA")
    blended = Image.blend(image_rgba, heatmap_img, alpha=alpha)
    return blended.convert("RGB")


def save_attention_heatmaps(
    image: Image.Image,
    heatmaps: Dict[str, np.ndarray],
    output_dir: Path | str,
    prefix: str = "",
    alpha: float = 0.55,
    cmap: str = "magma",
) -> None:
    """Persist heatmaps and their overlays to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, heatmap in heatmaps.items():
        if heatmap.ndim == 3:
            if heatmap.shape[0] != 1:
                raise ValueError("Batch saving is not supported; provide a single heatmap per task")
            heatmap_2d = heatmap[0]
        else:
            heatmap_2d = heatmap

        overlay = overlay_heatmap(image, heatmap_2d, alpha=alpha, cmap=cmap)
        suffix = f"_{name}" if not prefix else f"_{name}_{prefix}"
        overlay_path = output_dir / f"attention_overlay{suffix}.png"
        raw_path = output_dir / f"attention_map{suffix}.npy"

        overlay.save(overlay_path)
        np.save(raw_path, heatmap_2d)


def load_image(path: Path | str) -> Image.Image:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")