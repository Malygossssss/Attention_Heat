"""Miscellaneous helper utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch

LOGGER = logging.getLogger(__name__)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path | str) -> Tuple[list[str], list[str]]:
    """Load weights into a model with helpful logging."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Missing parameters when loading %s: %s", checkpoint_path, missing)
    if unexpected:
        LOGGER.warning("Unexpected parameters when loading %s: %s", checkpoint_path, unexpected)

    LOGGER.info("Loaded checkpoint '%s'", checkpoint_path)
    return missing, unexpected