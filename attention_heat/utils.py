"""Miscellaneous helper utilities."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any, Tuple

import torch

LOGGER = logging.getLogger(__name__)


def _ensure_yacs_stub() -> None:
    """Provide a minimal ``yacs.config.CfgNode`` implementation when absent."""

    if importlib.util.find_spec("yacs.config") is not None:
        return

    cfg_module = types.ModuleType("yacs.config")

    class CfgNode(dict):
        """Lightweight stand-in compatible with pickled ``CfgNode`` objects."""

        __slots__ = ()

        def __getattr__(self, key: str):
            try:
                return self[key]
            except KeyError as exc:  # pragma: no cover - defensive
                raise AttributeError(key) from exc

        def __setattr__(self, key: str, value) -> None:
            self[key] = value

        def __delattr__(self, key: str) -> None:
            del self[key]

        def clone(self):
            return type(self)(self)

    cfg_module.CfgNode = CfgNode  # type: ignore[attr-defined]

    yacs_module = types.ModuleType("yacs")
    yacs_module.config = cfg_module  # type: ignore[attr-defined]

    sys.modules.setdefault("yacs", yacs_module)
    sys.modules.setdefault("yacs.config", cfg_module)
    LOGGER.debug("Registered fallback yacs.config.CfgNode stub for checkpoint loading")


def read_checkpoint(checkpoint_path: Path | str) -> Any:
    """Load a checkpoint from disk while ensuring required stubs are available."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # PyTorch 2.6 changed the default ``weights_only`` argument of ``torch.load`` to
    # ``True`` which prevents loading checkpoints containing objects such as
    # ``yacs.config.CfgNode`` (used by our models). Explicitly setting
    # ``weights_only=False`` restores the previous behaviour while keeping the
    # rest of the loading logic unchanged.
    _ensure_yacs_stub()
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    *,
    checkpoint_data: Any | None = None,
) -> Tuple[list[str], list[str]]:
    """Load weights into a model with helpful logging."""

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_data is None:
        checkpoint = read_checkpoint(checkpoint_path)
    else:
        checkpoint = checkpoint_data
    state_dict = checkpoint.get("model", checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Missing parameters when loading %s: %s", checkpoint_path, missing)
    if unexpected:
        LOGGER.warning("Unexpected parameters when loading %s: %s", checkpoint_path, unexpected)

    LOGGER.info("Loaded checkpoint '%s'", checkpoint_path)
    return missing, unexpected