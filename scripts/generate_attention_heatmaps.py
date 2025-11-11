"""Command line entry point for generating multi-task attention heatmaps."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch
from torchvision import transforms

from attention_heat import (
    MultiTaskSwin,
    SwinBackboneConfig,
    TaskSpec,
    compute_task_heatmaps,
    load_checkpoint,
    read_checkpoint,
    save_attention_heatmaps,
)
from attention_heat.visualization import load_image


LOGGER = logging.getLogger("attention_heat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", nargs="+", required=True, help="One or more checkpoint paths")
    parser.add_argument("--image", type=str, default="data/2008_000008.jpg", help="Input image path")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to store results")
    parser.add_argument("--input-size", type=int, default=224, help="Input resolution for the model")
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU inference even if CUDA is available"
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index to run on (e.g. --cuda-device 0). Ignored when --cpu is set.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.55, help="Overlay alpha value for heatmaps"
    )
    parser.add_argument(
        "--cmap", type=str, default="magma", help="Matplotlib colormap name for heatmaps"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        help="Swin backbone identifier or 'auto' to infer from the checkpoint",
    )
    parser.add_argument(
        "--pretrained-backbone", action="store_true", help="Initialise the backbone with ImageNet weights"
    )
    return parser.parse_args()


def build_transforms(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def default_task_specs() -> List[TaskSpec]:
    return [
        TaskSpec("semantic_segmentation", output_channels=21, activation="softmax"),
        TaskSpec("human_part_detection", output_channels=7, activation="softmax"),
        TaskSpec("surface_normals", output_channels=3, activation="tanh"),
        TaskSpec("saliency_distillation", output_channels=1, activation="sigmoid"),
    ]


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.cpu:
        device = torch.device("cpu")
    elif args.cuda_device is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but --cuda-device was specified")
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)
    LOGGER.info("Using device: %s", device)

    checkpoint_groups = []
    for ckpt_arg in args.ckpt:
        ckpt_path = Path(ckpt_arg)
        if ckpt_path.is_dir():
            LOGGER.info("Scanning directory %s for checkpoints", ckpt_path)
            checkpoint_files = sorted(
                p for p in ckpt_path.iterdir() if p.suffix in {".pt", ".pth", ".bin"}
            )
            if not checkpoint_files:
                LOGGER.warning("No checkpoint files found in %s", ckpt_path)
                continue
        else:
            checkpoint_files = [ckpt_path]

        checkpoint_groups.append((ckpt_path, checkpoint_files))

    checkpoint_files = [file for _, files in checkpoint_groups for file in files]
    if not checkpoint_files:
        raise RuntimeError("No checkpoints to process")

    preloaded_checkpoints = {}
    backbone_name: Optional[str]
    backbone_config: Optional[SwinBackboneConfig]

    if args.backbone.lower() == "auto":
        first_checkpoint = checkpoint_files[0]
        checkpoint_data = read_checkpoint(first_checkpoint)
        preloaded_checkpoints[first_checkpoint] = checkpoint_data
        backbone_config = SwinBackboneConfig.from_checkpoint(checkpoint_data)
        if backbone_config is None:
            backbone_name = "swin_tiny_patch4_window7_224"
            LOGGER.warning(
                "Unable to infer Swin backbone configuration from %s; defaulting to %s",
                first_checkpoint,
                backbone_name,
            )
        else:
            backbone_name = None
            LOGGER.info(
                "Inferred Swin backbone from %s (embed_dim=%d, depths=%s, num_heads=%s, window=%d)",
                first_checkpoint,
                backbone_config.embed_dim,
                backbone_config.depths,
                backbone_config.num_heads,
                backbone_config.window_size,
            )
    else:
        backbone_name = args.backbone
        backbone_config = None

    if backbone_config is not None and args.pretrained_backbone:
        raise ValueError(
            "--pretrained-backbone cannot be combined with automatic backbone inference"
        )

    specs = default_task_specs()
    model = MultiTaskSwin(
        specs,
        backbone=backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        backbone_config=backbone_config,
    )
    model.to(device)
    model.eval()

    transform = build_transforms(args.input_size)
    image = load_image(args.image)
    input_tensor = transform(image).unsqueeze(0).to(device)
    image_resolution = input_tensor.shape[-2:]

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for _, files in checkpoint_groups:
            for checkpoint_file in files:
                LOGGER.info("Processing checkpoint %s", checkpoint_file)
                checkpoint_data = preloaded_checkpoints.get(checkpoint_file)
                load_checkpoint(model, checkpoint_file, checkpoint_data=checkpoint_data)

                model.attention_recorder.clear()
                predictions, _, feature_resolution = model.forward_with_backbone(input_tensor)
                attention_map = model.attention_recorder.build_attention_map(
                    batch_size=input_tensor.size(0),
                    feature_resolution=feature_resolution,
                    image_resolution=image_resolution,
                )

                heatmaps = compute_task_heatmaps(
                    predictions,
                    attention_map,
                    specs,
                    image_resolution=image_resolution,
                )

                checkpoint_dir = output_root / checkpoint_file.stem
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                save_attention_heatmaps(
                    image,
                    {name: heatmap.cpu().numpy() for name, heatmap in heatmaps.items()},
                    checkpoint_dir,
                    alpha=args.alpha,
                    cmap=args.cmap,
                )


if __name__ == "__main__":
    main()