#!/usr/bin/env python3
"""Batch run SAM3 image segmentation over a folder of images."""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_EXTENSIONS: Sequence[str] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


def find_images(root: Path) -> List[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def save_instance_mask(masks: torch.Tensor, scores: torch.Tensor, output_path: Path):
    """Persist all instance masks into a single label image.

    Instances are encoded as integer IDs (1..N) with highest-score masks taking precedence on overlaps.
    Saved as 16-bit PNG to allow many instances.
    """
    # masks: [N, 1, H, W] bool
    if masks.ndim != 4:
        raise ValueError(f"Expected masks with shape [N,1,H,W], got {masks.shape}")
    num_instances, _, height, width = masks.shape
    device = masks.device
    id_map = torch.zeros((height, width), dtype=torch.int32, device=device)

    # Apply higher-score masks first so they win on overlaps
    order = torch.argsort(scores.to(device), descending=True)
    for rank, idx in enumerate(order, start=1):
        mask = masks[int(idx)].squeeze(0)
        id_map = torch.where(
            mask, torch.tensor(rank, dtype=id_map.dtype, device=device), id_map
        )

    mask_img = Image.fromarray(id_map.cpu().numpy().astype("uint16"), mode="I;16")
    mask_img.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a folder of images with SAM3."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing images to process (recurses).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sam3_batch_output"),
        help="Folder to write masks and summary JSON.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the concept to segment.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for filtering detections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    images = find_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading SAM3 on {args.device} (prompt='{args.prompt}')...")
    model = build_sam3_image_model(device=args.device)
    processor = Sam3Processor(
        model, device=args.device, confidence_threshold=args.confidence
    )

    summary = []
    for img_path in sorted(images):
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        state = processor.set_image(image, state={})
        state = processor.set_text_prompt(prompt=args.prompt, state=state)

        masks = state["masks"]
        boxes = state["boxes"]
        scores = state["scores"]

        if masks.numel() == 0:
            print(f"[skip] {img_path.name}: no detections above threshold")
            continue

        mask_filename = f"{img_path.stem}_instances.png"
        mask_path = args.output_dir / mask_filename
        save_instance_mask(masks, scores, mask_path)

        summary.append(
            {
                "source_image": str(img_path),
                "mask": mask_filename,
                "instances": len(masks),
                "scores": [float(s.cpu()) for s in scores],
                "boxes_xyxy": [[float(x) for x in b.cpu()] for b in boxes],
            }
        )

        print(f"[done] {img_path.name}: wrote {len(masks)} instances into {mask_filename}")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Finished. Summary written to {summary_path}")


if __name__ == "__main__":
    main()



#   python scripts/batch_segment_images.py \
#     --prompt "car shadow" \
#     --output-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/sam3_masks \
#     --overlay-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/sam3_overlays \
#     --combine-prompts \
#     --device cuda \
#     --confidence 0.4