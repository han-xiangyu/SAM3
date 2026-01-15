#!/usr/bin/env python3
"""Batch run SAM3 image segmentation over a folder of images.
Modified: write an empty mask when no objects are detected.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence

import torch
import numpy as np
from PIL import Image, ImageDraw

# Assuming these imports exist in your environment
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_EXTENSIONS: Sequence[str] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)


def find_images(root: Path) -> List[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def save_instance_mask(masks: torch.Tensor, scores: torch.Tensor, output_path: Path):
    """Persist masks into a single binary image (White=Mask, Black=Background)."""
    # masks: [N, 1, H, W]
    if masks.ndim != 4:
        raise ValueError(f"Expected masks with shape [N,1,H,W], got {masks.shape}")

    # 1. Collapse all N instances into a single boolean mask
    merged_mask = torch.any(masks, dim=0).squeeze(0)  # [H, W] boolean

    # 2. Convert to 8-bit grayscale (0 or 255)
    mask_uint8 = (merged_mask.cpu().numpy().astype(np.uint8) * 255)

    # 3. Save as 'L' PNG
    mask_img = Image.fromarray(mask_uint8, mode="L")
    mask_img.save(output_path)


def save_empty_mask_like_image(image: Image.Image, output_path: Path):
    """Save an all-zero (black) single-channel mask with same resolution as the input image."""
    w, h = image.size
    empty = np.zeros((h, w), dtype=np.uint8)  # [H, W], all zeros
    Image.fromarray(empty, mode="L").save(output_path)


def save_overlay(image: Image.Image, masks: torch.Tensor, output_path: Path):
    """Save a visualization of the masks overlaid on the image."""
    overlay_img = image.convert("RGBA")

    num_masks = masks.shape[0]

    mask_layer = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)

    for i in range(num_masks):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)

        m = masks[i].squeeze().cpu().numpy()  # [H, W]

        solid_color = Image.new("RGBA", overlay_img.size, color)
        mask_pil = Image.fromarray((m * 255).astype(np.uint8), mode="L")

        mask_layer.paste(solid_color, (0, 0), mask_pil)

    final = Image.alpha_composite(overlay_img, mask_layer)
    final.convert("RGB").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment a folder of images with SAM3.")

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
        action="append",
        required=True,
        help="Text prompt(s). Can use multiple --prompt flags.",
    )

    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Folder to write visualization overlays.",
    )

    parser.add_argument(
        "--combine-prompts",
        action="store_true",
        help="If set, combines detections from all prompts into one file per image.",
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

    parser.add_argument(
        "--write-empty",
        action="store_true",
        default=True,
        help="If set, writes an empty mask when no detections are found (default: True).",
    )
    parser.add_argument(
        "--no-write-empty",
        dest="write_empty",
        action="store_false",
        help="Disable writing empty masks when no detections are found.",
    )

    parser.add_argument(
        "--write-summary",
        action="store_true",
        default=False,
        help="If set, writes a summary JSON file with detection details.",
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
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 on {args.device}...")
    print(f"Prompts: {args.prompt}")

    model = build_sam3_image_model(device=args.device)
    processor = Sam3Processor(
        model, device=args.device, confidence_threshold=args.confidence
    )

    summary = []

    for img_path in sorted(images):
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        final_masks_list = []
        final_scores_list = []
        final_boxes_list = []

        for p in args.prompt:
            state = processor.set_image(image, state={})
            state = processor.set_text_prompt(prompt=p, state=state)

            m = state["masks"]   # [N, 1, H, W]
            b = state["boxes"]
            s = state["scores"]

            # Only append if there is at least one instance
            if m is not None and m.numel() > 0 and m.shape[0] > 0:
                final_masks_list.append(m)
                final_scores_list.append(s)
                final_boxes_list.append(b)

        mask_filename = f"{img_path.stem}.png"
        mask_path = args.output_dir / mask_filename

        # If we found nothing for ANY prompt -> write empty mask (instead of skipping)
        if not final_masks_list:
            print(f"[empty] {img_path.name}: no detections for {args.prompt} -> writing empty mask")

            if args.write_empty:
                save_empty_mask_like_image(image, mask_path)

                # Optional overlay: just save the original image (no masks) for consistency
                if args.overlay_dir:
                    overlay_filename = f"{img_path.stem}.jpg"
                    overlay_path = args.overlay_dir / overlay_filename
                    image.save(overlay_path)

                if args.write_summary:
                    summary.append(
                        {
                            "source_image": str(img_path),
                            "mask": mask_filename,
                            "instances": 0,
                            "scores": [],
                            "boxes_xyxy": [],
                            "empty": True,
                        }
                    )
            else:
                print(f"[skip] {img_path.name}: no detections (empty mask writing disabled)")
            continue

        # Concatenate results from all prompts
        final_masks = torch.cat(final_masks_list, dim=0)
        final_scores = torch.cat(final_scores_list, dim=0)
        final_boxes = torch.cat(final_boxes_list, dim=0)

        # 1. Save Mask
        save_instance_mask(final_masks, final_scores, mask_path)

        # 2. Save Overlay (if requested)
        if args.overlay_dir:
            overlay_filename = f"{img_path.stem}.jpg"
            overlay_path = args.overlay_dir / overlay_filename
            save_overlay(image, final_masks, overlay_path)

        # 3. Update Summary
        if args.write_summary:
            summary.append(
                {
                    "source_image": str(img_path),
                    "mask": mask_filename,
                    "instances": int(final_masks.shape[0]),
                    "scores": [float(x.cpu()) for x in final_scores],
                    "boxes_xyxy": [[float(x) for x in b.cpu()] for b in final_boxes],
                    "empty": False,
                }
            )

        print(f"[done] {img_path.name}: wrote {final_masks.shape[0]} instances (Prompts: {args.prompt})")
    
    # Write summary JSON if requested
    if args.write_summary:
        summary_path = args.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Finished. Summary written to {summary_path}")
    else:
        print("Finished.")


if __name__ == "__main__":
    main()
