#!/usr/bin/env python3
"""Batch run SAM3 image segmentation over a folder of images."""

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
    # If ANY instance covers a pixel, that pixel becomes True.
    merged_mask = torch.any(masks, dim=0).squeeze(0)  # Shape: [H, W] (boolean)

    # 2. Convert to standard 8-bit grayscale integer (0 or 255)
    # True becomes 255 (White), False becomes 0 (Black)
    mask_uint8 = (merged_mask.cpu().numpy().astype(np.uint8) * 255)

    # 3. Save as standard 'L' (Grayscale) PNG
    mask_img = Image.fromarray(mask_uint8, mode="L")
    mask_img.save(output_path)


def save_overlay(image: Image.Image, masks: torch.Tensor, output_path: Path):
    """Save a visualization of the masks overlaid on the image."""
    # Convert PIL image to RGBA for alpha blending
    overlay_img = image.convert("RGBA")
    
    # Generate random colors for each mask
    num_masks = masks.shape[0]
    
    # Create a transparent layer to draw masks onto
    mask_layer = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)
    
    for i in range(num_masks):
        # random color with 50% opacity (128)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)
        
        # masks[i] is [1, H, W], squeeze to [H, W]
        m = masks[i].squeeze().cpu().numpy()
        
        # Create a bitmap from the boolean mask
        # We can draw the bitmap or convert to polygon. 
        # Fast way for PIL: Create an image from the mask and color it.
        
        # Create a solid color image
        solid_color = Image.new("RGBA", overlay_img.size, color)
        # Create the mask as a L (grayscale) image
        mask_pil = Image.fromarray((m * 255).astype(np.uint8), mode='L')
        
        # Paste the solid color using the mask
        mask_layer.paste(solid_color, (0, 0), mask_pil)

    # Alpha composite the original image and the mask layer
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
    
    # CHANGED: action='append' allows multiple --prompt flags
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text prompt(s). Can use multiple --prompt flags.",
    )
    
    # ADDED: Overlay directory
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Folder to write visualization overlays.",
    )
    
    # ADDED: Combine prompts flag
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    images = find_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found under {args.input_dir}")

    # Ensure output directories exist
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

        # Initialize storage for this image
        final_masks_list = []
        final_scores_list = []
        final_boxes_list = []

        # Iterate over all provided prompts (e.g., "cars", "bags")
        for p in args.prompt:
            # We reset state or pass fresh state if we want independent detections per prompt
            # Usually for SAM-style models, separate prompts mean separate queries.
            state = processor.set_image(image, state={})
            state = processor.set_text_prompt(prompt=p, state=state)

            m = state["masks"]  # [N, 1, H, W]
            b = state["boxes"]
            s = state["scores"]
            
            # Filter low confidence locally if model doesn't do it strictly enough, 
            # though processor usually handles it via init threshold.
            if m.numel() > 0:
                final_masks_list.append(m)
                final_scores_list.append(s)
                final_boxes_list.append(b)

        # If we found nothing for ANY prompt
        if not final_masks_list:
            print(f"[skip] {img_path.name}: no detections for {args.prompt}")
            continue

        # Concatenate results from all prompts
        final_masks = torch.cat(final_masks_list, dim=0)
        final_scores = torch.cat(final_scores_list, dim=0)
        final_boxes = torch.cat(final_boxes_list, dim=0)

        # 1. Save Mask
        mask_filename = f"{img_path.stem}.png"
        mask_path = args.output_dir / mask_filename
        save_instance_mask(final_masks, final_scores, mask_path)

        # 2. Save Overlay (if requested)
        if args.overlay_dir:
            overlay_filename = f"{img_path.stem}.jpg"
            overlay_path = args.overlay_dir / overlay_filename
            save_overlay(image, final_masks, overlay_path)

        # 3. Update Summary
        summary.append(
            {
                "source_image": str(img_path),
                "mask": mask_filename,
                "instances": len(final_masks),
                "scores": [float(s.cpu()) for s in final_scores],
                # Boxes might need flattening depending on shape
                "boxes_xyxy": [[float(x) for x in b.cpu()] for b in final_boxes],
            }
        )

        print(f"[done] {img_path.name}: wrote {len(final_masks)} instances (Prompts: {args.prompt})")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Finished. Summary written to {summary_path}")


if __name__ == "__main__":
    main()