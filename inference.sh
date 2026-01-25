#!/bin/bash

input_dir=/home/xiangyu/Downloads/MCM_logs/scanmatch_viewer_log/processed_outputs_objects/images/image_lucid_fc_jpeg

python scripts/batch_segment_images_with_hints.py \
    --input-dir $input_dir \
    --prompt "vehicles" \
    --prompt "vehicles shadows" \
    --prompt "pedestrians" \
    --prompt "bicycles" \
    --prompt "motorcycles" \
    --prompt "trucks" \
    --prompt "people" \
    --output-dir $input_dir/dynamic_masks \
    --combine-prompts \
    --device cuda \
    --write-empty \
    --confidence 0.4 \
    --hint-centers-jsonl /home/xiangyu/Downloads/MCM_logs/scanmatch_viewer_log/processed_outputs_objects/keyframes_object_centers.jsonl \
    --overlay-dir $input_dir/dynamic_overlays



# python scripts/batch_segment_images.py \
#     --input-dir $input_dir/images \
#     --prompt "sky" \
#     --output-dir $input_dir/sky_masks \
#     --combine-prompts \
#     --device cuda \
#     --write-empty \
#     --confidence 0.4
#     # --overlay-dir $input_dir/sky_overlays \

# python scripts/batch_segment_images.py \
#     --input-dir $input_dir/images \
#     --prompt "road" \
#     --output-dir $input_dir/ground_masks \
#     --combine-prompts \
#     --device cuda \
#     --write-empty \
#     --confidence 0.4
#     # --overlay-dir $input_dir/ground_overlays \
