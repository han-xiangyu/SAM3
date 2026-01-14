#!/bin/bash

input_dir=/media/xiangyu/Ultra/2025-12-02_m20240018s2_ga-atlanta/processed_outputs_global/colmap_output_start1000_keyframes5000_all_cams


python scripts/batch_segment_images.py \
    --input-dir $input_dir/images \
    --prompt "vehicles" \
    --prompt "vehicles shadows" \
    --prompt "pedestrians" \
    --prompt "bicycles" \
    --prompt "motorcycles" \
    --prompt "trucks" \
    --prompt "people" \
    --output-dir $input_dir/dynamic_masks \
    --overlay-dir $input_dir/dynamic_overlays \
    --combine-prompts \
    --device cuda \
    --confidence 0.4



python scripts/batch_segment_images.py \
    --input-dir $input_dir/images \
    --prompt "sky" \
    --output-dir $input_dir/sky_masks \
    --overlay-dir $input_dir/sky_overlays \
    --combine-prompts \
    --device cuda \
    --confidence 0.4


python scripts/batch_segment_images.py \
    --input-dir $input_dir/images \
    --prompt "road" \
    --output-dir $input_dir/ground_masks \
    --overlay-dir $input_dir/ground_overlays \
    --combine-prompts \
    --device cuda \
    --confidence 0.4