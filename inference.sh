#!/bin/bash


python scripts/batch_segment_images.py \
    --input-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/images \
    --prompt "cars" \
    --prompt "bags" \
    --prompt "vehicles shadows" \
    --prompt "pedestrians" \
    --output-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/dynamic_masks \
    --overlay-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/dynamic_overlays \
    --combine-prompts \
    --device cuda \
    --confidence 0.4



python scripts/batch_segment_images.py \
    --input-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/images \
    --prompt "sky" \
    --output-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/sky_masks \
    --overlay-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/sky_overlays \
    --combine-prompts \
    --device cuda \
    --confidence 0.4


python scripts/batch_segment_images.py \
    --input-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/images \
    --prompt "road" \
    --output-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/road_masks \
    --overlay-dir /home/xiangyu/Downloads/MCM_logs/sam3_test/road_overlays \
    --combine-prompts \
    --device cuda \
    --confidence 0.4