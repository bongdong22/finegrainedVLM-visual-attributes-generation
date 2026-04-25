#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash scripts/run_visual_attributes_split_components.sh \
#   /data/my_dataset/images \
#   /data/my_dataset/masks \
#   /data/my_dataset/visual_attributes_output \
#   tumor
#
# Or, for multiple object names with explicit mask-object mapping:
# bash scripts/run_visual_attributes_split_components.sh \
#   /data/my_dataset/images \
#   /data/my_dataset/masks \
#   /data/my_dataset/visual_attributes_output \
#   /data/my_dataset/object_map.csv

IMAGES_DIR="${1:-}"
MASKS_DIR="${2:-}"
OUTPUT_DIR="${3:-}"
OBJECT_SPEC="${4:-}"
CALIBRATION_MASKS_DIR="${5:-$MASKS_DIR}"

if [[ -z "$IMAGES_DIR" || -z "$MASKS_DIR" || -z "$OUTPUT_DIR" || -z "$OBJECT_SPEC" ]]; then
  echo "Usage: bash scripts/run_visual_attributes_split_components.sh <images_dir> <masks_dir> <output_dir> <object_name_or_object_map_csv> [calibration_masks_dir]"
  exit 1
fi

if [[ -f "$OBJECT_SPEC" ]]; then
  PYTHONPATH=src python -m fgvlm.visual_attributes \
    --images-dir "$IMAGES_DIR" \
    --masks-dir "$MASKS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --object-map-csv "$OBJECT_SPEC" \
    --calibration-masks-dir "$CALIBRATION_MASKS_DIR" \
    --image-glob "*.png" \
    --mask-glob "*.png" \
    --split-components
else
  PYTHONPATH=src python -m fgvlm.visual_attributes \
    --images-dir "$IMAGES_DIR" \
    --masks-dir "$MASKS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --object-name "$OBJECT_SPEC" \
    --calibration-masks-dir "$CALIBRATION_MASKS_DIR" \
    --image-glob "*.png" \
    --mask-glob "*.png" \
    --split-components
fi
