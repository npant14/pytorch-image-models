#!/bin/bash

set -euo pipefail

BASE_DIR="results/hangul_results_dprime"
OUTPUT_DIR="results/hangul_postprocess"
NUM_SAMPLES=1000
SPLIT_SEED=1

mkdir -p "${OUTPUT_DIR}"

python hangul/e4_korean_postprocess.py \
  --base "${BASE_DIR}" \
  --model vit_base \
  --layer blocks.7.norm2 \
  --num-samples "${NUM_SAMPLES}" \
  --split-seed "${SPLIT_SEED}" \
  --output "${OUTPUT_DIR}/vit_base_blocks.7.norm2.json"

python hangul/e4_korean_postprocess.py \
  --base "${BASE_DIR}" \
  --model resnet18_timm \
  --layer layer3.1.conv1 \
  --num-samples "${NUM_SAMPLES}" \
  --split-seed "${SPLIT_SEED}" \
  --output "${OUTPUT_DIR}/resnet18_timm_layer3.1.conv1.json"

python hangul/e4_korean_postprocess.py \
  --base "${BASE_DIR}" \
  --model alexnet_timm \
  --layer features.5 \
  --num-samples "${NUM_SAMPLES}" \
  --split-seed "${SPLIT_SEED}" \
  --output "${OUTPUT_DIR}/alexnet_timm_features.5.json"

python hangul/e4_korean_postprocess.py \
  --base "${BASE_DIR}" \
  --model hmax_v3_adj \
  --layer model_backbone.s2 \
  --num-samples "${NUM_SAMPLES}" \
  --split-seed "${SPLIT_SEED}" \
  --output "${OUTPUT_DIR}/hmax_v3_adj_model_backbone.s2.json"

echo "Wrote postprocessed results to ${OUTPUT_DIR}"
