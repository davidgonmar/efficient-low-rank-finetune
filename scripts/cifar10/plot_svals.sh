#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TIMM_FUSED_ATTN=0

OUT_DIR="${ROOT_DIR}/results/vit_tiny_patch16_224/svals_orig_vs_reg"

python "${SCRIPT_DIR}/plot_svals.py" \
  --model_name vit_tiny_patch16_224 \
  --pretrained_orig "${ROOT_DIR}/models/vit_tiny_cifar10_pretrain.pt" \
  --pretrained_reg "${ROOT_DIR}/models/vit_tiny_cifar10_regularized.pt" \
  --results_dir "${OUT_DIR}" \
  --seed 0 \