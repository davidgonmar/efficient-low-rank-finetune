#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TIMM_FUSED_ATTN=0 # so FLOPs are counted correctly


: << a
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name vit_tiny_patch16_224 \
  --pretrained_path "${ROOT_DIR}/models/vit_tiny_cifar10_pretrain.pt" \
  --results_dir "${ROOT_DIR}/results/vit_tiny_patch16_224/factorized_posttrain_pretrain/energy" \
  --mode energy \
  --seed 0



# flops auto
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name vit_tiny_patch16_224 \
  --pretrained_path "${ROOT_DIR}/models/vit_tiny_cifar10_pretrain.pt" \
  --results_dir "${ROOT_DIR}/results/vit_tiny_patch16_224/factorized_posttrain_pretrain/flops_auto" \
  --mode flops_auto \
  --seed 0
a

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name vit_tiny_patch16_224 \
  --pretrained_path "${ROOT_DIR}/models/vit_tiny_cifar10_regularized.pt" \
  --results_dir "${ROOT_DIR}/results/vit_tiny_patch16_224/factorized_posttrain_regularized/flops_auto" \
  --mode flops_auto \
  --seed 0

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name vit_tiny_patch16_224 \
  --pretrained_path "${ROOT_DIR}/models/vit_tiny_cifar10_regularized.pt" \
  --results_dir "${ROOT_DIR}/results/vit_tiny_patch16_224/factorized_posttrain_regularized/energy" \
  --mode energy \
  --seed 0


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PRETRAIN_RESULTS="${ROOT_DIR}/results/vit_tiny_patch16_224/factorized_posttrain_pretrain/energy/results.json"
REGULARIZED_RESULTS="${ROOT_DIR}/results/vit_tiny_patch16_224/factorized_posttrain_regularized/energy/results.json"
OUT_DIR="${ROOT_DIR}/results/vit_tiny_patch16_224/plots"

python "${SCRIPT_DIR}/plot.py" \
  --runs "${PRETRAIN_RESULTS}:ViT-Tiny pretrain" "${REGULARIZED_RESULTS}:ViT-Tiny regularized" \
  --title "ViT-Tiny compressibility with Energy-SVD" \
  --out_dir "${OUT_DIR}" \
  --name "energy"

PRETRAIN_RESULTS="${ROOT_DIR}/results/vit_tiny_patch16_224/factorized_posttrain_pretrain/flops_auto/results.json"
REGULARIZED_RESULTS="${ROOT_DIR}/results/vit_tiny_patch16_224/factorized_posttrain_regularized/flops_auto/results.json"
OUT_DIR="${ROOT_DIR}/results/vit_tiny_patch16_224/plots"

python "${SCRIPT_DIR}/plot.py" \
  --runs "${PRETRAIN_RESULTS}:ViT-Tiny pretrain" "${REGULARIZED_RESULTS}:ViT-Tiny regularized" \
  --title "ViT-Tiny compressibility with BALF" \
  --out_dir "${OUT_DIR}" \
  --name "flops_auto"


