#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"



: << a
python "${SCRIPT_DIR}/train_vit.py" \
  --model_name vit_tiny_patch16_224 \
  --save_path "${ROOT_DIR}/models/vit_tiny_cifar10_pretrain.pt" \
  --seed 0
a

# regularized
python "${SCRIPT_DIR}/train_reg_vit.py" \
  --model_name vit_tiny_patch16_224 \
  --save_path "${ROOT_DIR}/models/vit_tiny_cifar10_regularized.pt" \
  --pretrained_path "${ROOT_DIR}/models/vit_tiny_cifar10_pretrain.pt" \
  --seed 0 \
  --epochs 100 \
  --reg_lambda 0.001