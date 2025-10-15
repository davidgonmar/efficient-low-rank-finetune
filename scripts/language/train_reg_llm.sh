#!/usr/bin/env bash
set -e


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_IN="${1:-TinyLlama/TinyLlama_v1.1}"


MODEL_BASENAME="$(basename "${MODEL_IN}")"
MERGED_DIR="${ROOT_DIR}/models/regularized/${MODEL_BASENAME}"


ADAPTER_DIR="${ROOT_DIR}/models/regularized/${MODEL_BASENAME}_adapter"

mkdir -p "$(dirname "${MERGED_DIR}")"

python "${SCRIPT_DIR}/train_reg_llm.py" \
  --model_name "${MODEL_IN}" \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --max_length 2048 \
  --output_dir "${ADAPTER_DIR}" \
  --merged_output_dir "${MERGED_DIR}" \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --reg_lambda 1e-1 \
  --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --save_steps 200 \
  --save_total_limit 1 \
  --logging_steps 50 \
  --bf16

echo ""
echo "Regularized FINAL model saved to:"
echo "  ${MERGED_DIR}"
echo ""
echo "Use this in your compression script:"
echo "  MODEL_NAME_REGULARIZED=\"${MERGED_DIR}\""
