#!/usr/bin/env bash
set -e


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_IN="facebook/opt-1.3b"


MODEL_BASENAME="$(basename "${MODEL_IN}")"
MERGED_DIR="${ROOT_DIR}/models/regularized/${MODEL_BASENAME}"


ADAPTER_DIR="${ROOT_DIR}/models/regularized/${MODEL_BASENAME}_adapter"

mkdir -p "$(dirname "${MERGED_DIR}")"

python "${SCRIPT_DIR}/train_reg_llm.py" \
  --model_name "${MODEL_IN}" \
  --dataset_name "togethercomputer/RedPajama-Data-1T-Sample" \
  --dataset_config "" \
  --max_length 2048 \
  --output_dir "${ADAPTER_DIR}" \
  --merged_output_dir "${MERGED_DIR}" \
  --num_train_epochs 4 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.00 \
  --reg_lambda 0.0003 \
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
