#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Inputs (match train script naming)
MODEL_PRETRAIN_IN="meta-llama/Llama-2-7b-hf"
# If REGULARIZED not given, assume the train script’s merged output layout
MODEL_REGULARIZED_IN="${2:-"${ROOT_DIR}/models/regularized/$(basename "${MODEL_PRETRAIN_IN}")"}"

BASE_PRETRAIN="$(basename "${MODEL_PRETRAIN_IN}")"
BASE_REGULARIZED="$(basename "${MODEL_REGULARIZED_IN}")_REG"

echo $BASE_PRETRAIN
echo $BASE_REGULARIZED


# Common eval config
DATASET="wikitext2"
SEQ_LEN=2048
BATCH_SIZE=8
CALIB_SIZE=256
SEED=0
EVAL_TASKS="hellaswag,piqa,winogrande,boolq,arc_easy,arc_challenge"

# Results layout mirrors the train script idea of model-scoped folders:
# results/llm/<MODEL_BASENAME>/factorized_posttrain/{rank,params_auto}/results.json
RES_PRETRAIN_rank_DIR="${ROOT_DIR}/results/llm/${BASE_PRETRAIN}/factorized_posttrain/rank"
RES_REGULARIZED_rank_DIR="${ROOT_DIR}/results/llm/${BASE_REGULARIZED}/factorized_posttrain/rank"
RES_PRETRAIN_PARAMS_DIR="${ROOT_DIR}/results/llm/${BASE_PRETRAIN}/factorized_posttrain/params_auto"
RES_REGULARIZED_PARAMS_DIR="${ROOT_DIR}/results/llm/${BASE_REGULARIZED}/factorized_posttrain/params_auto"

mkdir -p "${RES_PRETRAIN_rank_DIR}" \
         "${RES_REGULARIZED_rank_DIR}" \
         "${RES_PRETRAIN_PARAMS_DIR}" \
         "${RES_REGULARIZED_PARAMS_DIR}"
: <<'EOF'
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name "${MODEL_REGULARIZED_IN}" \
  --results_dir "${RES_REGULARIZED_rank_DIR}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --calib_size "${CALIB_SIZE}" \
  --mode rank \
  --seed "${SEED}" \
  --eval_tasks "${EVAL_TASKS}"

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name "${MODEL_REGULARIZED_IN}" \
  --results_dir "${RES_REGULARIZED_PARAMS_DIR}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --calib_size "${CALIB_SIZE}" \
  --mode params_auto \
  --seed "${SEED}" \
  --eval_tasks "${EVAL_TASKS}"


python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name "${MODEL_PRETRAIN_IN}" \
  --results_dir "${RES_PRETRAIN_rank_DIR}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --calib_size "${CALIB_SIZE}" \
  --mode rank \
  --seed "${SEED}" \
  --eval_tasks "${EVAL_TASKS}"


EOF

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name "${MODEL_PRETRAIN_IN}" \
  --results_dir "${RES_PRETRAIN_PARAMS_DIR}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --calib_size "${CALIB_SIZE}" \
  --mode params_auto \
  --seed "${SEED}" \
  --eval_tasks "${EVAL_TASKS}"
# ---------------------
# Plots
# ---------------------
OUT_DIR="${ROOT_DIR}/results/llm/plots/${BASE_PRETRAIN}_vs_${BASE_REGULARIZED}"
mkdir -p "${OUT_DIR}"

PRETRAIN_RESULTS_rank="${RES_PRETRAIN_rank_DIR}/results.json"
REGULARIZED_RESULTS_rank="${RES_REGULARIZED_rank_DIR}/results.json"
PRETRAIN_RESULTS_params="${RES_PRETRAIN_PARAMS_DIR}/results.json"
REGULARIZED_RESULTS_params="${RES_REGULARIZED_PARAMS_DIR}/results.json"

python "${SCRIPT_DIR}/plot.py" \
  --runs "${PRETRAIN_RESULTS_rank}:${BASE_PRETRAIN} pretrain" "${REGULARIZED_RESULTS_rank}:${BASE_REGULARIZED} regularized" \
  --title "Compressibility with rank-SVD — ${BASE_PRETRAIN} vs ${BASE_REGULARIZED}" \
  --out_dir "${OUT_DIR}" \
  --name "rank"

python "${SCRIPT_DIR}/plot.py" \
  --runs "${PRETRAIN_RESULTS_params}:${BASE_PRETRAIN} pretrain" "${REGULARIZED_RESULTS_params}:${BASE_REGULARIZED} regularized" \
  --title "Compressibility with params_auto — ${BASE_PRETRAIN} vs ${BASE_REGULARIZED}" \
  --out_dir "${OUT_DIR}" \
  --name "params_auto"

