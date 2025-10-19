#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./compress.sh [PRETRAIN_MODEL] [REGULARIZED_MODEL]
# Defaults:
#   PRETRAIN_MODEL    = TinyLlama/TinyLlama_v1.1
#   REGULARIZED_MODEL = $ROOT_DIR/models/regularized/$(basename PRETRAIN_MODEL)
#
# Examples:
#   ./compress.sh
#   ./compress.sh TinyLlama/TinyLlama_v1.1
#   ./compress.sh meta-llama/Llama-3.1-8B $PWD/models/regularized/Llama-3.1-8B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Inputs (match train script naming)
MODEL_PRETRAIN_IN="${1:-TinyLlama/TinyLlama_v1.1}"
# If REGULARIZED not given, assume the train script’s merged output layout
MODEL_REGULARIZED_IN="${2:-"${ROOT_DIR}/models/regularized/$(basename "${MODEL_PRETRAIN_IN}")"}"

BASE_PRETRAIN="$(basename "${MODEL_PRETRAIN_IN}")"
BASE_REGULARIZED="$(basename "${MODEL_REGULARIZED_IN}")_REG"

echo $BASE_PRETRAIN
echo $BASE_REGULARIZED


# Common eval config
DATASET="ptb"
SEQ_LEN=2048
BATCH_SIZE=4
CALIB_SIZE=128
SEED=0

# Results layout mirrors the train script idea of model-scoped folders:
# results/llm/<MODEL_BASENAME>/factorized_posttrain/{energy,params_auto}/results.json
RES_PRETRAIN_ENERGY_DIR="${ROOT_DIR}/results/llm/${BASE_PRETRAIN}/factorized_posttrain/energy"
RES_REGULARIZED_ENERGY_DIR="${ROOT_DIR}/results/llm/${BASE_REGULARIZED}/factorized_posttrain/energy"
RES_PRETRAIN_PARAMS_DIR="${ROOT_DIR}/results/llm/${BASE_PRETRAIN}/factorized_posttrain/params_auto"
RES_REGULARIZED_PARAMS_DIR="${ROOT_DIR}/results/llm/${BASE_REGULARIZED}/factorized_posttrain/params_auto"

mkdir -p "${RES_PRETRAIN_ENERGY_DIR}" \
         "${RES_REGULARIZED_ENERGY_DIR}" \
         "${RES_PRETRAIN_PARAMS_DIR}" \
         "${RES_REGULARIZED_PARAMS_DIR}"




# ---------------------
# Energy-SVD sweeps
# ---------------------


python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name "${MODEL_REGULARIZED_IN}" \
  --results_dir "${RES_REGULARIZED_ENERGY_DIR}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --calib_size "${CALIB_SIZE}" \
  --mode energy \
  --seed "${SEED}"

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name "${MODEL_PRETRAIN_IN}" \
  --results_dir "${RES_PRETRAIN_ENERGY_DIR}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --calib_size "${CALIB_SIZE}" \
  --mode energy \
  --seed "${SEED}"

# ---------------------
# Plots
# ---------------------
OUT_DIR="${ROOT_DIR}/results/llm/plots/${BASE_PRETRAIN}_vs_${BASE_REGULARIZED}"
mkdir -p "${OUT_DIR}"

PRETRAIN_RESULTS_ENERGY="${RES_PRETRAIN_ENERGY_DIR}/results.json"
REGULARIZED_RESULTS_ENERGY="${RES_REGULARIZED_ENERGY_DIR}/results.json"

python "${SCRIPT_DIR}/plot.py" \
  --runs "${PRETRAIN_RESULTS_ENERGY}:${BASE_PRETRAIN} pretrain" "${REGULARIZED_RESULTS_ENERGY}:${BASE_REGULARIZED} regularized" \
  --title "Compressibility with Energy-SVD — ${BASE_PRETRAIN} vs ${BASE_REGULARIZED}" \
  --out_dir "${OUT_DIR}" \
  --name "energy"

PRETRAIN_RESULTS_PARAMS="${RES_PRETRAIN_PARAMS_DIR}/results.json"
REGULARIZED_RESULTS_PARAMS="${RES_REGULARIZED_PARAMS_DIR}/results.json"

python "${SCRIPT_DIR}/plot.py" \
  --runs "${PRETRAIN_RESULTS_PARAMS}:${BASE_PRETRAIN} pretrain" "${REGULARIZED_RESULTS_PARAMS}:${BASE_REGULARIZED} regularized" \
  --title "Compressibility with BALF — ${BASE_PRETRAIN} vs ${BASE_REGULARIZED}" \
  --out_dir "${OUT_DIR}" \
  --name "params_auto"

echo ""
echo "Compression completed."
echo "Pretrain model:     ${MODEL_PRETRAIN_IN} -> ${BASE_PRETRAIN}"
echo "Regularized model:  ${MODEL_REGULARIZED_IN} -> ${BASE_REGULARIZED}"
echo "Energy results:"
echo "  ${PRETRAIN_RESULTS_ENERGY}"
echo "  ${REGULARIZED_RESULTS_ENERGY}"
echo "Params-auto results:"
echo "  ${PRETRAIN_RESULTS_PARAMS}"
echo "  ${REGULARIZED_RESULTS_PARAMS}"
echo "Plots saved to:"
echo "  ${OUT_DIR}"
echo ""


