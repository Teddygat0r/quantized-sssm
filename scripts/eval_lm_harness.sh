#!/usr/bin/env bash
set -euo pipefail

# Simple lm-eval-harness runner for common zero-shot multiple-choice tasks.
# Usage:
#   bash scripts/eval_lm_harness.sh "AntonV/mamba2-130m-hf"
# Optional env vars:
#   DEVICE=cuda|cpu
#   BATCH_SIZE=auto|<int>
#   FEWSHOT=<int>
#   OUTPUT_DIR=results/lm_eval
#   MODEL_ARGS_EXTRA="<comma-separated key=value pairs>"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_lm_harness.sh <model_name_or_path> [model_args_extra]"
  echo
  echo "Example:"
  echo "  bash scripts/eval_lm_harness.sh AntonV/mamba2-130m-hf"
  echo "  bash scripts/eval_lm_harness.sh AntonV/mamba2-130m-hf dtype=bfloat16"
  exit 1
fi

MODEL_NAME_OR_PATH="$1"
shift || true
MODEL_ARGS_EXTRA="${1:-${MODEL_ARGS_EXTRA:-}}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
FEWSHOT="${FEWSHOT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/lm_eval}"
TASKS="hellaswag,piqa,arc_challenge,winogrande"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_ARGS="pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True"

if [[ -n "${MODEL_ARGS_EXTRA}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${MODEL_ARGS_EXTRA}"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_DIR}/$(basename "${MODEL_NAME_OR_PATH}")_${TIMESTAMP}.json"

echo "Running lm-eval-harness benchmark..."
echo "  model: ${MODEL_NAME_OR_PATH}"
echo "  tasks: ${TASKS}"
echo "  device: ${DEVICE}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  num_fewshot: ${FEWSHOT}"
echo "  output: ${OUTPUT_PATH}"

lm_eval \
  --model hf \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_fewshot "${FEWSHOT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples

echo "Done. Results saved to ${OUTPUT_PATH}"
