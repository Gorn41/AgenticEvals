#!/usr/bin/env bash
set -euo pipefail

# Activate Python virtual environment
if [ ! -f ".venv/bin/activate" ]; then
  echo "Error: Python venv not found at .venv/. Create it and install dependencies." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Prefer local axlearn (repo) over installed site-packages
export PYTHONPATH="$(pwd)/axlearn:${PYTHONPATH:-}"

# Load environment variables from .env if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set -a
fi

# Ensure required variables are set
: "${VERTEX_AI_PROJECT:?Set VERTEX_AI_PROJECT in environment or .env}"
: "${VERTEX_AI_LOCATION:?Set VERTEX_AI_LOCATION in environment or .env}"
: "${GRADER_OPENAI_API_KEY:?Set GRADER_OPENAI_API_KEY in environment or .env}"

export VERTEX_AI_PROJECT
export VERTEX_AI_LOCATION
export GRADER_OPENAI_API_KEY

MODEL_NAME=${MODEL_NAME:-"gemini-2.0-flash"}
CLIENT_NAME=${CLIENT_NAME:-"gemini"}
IN_DIR=${IN_DIR:-"MMAU_datasets"}
CONCURRENCY=${CONCURRENCY:-1}
RETRY_SLEEP=${RETRY_SLEEP:-10}
DECODE_PARAMETERS=${DECODE_PARAMETERS:-'{"max_tokens":256,"temperature":0}'}

# Simple CLI parsing: --model <name> or --model=<name>
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      shift
      MODEL_NAME=${1:-$MODEL_NAME}
      ;;
    --model=*)
      MODEL_NAME="${1#*=}"
      ;;
    --client)
      shift
      CLIENT_NAME=${1:-$CLIENT_NAME}
      ;;
    --client=*)
      CLIENT_NAME="${1#*=}"
      ;;
    --in_dir)
      shift
      IN_DIR=${1:-$IN_DIR}
      ;;
    --in_dir=*)
      IN_DIR="${1#*=}"
      ;;
    *)
      echo "Unknown option: $1" >&2
      ;;
  esac
  shift || true
done

if [ ! -d "${IN_DIR}" ]; then
  echo "Error: Input directory not found: ${IN_DIR}" >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="outputs/run_${TS}"
mkdir -p "${OUT_ROOT}"

shopt -s nullglob
for REQUESTS_FILE in "${IN_DIR}"/*.jsonl; do
  FILENAME=$(basename "${REQUESTS_FILE}")

  # Determine metric by filename pattern first
  METRIC_NAME=""
  case "${FILENAME}" in
    *tool_use_execution* | *tool_use_self_correct* ) METRIC_NAME="tool_use_execution" ;;
    *tool_use_plan* ) METRIC_NAME="tool_use_plan" ;;
    *code_contests*understand* ) METRIC_NAME="code_contests_understand" ;;
    *code_contests*plan* ) METRIC_NAME="code_contests_plan" ;;
    *code_contests*regular* | *code_contests*solve* | *code_contests*.jsonl ) METRIC_NAME="code_contests" ;;
    *math* ) METRIC_NAME="math" ;;
    *code_kaggle* ) METRIC_NAME="code_kaggle" ;;
    * ) METRIC_NAME="" ;;
  esac

  # Fallback to lightweight content-based detection if not matched
  if [ -z "${METRIC_NAME}" ]; then
    if grep -qm1 '"gt_answers"' "${REQUESTS_FILE}"; then
      METRIC_NAME="code_kaggle"
    elif grep -qm1 '"public_tests"' "${REQUESTS_FILE}"; then
      METRIC_NAME="code_contests"
    elif grep -qm1 '"target_plan_number"' "${REQUESTS_FILE}"; then
      METRIC_NAME="tool_use_plan"
    elif grep -qm1 '"tools"' "${REQUESTS_FILE}"; then
      METRIC_NAME="tool_use_execution"
    fi
  fi

  if [ -z "${METRIC_NAME}" ]; then
    echo "Skipping unrecognized dataset file: ${FILENAME}" >&2
    continue
  fi

  BASE_NO_EXT="${FILENAME%.jsonl}"
  OUT_DIR="${OUT_ROOT}/${METRIC_NAME}/${BASE_NO_EXT}"
  mkdir -p "${OUT_DIR}"

  RESPONSES_FILE="${OUT_DIR}/${MODEL_NAME}_${METRIC_NAME}_responses.jsonl"
  METRICS_FILE="${OUT_DIR}/${MODEL_NAME}_${METRIC_NAME}_metrics.json"

  echo "Processing ${REQUESTS_FILE} with metric ${METRIC_NAME}..."

  # If the input already contains responses, evaluate directly; otherwise generate first.
  if grep -q '"response"' "${REQUESTS_FILE}"; then
    RESPONSES_FILE="${REQUESTS_FILE}"
  else
    python3 -m axlearn.open_api.generator \
      --model "${MODEL_NAME}" \
      --client_name "${CLIENT_NAME}" \
      --input_file "${REQUESTS_FILE}" \
      --output_file "${RESPONSES_FILE}" \
      --concurrency "${CONCURRENCY}" \
      --retry_sleep_in_seconds "${RETRY_SLEEP}" \
      --decode_parameters "${DECODE_PARAMETERS}"
  fi

  # Special env for code_kaggle (execution needs a writable data dir)
  if [ "${METRIC_NAME}" = "code_kaggle" ] || [ "${METRIC_NAME}" = "code_kaggle_retry" ] || [ "${METRIC_NAME}" = "code_kaggle_oracle" ]; then
    export CODE_KAGGLE_DATA_DIR="${OUT_DIR}/code_kaggle_data"
    mkdir -p "${CODE_KAGGLE_DATA_DIR}"
  fi

  python3 -m axlearn.open_api.evaluator \
    --model "${MODEL_NAME}" \
    --client_name "${CLIENT_NAME}" \
    --metric_name "${METRIC_NAME}" \
    --input_file "${RESPONSES_FILE}" \
    --output_file "${METRICS_FILE}"

  echo "Wrote metrics to ${METRICS_FILE}"
done


