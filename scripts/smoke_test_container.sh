#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-12133:v1}"
INPUT_ROOT="${INPUT_ROOT:-/vepfs-mlp2/c20250602/500050/lh/xqf/kddcup2026-data-agents-starter-kit/data/public/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vepfs-mlp2/c20250602/500050/lh/xqf/kddcup2026-data-agents-starter-kit/artifacts/submission_container_output}"
LOG_ROOT="${LOG_ROOT:-/vepfs-mlp2/c20250602/500050/lh/xqf/kddcup2026-data-agents-starter-kit/artifacts/submission_container_logs}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

docker run --rm   --cpus=8   --memory=32g   -v "${INPUT_ROOT}":/input:ro   -v "${OUTPUT_ROOT}":/output:rw   -v "${LOG_ROOT}":/logs:rw   -e MODEL_API_URL="${MODEL_API_URL:?MODEL_API_URL is required}"   -e MODEL_API_KEY="${MODEL_API_KEY:?MODEL_API_KEY is required}"   -e MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"   -e EVAL_TASK_IDS="${EVAL_TASK_IDS:-11}"   -e EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-2}"   -e EVAL_LOG_MODE="${EVAL_LOG_MODE:-submission}"   "${IMAGE_TAG}"
