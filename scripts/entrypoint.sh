#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_API_URL="${MODEL_API_URL:?MODEL_API_URL is required}"
export MODEL_API_KEY="${MODEL_API_KEY:?MODEL_API_KEY is required}"
export MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
export IS_SANDBOX="${IS_SANDBOX:-1}"
export EVAL_INPUT_ROOT="${EVAL_INPUT_ROOT:-/input}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/output}"
export EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-/logs}"
export EVAL_WORK_ROOT="${EVAL_WORK_ROOT:-/tmp/claude_eval_workspace}"
export CLAUDE_CLI_PATH="${CLAUDE_CLI_PATH:-$(command -v claude)}"
export CLAUDE_SETTING_SOURCES="${CLAUDE_SETTING_SOURCES:-project,local}"
export CLAUDE_DEBUG_TO_STDERR="${CLAUDE_DEBUG_TO_STDERR:-0}"
export EVAL_LOG_MODE="${EVAL_LOG_MODE:-submission}"
export EVAL_VERBOSE_LOGS="${EVAL_VERBOSE_LOGS:-0}"
export CLAUDE_HOME="${CLAUDE_HOME:-/tmp/claude-home}"
export HOME="${HOME:-/root}"
export CCR_HOME="${CCR_HOME:-${HOME}/.claude-code-router}"
export CCR_HOST="${CCR_HOST:-127.0.0.1}"
export CCR_PORT="${CCR_PORT:-3456}"
export CLAUDE_ROUTER_BASE_URL="${CLAUDE_ROUTER_BASE_URL:-${CLAUDE_CC_SWITCH_BASE_URL:-http://127.0.0.1:${CCR_PORT}}}"
export CCR_APIKEY="${CCR_APIKEY:-}"
export CCR_SERVER_ENTRY="${CCR_SERVER_ENTRY:-${PROJECT_ROOT}/vendor/claude-code-router/packages/server/dist/index.js}"
export NO_PROXY="127.0.0.1,localhost"
export HTTP_PROXY=""
export HTTPS_PROXY=""
export ALL_PROXY=""
export http_proxy=""
export https_proxy=""
export all_proxy=""

mkdir -p "${EVAL_LOG_ROOT}" "${EVAL_WORK_ROOT}" "${EVAL_OUTPUT_ROOT}" "${CLAUDE_HOME}" "${HOME}" "${CCR_HOME}"
exec 2>&1

if [[ ! -x "${CLAUDE_CLI_PATH}" ]]; then
  echo "[fatal] claude CLI not found or not executable: ${CLAUDE_CLI_PATH}"
  exit 2
fi
if ! command -v node >/dev/null 2>&1; then
  echo "[fatal] node is required but not found in PATH"
  exit 2
fi
if [[ ! -f "${CCR_SERVER_ENTRY}" ]]; then
  echo "[fatal] claude-code-router server entry not found: ${CCR_SERVER_ENTRY}"
  exit 2
fi

cleanup() {
  if [[ -n "${CCR_PID:-}" ]]; then
    kill "${CCR_PID}" >/dev/null 2>&1 || true
    wait "${CCR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_port() {
  python - "$1" "$2" <<'PY2'
import socket
import sys
import time
host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 20
while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect((host, port))
    except OSError:
        time.sleep(0.2)
    else:
        sock.close()
        raise SystemExit(0)
raise SystemExit(1)
PY2
}

echo "[info] claudeCliPath=${CLAUDE_CLI_PATH}"
echo "[info] logMode=${EVAL_LOG_MODE}"
echo "[info] evalInputRoot=${EVAL_INPUT_ROOT}"
echo "[info] evalOutputRoot=${EVAL_OUTPUT_ROOT}"
echo "[info] evalLogRoot=${EVAL_LOG_ROOT}"
echo "[info] ccrHome=${CCR_HOME}"
echo "[info] ccrServerEntry=${CCR_SERVER_ENTRY}"
echo "[info] ccrListen=http://${CCR_HOST}:${CCR_PORT}"

python "${PROJECT_ROOT}/scripts/write_ccr_config.py"

SERVICE_PORT="${CCR_PORT}" \
HOME="${HOME}" \
node "${CCR_SERVER_ENTRY}" >>"${EVAL_LOG_ROOT}/claude_code_router.log" 2>&1 &
CCR_PID=$!
wait_for_port "${CCR_HOST}" "${CCR_PORT}"

cd "${PROJECT_ROOT}"
exec python -m app.run_eval | tee -a "${EVAL_LOG_ROOT}/runtime.log"
