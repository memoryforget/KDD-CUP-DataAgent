# KDD Cup 2026 Submission Build

This repository is a submission-oriented benchmark runner built around:

`Claude Agent SDK -> claude-code-router -> OpenAI-compatible model API`

It is designed for the competition runtime contract:
- `/input` mounted read-only
- `/output` mounted read-write
- `/logs` mounted read-write
- `MODEL_API_URL`, `MODEL_API_KEY`, and `MODEL_NAME` injected at runtime

## Layout
- `app/run_eval.py`: task runner, agent session orchestration, MCP tool registration, and answer writing
- `app/data_query_tools.py`: task-local structured data loading and unified SQLite query workspace
- `scripts/entrypoint.sh`: container startup flow
- `scripts/write_ccr_config.py`: router config generation from runtime env vars
- `scripts/smoke_test_container.sh`: local Docker smoke test helper
- `vendor/claude-code-router/packages/server/dist`: vendored router server bundle used at runtime
- `Dockerfile`: runtime image build recipe

## Runtime Flow
1. `entrypoint.sh` validates required environment variables and paths.
2. `write_ccr_config.py` generates the local `claude-code-router` config under `CCR_HOME`.
3. Node starts the vendored router server.
4. `app/run_eval.py` discovers tasks under `/input`.
5. Each task gets an isolated work directory, log file, and temporary SQLite workspace.
6. Final answers are written to `/output/task_<id>/prediction.csv`.

## Data Access Design
For each task, structured files under `context/` are unified into a temporary SQLite workspace:
- CSV files are imported as SQLite tables.
- JSON files are imported as SQLite tables.
- Existing SQLite or DB files are mirrored into the same workspace.

This gives the agent one consistent SQL query surface instead of multiple file-specific parsing paths.

## Exposed MCP Tools
The agent currently sees only a small task-local tool set:
- `list_context_files`
- `describe_query_workspace`
- `query_data`
- `answer`

Design intent:
- reduce tool distraction
- prefer structured reasoning through SQLite

## Agent Constraints
The system prompt and runtime settings currently enforce these rules:
- only read files inside the current task directory
- do not read other tasks or any `gold.csv`
- read `context/knowledge.md` before semantic interpretation when present
- use `describe_query_workspace` before querying tables
- use `query_data` for filtering, joins, aggregation, sorting, and ranking when structured data is available
- submit the final result only through the `answer` tool


## Default Logging
This project defaults to submission-safe logging:
- one runtime log at `/logs/runtime.log`
- one task log per task under `/logs/tasks/`
- no full reasoning dumps in normal mode
- no raw context dumps in normal mode

Useful toggles:
- `EVAL_LOG_MODE=submission|debug`
- `EVAL_VERBOSE_LOGS=0|1`

## Required Runtime Environment Variables
Required:
- `MODEL_API_URL`
- `MODEL_API_KEY`
- `MODEL_NAME`

Useful overrides:
- `EVAL_MAX_WORKERS`: concurrent task count
- `EVAL_MAX_TASKS`: cap number of tasks
- `EVAL_TASK_IDS`: comma-separated subset such as `11,38,80` or `task_11,task_38`
- `CLAUDE_EVAL_MAX_TURNS`: per-task max turns
- `EVAL_LOG_MODE=submission|debug`
- `EVAL_VERBOSE_LOGS=0|1`

`MODEL_API_URL` may be given as an OpenAI-compatible `/v1` base URL. The router config step will normalize it to `/v1/chat/completions`.

## Local Build
```bash
docker build -t team1213:v2 .
```

Example archive export:
```bash
docker save team1213:v2 | gzip > team1213_v2.tar.gz
```

## Local Smoke Test
Make sure your model service is already running and reachable through `MODEL_API_URL`.

```bash
export MODEL_API_URL=http://127.0.0.1:8000/v1
export MODEL_API_KEY=dummy
export MODEL_NAME=qwen3.5-35b-a3b
export EVAL_TASK_IDS=11
bash scripts/smoke_test_container.sh team1213:v2
```

## Example Official-Style Run
```bash
docker run --rm \
  --network=eval_net \
  -v /eval/data/input:/input:ro \
  -v /eval/submission/output:/output:rw \
  -v /eval/submission/logs:/logs:rw \
  -e MODEL_API_URL=<model_url> \
  -e MODEL_API_KEY=<api_key> \
  -e MODEL_NAME=qwen3.5-35b-a3b \
  -e EVAL_MAX_WORKERS=1 \
  team1213:v2
```

No extra startup arguments are required because the image already sets `ENTRYPOINT`.

## Pre-Submission Checks
Before final submission, verify:
- the image builds cleanly
- `claude-code-router` starts inside the container
- the container can finish a representative regression subset
- logs under `/logs` remain manageable
- the final image archive size stays within the competition limit
