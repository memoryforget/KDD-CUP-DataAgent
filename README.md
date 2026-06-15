# KDD Cup 2026 Submission Build

This directory is an isolated, submission-oriented version of the current local pipeline:

`Claude Agent SDK -> claude-code-router -> MODEL_API_URL`

It is designed for the competition runtime contract:
- `/input` mounted read-only
- `/output` mounted read-write
- `/logs` mounted read-write
- `MODEL_API_URL`, `MODEL_API_KEY`, `MODEL_NAME` injected at runtime

## Layout
- `app/run_eval.py`: benchmark runner with per-task execution, MCP `answer` tool, CSV normalization, and concurrent scheduling
- `scripts/entrypoint.sh`: container startup flow
- `scripts/write_ccr_config.py`: writes the claude-code-router config from runtime env vars
- `scripts/smoke_test_container.sh`: local Docker smoke test helper
- `vendor/claude-code-router/packages/server/dist`: vendored router server bundle used at runtime
- `Dockerfile`: runtime image build recipe

## Runtime flow
Container startup is intentionally small:
1. `entrypoint.sh` validates required env vars and paths.
2. `write_ccr_config.py` generates router config under `CCR_HOME`.
3. Node starts the vendored `claude-code-router` server.
4. `app/run_eval.py` discovers tasks under `/input` and writes predictions under `/output/task_<id>/prediction.csv`.

## Default behavior
This project defaults to submission-safe logging:
- one runtime log at `/logs/runtime.log`
- one task log per task under `/logs/tasks/`
- no raw task context dumps in normal mode
- no long reasoning dumps in normal mode

Use `EVAL_LOG_MODE=debug` only for local debugging.

## Important runtime env vars
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

## Local build
```bash
docker build -t team1213:v1 .
```

For official submission packaging, use the assigned team id and submission version exactly.
For your current team id, the first submission should be:
- image name: `team1213:v1`
- archive filename: `team1213_v1.tar.gz`

Example export command:
```bash
docker save team1213:v1 | gzip > team1213_v1.tar.gz
```

## Local smoke test
Make sure your model service is already running and reachable through `MODEL_API_URL`.

```bash
export MODEL_API_URL=http://127.0.0.1:8000/v1
export MODEL_API_KEY=dummy
export MODEL_NAME=qwen3.5-35b-a3b
export EVAL_TASK_IDS=11
bash scripts/smoke_test_container.sh team1213:v1
```

## Competition packaging notes
This setup is aligned with the official rules in the following ways:
- it reads the model endpoint only from runtime env vars
- it writes predictions only under `/output/task_<id>/prediction.csv`
- it persists logs under `/logs`
- it can process all tasks by traversing `/input`
- it supports concurrent execution through `EVAL_MAX_WORKERS`
- it sets a direct `ENTRYPOINT` suitable for `docker run` without extra parameters

## Remaining pre-submission checks
Before the final image is submitted, still verify these points on a machine with Docker:
- the image builds cleanly
- `claude-code-router` starts inside the container
- the container can finish at least a representative regression set
- log volume under `/logs` is acceptable
- total image archive size stays below the competition limit

## Official Run Contract
After `docker load -i team1213_v1.tar.gz`, the evaluation system should be able to start the image directly with a command in the official shape:

```bash
docker run --rm  -it --add-host=host.docker.internal:host-gateway -v /vepfs-mlp2/c20250602/500050/lh/xqf/kddcup2026-data-agents-starter-kit/demo_samples_phase2/input:/input:ro   -v /vepfs-mlp2/c20250602/500050/lh/xqf/tmp_kdd_testvideo/kddcup_eval_output:/output:rw   -v /vepfs-mlp2/c20250602/500050/lh/xqf/tmp_kdd_testvideo/kddcup_eval_logs:/logs:rw   -e MODEL_API_URL=http://host.docker.internal:8005/v1   -e MODEL_API_KEY=1   -e MODEL_NAME=qwen3.5-35b-a3b   team1213:v1
```

No extra startup arguments are required because the image already sets `ENTRYPOINT`.
