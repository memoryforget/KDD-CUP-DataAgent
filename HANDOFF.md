# v4-B+ Handoff Notes

> 给打包/测试同学的简要说明。基于 origin/v3，分支 `v4-dev`。
> 提交目标：team1213:v2 (假设 v1 是 v3 baseline)

## 与 v3 的差异（汇总）

### 基础设施层（Phase A）
- **CCR watchdog**：`scripts/entrypoint.sh` 加监督进程，CCR 死了自动重启；node 启动加 `--max-old-space-size=2048`
- **失败重试**：`app/run_eval.py` 的 `run_task_with_retry`。环境变量 `EVAL_TASK_MAX_RETRIES`（默认 1）。第二轮 prompt 注入失败原因
- **单 task 超时**：`asyncio.wait_for(timeout=EVAL_TASK_TIMEOUT_SEC)`（默认 600s）
- **兜底 prediction.csv**：两次都失败时自动写 `result\nunknown\n`，避免缺文件

### 结构化数据工具层（Phase B）
新增 `app/structured_tools.py`，注册 5 个 MCP 工具：
- `inspect_data` — 扫 context/，返回 csv/json/sqlite/pdf/docx 的 schema 摘要（含枚举值列表）
- `sqlite_query` — 安全只读 SELECT/WITH（拒绝 INSERT/UPDATE/DROP）
- `pandas_query` — 在 csv 上跑 pandas 表达式（沙箱化 eval）
- `read_pdf_pages` — pymupdf 读指定页文字
- `read_docx_full` — python-docx 读段落

### Prompt 注入层（Phase B+）
在 `app/run_eval.py` 的 `run_task` 里：
- **自动调 inspect_data + 把摘要拼到 task prompt**（agent 不需要主动调）
- **自动读 knowledge.md + 拼到 task prompt**（agent 强制看到）

### 答案后处理层
- **`auto_trim_answer`**：answer 工具自动 drop 全空列、（n>1 时）全相同值列。可由 `EVAL_ANSWER_AUTO_TRIM=0` 关闭
- **`validate_answer_submission`** 不变（去重列名、规范化空值）

### 依赖
`requirements.txt` 新增：`pandas`、`pymupdf`、`python-docx`

## 本地验证结果（demo 50-task）

| | baseline (v3) | v4-B+ run1 | v4-B+ run2 |
|---|---|---|---|
| avg score | 0.4917 | **0.6333** | 0.5933 |
| 完美 (1.000) | 21/50 | 31/50 | 30/50 |
| 缺 csv | 8/50 | 0/50 ✅ | 0/50 ✅ |
| 跑时（workers=30）| 35min @5 | 18min | 18min |

跨 3 次 run（含 baseline）stdev = 0.073。**v4-B+ 相对 baseline 稳定 +0.10~0.14。**

## 比赛运行约定（与 v3 一致）

启动时由 `scripts/entrypoint.sh` 处理。需要的环境变量：
- `MODEL_API_URL` (必)
- `MODEL_API_KEY` (必)
- `MODEL_NAME` (必)

可选环境变量（v4 新增）：
- `EVAL_MAX_WORKERS` — 并发，建议 30（之前 v3 默认 4）
- `CLAUDE_EVAL_MAX_TURNS` — 单 task 最大轮数（默认 40）
- `EVAL_TASK_TIMEOUT_SEC` — 单 task 总超时（默认 600）
- `EVAL_TASK_MAX_RETRIES` — 失败重试次数（默认 1）
- `EVAL_TASK_FALLBACK_CSV` — 兜底 csv 开关（默认 1）
- `EVAL_ANSWER_AUTO_TRIM` — 自动裁剪冗余列（默认 1）

输入输出契约不变：`/input` ro、`/output` rw、`/logs` rw、prediction 写在 `/output/task_<id>/prediction.csv`。

## Docker 构建（与 v3 同流程）

```bash
docker build -t team1213:v2 .
docker save team1213:v2 | gzip > team1213_v2.tar.gz
```

注意 Dockerfile 第 29 行 `COPY models/all-MiniLM-L6-v2 /opt/models/all-MiniLM-L6-v2` 需要本地有 `models/all-MiniLM-L6-v2` 目录（可选向量 RAG 用，未提供也能跑，主流程不依赖向量召回）。

## 本地 smoke test

```bash
export MODEL_API_URL=http://127.0.0.1:8000/v1
export MODEL_API_KEY=dummy
export MODEL_NAME=qwen3.5-35b-a3b
export EVAL_TASK_IDS=22,38,67
bash scripts/smoke_test_container.sh team1213:v2
```

## 已知"稳定 0 分"的 task（不是基础设施问题，是题目本身难/有歧义）

这些 12 个稳定答错，不依赖于本次提交质量：
task_19, task_27, task_80, task_86, task_89, task_163, task_169, task_173, task_180, task_199, task_259, task_355
