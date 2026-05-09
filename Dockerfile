FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive     PIP_NO_CACHE_DIR=1     PYTHONUNBUFFERED=1     NODE_ENV=production
# 已优化：Debian国内阿里云源，解决apt慢
RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends     bash     ca-certificates     curl     procps     tini     xz-utils     \
    && rm -rf /var/lib/apt/lists/*

ARG NODE_VERSION=20.19.5

RUN ARCH="$(dpkg --print-architecture)"     \
    && case "$ARCH" in         \
        amd64) NODE_ARCH='x64' ;;         \
        arm64) NODE_ARCH='arm64' ;;         \
        *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;       \
    esac     \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" -o /tmp/node.tar.xz     \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1     \
    && rm -f /tmp/node.tar.xz     \
    && node -v     \
    && npm -v

WORKDIR /opt/kddcup-submission

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --default-timeout=1000

COPY models/all-MiniLM-L6-v2 /opt/models/all-MiniLM-L6-v2
ENV EVAL_RAG_EMBEDDING_MODEL=/opt/models/all-MiniLM-L6-v2 \
    EVAL_RAG_ENABLE_VECTOR=1 \
    EVAL_RAG_VECTOR_WEIGHT=0.30 \
    HF_HOME=/opt/models/.cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

RUN npm install -g @anthropic-ai/claude-code@2.1.112

COPY app ./app
COPY scripts ./scripts
COPY README.md ./README.md
COPY vendor/claude-code-router/packages/server/dist ./vendor/claude-code-router/packages/server/dist

RUN chmod +x     /opt/kddcup-submission/scripts/entrypoint.sh     /opt/kddcup-submission/scripts/smoke_test_container.sh

ENV CLAUDE_CLI_PATH=/usr/local/bin/claude     EVAL_INPUT_ROOT=/input     EVAL_OUTPUT_ROOT=/output     EVAL_LOG_ROOT=/logs     EVAL_WORK_ROOT=/tmp/claude_eval_workspace     CLAUDE_ROUTER_BASE_URL=http://127.0.0.1:3456     CLAUDE_SETTING_SOURCES=project,local     CLAUDE_DEBUG_TO_STDERR=0     EVAL_LOG_MODE=submission     EVAL_VERBOSE_LOGS=0

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/kddcup-submission/scripts/entrypoint.sh"]