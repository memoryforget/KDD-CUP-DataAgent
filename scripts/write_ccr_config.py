#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


OPENAI_CHAT_SUFFIX = "/v1/chat/completions"
OPENAI_RESPONSES_SUFFIX = "/v1/responses"
ANTHROPIC_MESSAGES_SUFFIX = "/v1/messages"


def normalize_provider_api_base_url(raw_url: str) -> str:
    url = raw_url.rstrip("/")
    lower = url.lower()

    if lower.endswith(OPENAI_CHAT_SUFFIX):
        return url
    if lower.endswith(OPENAI_RESPONSES_SUFFIX):
        return url[: -len(OPENAI_RESPONSES_SUFFIX)] + OPENAI_CHAT_SUFFIX
    if lower.endswith(ANTHROPIC_MESSAGES_SUFFIX):
        return url[: -len(ANTHROPIC_MESSAGES_SUFFIX)] + OPENAI_CHAT_SUFFIX
    if lower.endswith("/v1"):
        return url + "/chat/completions"
    if lower.endswith("/chat/completions"):
        return url
    return url + OPENAI_CHAT_SUFFIX


def main() -> None:
    home = Path(os.environ["CCR_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "logs").mkdir(parents=True, exist_ok=True)

    provider_name = os.environ.get("CCR_PROVIDER_NAME", "openai")
    model_name = os.environ["MODEL_NAME"]
    api_base_url = normalize_provider_api_base_url(os.environ["MODEL_API_URL"])
    api_key = os.environ["MODEL_API_KEY"]
    host = os.environ.get("CCR_HOST", "127.0.0.1")
    port = int(os.environ.get("CCR_PORT", "3456"))
    router_apikey = os.environ.get("CCR_APIKEY", "")

    config = {
        "HOST": host,
        "PORT": port,
        "APIKEY": router_apikey,
        "LOG": True,
        "LOG_LEVEL": "debug",
        "API_TIMEOUT_MS": 600000,
        "Providers": [
            {
                "name": provider_name,
                "api_base_url": api_base_url,
                "api_key": api_key,
                "models": [model_name],
                "transformer": {"use": ["anthropic"]},
            }
        ],
        "Router": {
            "default": f"{provider_name},{model_name}",
            "background": f"{provider_name},{model_name}",
            "think": f"{provider_name},{model_name}",
            "longContext": f"{provider_name},{model_name}",
            "webSearch": f"{provider_name},{model_name}",
            "image": f"{provider_name},{model_name}",
        },
    }

    config_path = home / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] wrote claude-code-router config to {config_path}")
    print(f"[info] normalized provider api_base_url={api_base_url}")


if __name__ == "__main__":
    main()
