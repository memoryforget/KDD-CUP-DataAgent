"""Video processing tools for the answer MCP server.

Uses the SAME model (qwen3.5-35b-a3b served via vLLM) with native video_url
input support to analyze video content. The model handles frame sampling
internally when given a video_url.

Following the official Phase 2 starter kit, video is always sent as a base64
data URI. This avoids relying on model-server access to local file paths.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

_requests = None


def _get_requests():
    global _requests
    if _requests is None:
        import requests
        _requests = requests
    return _requests


# --------------- Configuration ---------------

# Use the SAME model API as the main model
# Note: These are read at import time. If env vars change after import,
# use the function-level overrides or re-import.
def _get_model_api_url():
    return os.environ.get("MODEL_API_URL", "")

def _get_model_api_key():
    return os.environ.get("MODEL_API_KEY", "")

def _get_model_name():
    return os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b")

# API call parameters
VIDEO_API_TIMEOUT = 180  # videos take longer to process
VIDEO_MAX_TOKENS = 4096
VIDEO_MAX_RETRIES = 2
VIDEO_FPS = 2  # frame sampling rate for video


# --------------- Video metadata ---------------

def get_video_metadata(video_path: Path) -> dict[str, Any]:
    """Get basic metadata about a video file using cv2."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"is_error": True, "message": f"cannot open video: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0,
            "resolution": f"{width}x{height}",
        }
    except ImportError:
        # Fallback: just report file size
        return {
            "path": str(video_path),
            "size_bytes": video_path.stat().st_size,
        }


# --------------- Model API call with video ---------------

def _build_video_url(video_path: Path) -> str:
    """Build the video URL for the API call.

    Always encode the task-local video as a base64 data URI.
    The official Phase 2 starter kit uses this format for video_url input.
    """
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:video/mp4;base64,{video_b64}"


def _call_model_with_video(
    video_path: Path,
    prompt: str,
    max_tokens: int = VIDEO_MAX_TOKENS,
    fps: int = VIDEO_FPS,
) -> str:
    """Call the model API with a video input using video_url content type.

    Uses the same model (qwen3.5-35b-a3b) which natively supports video via vLLM.
    Returns the model's text response, or an error string prefixed with 'ERROR:'.
    """
    requests = _get_requests()
    api_url = _get_model_api_url().rstrip("/") + "/chat/completions"
    model_name = _get_model_name()
    api_key = _get_model_api_key()

    video_url = _build_video_url(video_path)

    # Build message content with video_url + text
    content = [
        {
            "type": "video_url",
            "video_url": {"url": video_url},
        },
        {
            "type": "text",
            "text": prompt,
        },
    ]

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "extra_body": {
            "mm_processor_kwargs": {"fps": fps, "do_sample_frames": True},
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(VIDEO_MAX_RETRIES + 1):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=VIDEO_API_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and data["choices"]:
                    msg_content = data["choices"][0]["message"]["content"]
                    # Handle thinking models - strip <think> blocks if present
                    if "<think>" in msg_content and "</think>" in msg_content:
                        think_end = msg_content.rfind("</think>")
                        msg_content = msg_content[think_end + len("</think>"):].strip()
                    return msg_content
                return f"ERROR: unexpected response format: {json.dumps(data)[:300]}"
            else:
                error_msg = f"ERROR: API returned {resp.status_code}: {resp.text[:400]}"
                if attempt < VIDEO_MAX_RETRIES:
                    time.sleep(3 * (attempt + 1))
                    continue
                return error_msg
        except Exception as exc:
            if attempt < VIDEO_MAX_RETRIES:
                time.sleep(3 * (attempt + 1))
                continue
            return f"ERROR: request failed: {exc!r}"

    return "ERROR: all retries exhausted"


# --------------- High-level tools ---------------

def analyze_video_frames(
    video_path: Path,
    question: str,
    interval_sec: float = 2.0,
    max_frames: int = 30,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Analyze a video file using the model's native video understanding.

    Sends the entire video to the model with a question. The model internally
    samples frames at the configured fps rate.

    Args:
        video_path: Path to the video file
        question: The question to ask about the video content
        interval_sec: Not used directly (model handles frame sampling via fps)
        max_frames: Not used directly (controlled by fps parameter)
        batch_size: Not used (video sent as a whole)

    Returns a dict with:
        - video_metadata: basic video info
        - frame_count: estimated frames the model sees (duration * fps)
        - summary: the model's response
    """
    if not video_path.exists():
        return {"is_error": True, "message": f"video not found: {video_path}"}

    if not _get_model_api_url():
        return {"is_error": True, "message": "MODEL_API_URL not configured"}

    metadata = get_video_metadata(video_path)
    if metadata.get("is_error"):
        return metadata

    # Calculate estimated frames the model will see
    duration = metadata.get("duration_sec", 0)
    estimated_frames = int(duration * VIDEO_FPS) if duration else 0

    # Build a focused prompt for the video analysis
    full_prompt = (
        f"请完整观看这段视频，严格按照以下要求回答问题：\n\n"
        f"问题：{question}\n\n"
        f"强制要求：\n"
        f"1. 只提取视频中明确出现的信息，**绝对不要编造、推断或总结任何内容**\n"
        f"2. 注意视频中所有文字：包括字幕、屏幕显示、表格、名片、图表等\n"
        f"3. 遇到数据表格、列表或排名，**完整提取所有可见条目和数值**，保持原有格式\n"
        f"4. 所有数值、姓名、电话号码、日期必须**精确到每一个字符**，不得修改\n"
        f"5. 如果视频中**完全没有找到相关信息**，请直接回答\"unknown\"，不要说其他话\n"
        f"6. **一次性给出所有答案**，不要分多次输出，不要添加任何解释、说明或前缀\n"
        f"7. 答案中不要包含任何思考过程，只输出最终结果"
    )

    response = _call_model_with_video(video_path, full_prompt, fps=VIDEO_FPS)

    if response.startswith("ERROR:"):
        return {
            "is_error": True,
            "message": response,
            "video_metadata": metadata,
        }

    return {
        "video_metadata": metadata,
        "frame_count": estimated_frames,
        "summary": response,
    }
