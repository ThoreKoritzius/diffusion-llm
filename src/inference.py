#!/usr/bin/env python3
"""
inference.py - Flask GUI with live decoding and optional GIF export

Usage:
  python inference.py
  python inference.py --no-open
  python inference.py --prompt "..." --context "..." --model-dir "diffusion-sql"

This version includes:
 - Public-safe queueing, rate limiting, and timeout controls for CPU-only hosts.
 - Live preview for content inside <SQL>...</SQL>.
 - Optional GIF generation (disabled by default for public robustness).
"""
import argparse
import io
import os
import time
import math
import threading
import uuid
import textwrap
import json
import random
import shlex
import traceback
import re
import hashlib
from typing import List, Dict
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, jsonify, Response, send_file, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
import webbrowser
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import numpy as np
import torch
import redis
from transformers import AutoTokenizer, AutoModelForMaskedLM

from denoising import denoise_steps

# -------------------------
def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--public", action="store_true", help="Bind on 0.0.0.0 and disable browser auto-open")
    parser.add_argument("--prompt", type=str, default="", help="Prefill prompt (optional)")
    parser.add_argument("--context", type=str, default="", help="Prefill context (optional)")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("MODEL_DIR_IN_CONTAINER", "diffusion-sql"), help="Default model dir (optional)")
    parser.add_argument("--host", type=str, default=os.environ.get("INFERENCE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env_int("INFERENCE_PORT", 7860))
    parser.add_argument("--seed", type=int, default=env_int("INFERENCE_SEED", 42), help="Deterministic seed (default 42).")
    parser.add_argument("--max-concurrent-runs", type=int, default=env_int("MAX_CONCURRENT_RUNS", 1))
    parser.add_argument("--max-queued-runs", type=int, default=env_int("MAX_QUEUED_RUNS", 5))
    parser.add_argument("--request-timeout-seconds", type=int, default=env_int("REQUEST_TIMEOUT_SECONDS", 60))
    parser.add_argument("--max-prompt-chars", type=int, default=env_int("MAX_PROMPT_CHARS", 1000))
    parser.add_argument("--max-context-chars", type=int, default=env_int("MAX_CONTEXT_CHARS", 8000))
    parser.add_argument("--max-steps", type=int, default=env_int("MAX_STEPS", 48))
    parser.add_argument("--max-max-len", type=int, default=env_int("MAX_MAX_LEN", 512))
    parser.add_argument("--max-sql-len", type=int, default=env_int("MAX_SQL_LEN", 128))
    parser.add_argument("--enable-gif", action="store_true", help="Enable GIF generation for runs")
    parser.add_argument("--run-ttl-seconds", type=int, default=env_int("RUN_TTL_SECONDS", 900))
    parser.add_argument("--worker", action="store_true", help="Run worker loop instead of web server")
    parser.add_argument("--redis-url", type=str, default=os.environ.get("REDIS_URL", "redis://redis:6379/0"))
    parser.add_argument("--inference-dtype", type=str, default=os.environ.get("INFERENCE_DTYPE", "fp16"), choices=["fp16", "fp32", "bf16", "auto"], help="Preferred model dtype for inference")
    return parser


def parse_runtime_args() -> argparse.Namespace:
    parser = build_parser()
    env_args = os.environ.get("INFERENCE_APP_ARGS", "").strip()
    if env_args:
        parsed, _unknown = parser.parse_known_args(shlex.split(env_args))
    else:
        parsed, _unknown = parser.parse_known_args()
    if parsed.public:
        parsed.host = "0.0.0.0"
        parsed.no_open = True
    return parsed


args = parse_runtime_args()

HOST = args.host
PORT = args.port
DEFAULT_MODEL_DIR = args.model_dir
DEFAULT_SEED = args.seed
MAX_CONCURRENT_RUNS = max(1, args.max_concurrent_runs)
MAX_QUEUED_RUNS = max(1, args.max_queued_runs)
REQUEST_TIMEOUT_SECONDS = max(1, args.request_timeout_seconds)
MAX_PROMPT_CHARS = max(1, args.max_prompt_chars)
MAX_CONTEXT_CHARS = max(1, args.max_context_chars)
MAX_STEPS = max(1, args.max_steps)
MAX_MAX_LEN = max(8, args.max_max_len)
MAX_SQL_LEN = max(1, args.max_sql_len)
ENABLE_GIF = bool(args.enable_gif)
RUN_TTL_SECONDS = max(60, args.run_ttl_seconds)
IS_WORKER = bool(args.worker)
REDIS_URL = args.redis_url
INFERENCE_DTYPE = (args.inference_dtype or "fp16").lower()
ALLOW_FAKE_REDIS = env_bool("ALLOW_FAKE_REDIS", "REDIS_URL" not in os.environ)
RATELIMIT_STORAGE_URI = os.environ.get("RATELIMIT_STORAGE_URI") or ("memory://" if ALLOW_FAKE_REDIS else REDIS_URL)
START_RUN_RATE_LIMIT = os.environ.get("START_RUN_RATE_LIMIT", "30 per minute; 300 per hour")
EXPORT_GIF_RATE_LIMIT = os.environ.get("EXPORT_GIF_RATE_LIMIT", "3 per minute; 20 per hour")
ADAPTIVE_RATE_LIMITS = env_bool("ADAPTIVE_RATE_LIMITS", True)
IDLE_STARTS_PER_MINUTE = max(1, env_int("IDLE_STARTS_PER_MINUTE", 12))
NORMAL_STARTS_PER_MINUTE = max(1, env_int("NORMAL_STARTS_PER_MINUTE", 6))
BUSY_STARTS_PER_MINUTE = max(1, env_int("BUSY_STARTS_PER_MINUTE", 2))
ETA_FALLBACK_SECONDS = max(5.0, env_float("ETA_FALLBACK_SECONDS", min(30, max(10, REQUEST_TIMEOUT_SECONDS * 0.5))))
FIRST_FRAME_FALLBACK_SECONDS = max(1.0, env_float("FIRST_FRAME_FALLBACK_SECONDS", 8.0))
RESULT_CACHE_ENABLED = env_bool("RESULT_CACHE_ENABLED", True)
RESULT_CACHE_TTL_SECONDS = max(60, env_int("RESULT_CACHE_TTL_SECONDS", 86400))
RESULT_CACHE_VERSION = os.environ.get("RESULT_CACHE_VERSION", "v1")
CHARS_PER_TOKEN_EST = 3.0
STRUCTURE_TOKEN_RESERVE = 32
PROMPT_BUDGET_RATIO = 0.35

# -------------------------
# Flask app & template
# -------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_url_path="/static",
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=RATELIMIT_STORAGE_URI,
)


# -------------------------
# Run storage / state
# -------------------------
MODEL_CACHE = {"dir": None, "tokenizer": None, "model": None}
MODEL_VALIDATION_CACHE: Dict[str, Dict] = {}
MODEL_INFO_CACHE: Dict[str, Dict] = {}
OUT_DIR = os.path.join(os.path.abspath("."), "generated_gifs")
REDIS_CLIENT = None

QUEUE_KEY = "diffusion:queue"
ACTIVE_SET_KEY = "diffusion:active"
METRICS_KEY = "diffusion:metrics"


def run_key(run_id: str) -> str:
    return f"diffusion:run:{run_id}"


def snaps_key(run_id: str) -> str:
    return f"diffusion:run:{run_id}:snaps"


STANDALONE_MODE = False


def get_redis() -> redis.Redis:
    global REDIS_CLIENT, STANDALONE_MODE
    if REDIS_CLIENT is None:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        try:
            client.ping()
            REDIS_CLIENT = client
        except (redis.exceptions.RedisError, OSError):
            if not ALLOW_FAKE_REDIS:
                raise
            import fakeredis
            print(f"[WARN] Redis unavailable at {REDIS_URL}; using in-memory store (standalone mode, single process)")
            REDIS_CLIENT = fakeredis.FakeRedis(decode_responses=True)
            STANDALONE_MODE = True
    return REDIS_CLIENT


def now_ts() -> float:
    return time.time()


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def compute_char_heuristic_limits(max_len: int, sql_len: int) -> Dict[str, int]:
    max_len = max(8, int(max_len))
    sql_len = max(1, int(sql_len))
    effective_sql_reserve = max(sql_len, 16)
    text_token_budget = max(32, max_len - effective_sql_reserve - STRUCTURE_TOKEN_RESERVE)
    combined_char_budget = int(math.floor(text_token_budget * CHARS_PER_TOKEN_EST))
    prompt_char_cap = max(120, int(math.floor(combined_char_budget * PROMPT_BUDGET_RATIO)))
    context_char_cap = max(240, combined_char_budget - prompt_char_cap)
    return {
        "effective_sql_reserve": effective_sql_reserve,
        "text_token_budget": text_token_budget,
        "combined_char_budget": combined_char_budget,
        "prompt_char_cap": prompt_char_cap,
        "context_char_cap": context_char_cap,
    }


def extract_sql_only_from_text(s: str) -> str:
    if not s:
        return ""
    start_marker = "<SQL>"
    end_marker = "</SQL>"
    start_idx = s.rfind(start_marker)
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)
    end_idx = s.find(end_marker, start_idx)
    if end_idx == -1:
        return s[start_idx:].strip()
    if end_idx < start_idx:
        return ""
    return s[start_idx:end_idx].strip()


def strip_final_masks(sql_text: str) -> str:
    if not sql_text:
        return ""
    cleaned = sql_text.replace("____", "")
    # Handle spaced tag variants from token-level displays, e.g. "< p a d >".
    cleaned = re.sub(r"<\s*/?\s*p\s*a\s*d\s*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*p\s*a\s*d\s*\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*/?\s*s\s*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*m\s*a\s*s\s*k\s*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("<pad>", " ").replace("<PAD>", " ").replace("[PAD]", " ")
    cleaned = cleaned.replace("<s>", " ").replace("</s>", " ").replace("<mask>", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def decode_sql_from_token_ids(tokenizer, token_ids: List[int], mask_id: int | None, pad_id: int | None) -> str:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if mask_id is not None:
        special_ids.add(int(mask_id))
    if pad_id is not None:
        special_ids.add(int(pad_id))
    filtered_ids = [int(tid) for tid in token_ids if int(tid) not in special_ids]
    if not filtered_ids:
        return ""
    return tokenizer.decode(filtered_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


def replace_sql_section(full_text: str, sql_only: str) -> str:
    start_marker = "<SQL>"
    end_marker = "</SQL>"
    try:
        start_idx = full_text.index(start_marker) + len(start_marker)
        end_idx = full_text.index(end_marker, start_idx)
        return full_text[:start_idx] + (" " + sql_only + " " if sql_only else " ") + full_text[end_idx:]
    except ValueError:
        return full_text


def to_redis_value(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def from_redis_run(raw: Dict[str, str]) -> Dict:
    if not raw:
        return {}
    out = dict(raw)
    out["done"] = out.get("done", "0") == "1"
    out["timed_out"] = out.get("timed_out", "0") == "1"
    out["cancel_requested"] = out.get("cancel_requested", "0") == "1"
    out["cache_hit"] = out.get("cache_hit", "0") == "1"
    for k in ("created_at", "updated_at", "started_at", "finished_at"):
        if out.get(k):
            try:
                out[k] = float(out[k])
            except ValueError:
                out[k] = None
        else:
            out[k] = None
    if out.get("snapshot_count"):
        try:
            out["snapshot_count"] = int(out["snapshot_count"])
        except ValueError:
            out["snapshot_count"] = 0
    else:
        out["snapshot_count"] = 0
    if out.get("payload"):
        try:
            out["payload"] = json.loads(out["payload"])
        except json.JSONDecodeError:
            out["payload"] = {}
    else:
        out["payload"] = {}
    return out


def redis_set_run(run_id: str, mapping: Dict) -> None:
    r = get_redis()
    payload = {k: to_redis_value(v) for k, v in mapping.items()}
    if payload:
        r.hset(run_key(run_id), mapping=payload)
    r.expire(run_key(run_id), RUN_TTL_SECONDS)
    r.expire(snaps_key(run_id), RUN_TTL_SECONDS)


def redis_get_run(run_id: str) -> Dict:
    r = get_redis()
    return from_redis_run(r.hgetall(run_key(run_id)))


def redis_update_run(run_id: str, **updates) -> None:
    updates["updated_at"] = now_ts()
    redis_set_run(run_id, updates)


def redis_append_snapshot(run_id: str, snapshot_obj: Dict) -> None:
    r = get_redis()
    pipe = r.pipeline()
    pipe.rpush(snaps_key(run_id), json.dumps(snapshot_obj))
    pipe.hincrby(run_key(run_id), "snapshot_count", 1)
    pipe.hset(run_key(run_id), "updated_at", to_redis_value(now_ts()))
    pipe.expire(run_key(run_id), RUN_TTL_SECONDS)
    pipe.expire(snaps_key(run_id), RUN_TTL_SECONDS)
    pipe.execute()


def redis_get_snapshot_delta(run_id: str, after_idx: int) -> tuple[int, List[Dict]]:
    r = get_redis()
    total = r.llen(snaps_key(run_id))
    if after_idx < 0:
        after_idx = 0
    if after_idx >= total:
        return total, []
    rows = r.lrange(snaps_key(run_id), after_idx, -1)
    out = []
    for row in rows:
        try:
            out.append(json.loads(row))
        except json.JSONDecodeError:
            continue
    return total, out


def cache_signature_payload(payload: Dict) -> Dict:
    return {
        "version": RESULT_CACHE_VERSION,
        "prompt": payload.get("prompt", ""),
        "context": payload.get("context", ""),
        "model_dir": payload.get("model_dir", ""),
        "steps": int(payload.get("steps", 0)),
        "max_len": int(payload.get("max_len", 0)),
        "sql_len": int(payload.get("sql_len", 0)),
        "top_k": int(payload.get("top_k", 0)),
        "top_p": float(payload.get("top_p", 0)),
        "seed": int(DEFAULT_SEED),
        "dtype": INFERENCE_DTYPE,
    }


def result_cache_key(payload: Dict) -> str:
    raw = json.dumps(cache_signature_payload(payload), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"diffusion:cache:{digest}"


def read_result_cache(payload: Dict) -> Dict | None:
    if not RESULT_CACHE_ENABLED:
        return None
    raw = get_redis().get(result_cache_key(payload))
    if not raw:
        return None
    try:
        cached = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not cached.get("snapshots"):
        return None
    return cached


def write_result_cache(payload: Dict, sql_only: str, display: str, snapshots: List[Dict]) -> None:
    if not RESULT_CACHE_ENABLED or not snapshots:
        return
    cached = {
        "payload": cache_signature_payload(payload),
        "sql_only": sql_only or "",
        "display": display or "",
        "snapshots": snapshots,
        "created_at": now_ts(),
    }
    get_redis().setex(result_cache_key(payload), RESULT_CACHE_TTL_SECONDS, json.dumps(cached))


def create_cached_run(payload: Dict, cached: Dict) -> str:
    run_id = uuid.uuid4().hex
    now = now_ts()
    snapshots = cached.get("snapshots") or []
    redis_set_run(run_id, {
        "state": "done",
        "status": "cached result",
        "done": True,
        "timed_out": False,
        "cancel_requested": False,
        "sql_only": cached.get("sql_only", ""),
        "display": cached.get("display", ""),
        "gif_url": "",
        "created_at": now,
        "updated_at": now,
        "started_at": now,
        "finished_at": now,
        "snapshot_count": 0,
        "payload": payload,
        "cache_hit": True,
    })
    for snap in snapshots:
        redis_append_snapshot(run_id, snap)
    redis_update_run(run_id, snapshot_count=len(snapshots), status="cached result")
    return run_id


def queue_position(run_id: str) -> int | None:
    r = get_redis()
    q = r.lrange(QUEUE_KEY, 0, -1)
    try:
        return q.index(run_id) + 1
    except ValueError:
        return None


def prune_inactive_workers() -> None:
    r = get_redis()
    for run_id in r.smembers(ACTIVE_SET_KEY):
        run = redis_get_run(run_id)
        if not run or run.get("done") or run.get("state") not in {"running", "stopping"}:
            r.srem(ACTIVE_SET_KEY, run_id)


def active_worker_count() -> int:
    prune_inactive_workers()
    return int(get_redis().scard(ACTIVE_SET_KEY))


def average_run_duration_seconds() -> float:
    raw = get_redis().hget(METRICS_KEY, "avg_duration")
    if raw is None:
        return ETA_FALLBACK_SECONDS
    try:
        avg = float(raw)
    except ValueError:
        return ETA_FALLBACK_SECONDS
    return max(1.0, min(float(REQUEST_TIMEOUT_SECONDS), avg))


def average_first_frame_seconds() -> float:
    raw = get_redis().hget(METRICS_KEY, "avg_first_frame")
    if raw is None:
        return min(FIRST_FRAME_FALLBACK_SECONDS, float(REQUEST_TIMEOUT_SECONDS))
    try:
        avg = float(raw)
    except ValueError:
        return min(FIRST_FRAME_FALLBACK_SECONDS, float(REQUEST_TIMEOUT_SECONDS))
    return max(1.0, min(float(REQUEST_TIMEOUT_SECONDS), avg))


def update_metric_average(avg_field: str, count_field: str, value: float) -> None:
    r = get_redis()
    prev_avg = r.hget(METRICS_KEY, avg_field)
    prev_n = r.hget(METRICS_KEY, count_field)
    try:
        prev_avg_f = float(prev_avg) if prev_avg else 0.0
        prev_n_i = int(prev_n) if prev_n else 0
    except ValueError:
        prev_avg_f = 0.0
        prev_n_i = 0
    new_n = prev_n_i + 1
    new_avg = value if prev_n_i == 0 else ((prev_avg_f * prev_n_i) + value) / new_n
    r.hset(METRICS_KEY, mapping={avg_field: new_avg, count_field: new_n})


def estimate_active_remaining_seconds(avg_duration: float | None = None) -> int:
    avg = avg_duration if avg_duration is not None else average_run_duration_seconds()
    remaining = 0.0
    now = now_ts()
    r = get_redis()
    for run_id in r.smembers(ACTIVE_SET_KEY):
        run = redis_get_run(run_id)
        started_at = run.get("started_at") if run else None
        if not started_at:
            remaining += avg
            continue
        try:
            elapsed = max(0.0, now - float(started_at))
        except (TypeError, ValueError):
            elapsed = 0.0
        remaining += max(3.0, avg - elapsed)
    return max(0, int(math.ceil(remaining)))


def estimate_eta_seconds(position: int | None) -> int | None:
    if position is None:
        return None
    avg = average_run_duration_seconds()
    first_frame_avg = average_first_frame_seconds()
    runs_ahead = max(0, int(position) - 1)
    eta = estimate_active_remaining_seconds(avg) + (runs_ahead * avg) + first_frame_avg
    return max(1, int(math.ceil(eta)))


def estimate_eta_confidence(position: int | None, eta_seconds: int | None) -> str:
    if position is None or eta_seconds is None:
        return "low"
    if position <= 1:
        return "high"
    raw = get_redis().hget(METRICS_KEY, "avg_duration")
    if raw is None:
        return "low"
    try:
        avg = float(raw)
    except ValueError:
        return "low"
    if avg <= 0:
        return "low"
    if position <= 3:
        return "medium"
    return "low"


def queue_timing_payload(position: int | None) -> Dict:
    eta = estimate_eta_seconds(position)
    avg = average_run_duration_seconds()
    first_frame_avg = average_first_frame_seconds()
    return {
        "queue_position": position,
        "eta_seconds": eta,
        "eta_confidence": estimate_eta_confidence(position, eta),
        "active_remaining_seconds": estimate_active_remaining_seconds(avg),
        "avg_run_seconds": int(math.ceil(avg)),
        "avg_first_frame_seconds": int(math.ceil(first_frame_avg)),
        "server_time": int(now_ts()),
    }


def client_rate_key(prefix: str = "start") -> str:
    client = get_remote_address() or "unknown"
    safe = re.sub(r"[^a-zA-Z0-9_.:-]", "_", client)
    return f"diffusion:rate:{prefix}:{safe}"


def adaptive_start_profile() -> Dict:
    queue_len = int(get_redis().llen(QUEUE_KEY))
    active_count = active_worker_count()
    if active_count == 0 and queue_len == 0:
        return {
            "demand": "low",
            "limit": IDLE_STARTS_PER_MINUTE,
            "window_seconds": 60,
            "queue_limit": MAX_QUEUED_RUNS,
        }
    if queue_len < max(1, MAX_QUEUED_RUNS // 2):
        return {
            "demand": "normal",
            "limit": NORMAL_STARTS_PER_MINUTE,
            "window_seconds": 60,
            "queue_limit": MAX_QUEUED_RUNS,
        }
    return {
        "demand": "busy",
        "limit": BUSY_STARTS_PER_MINUTE,
        "window_seconds": 60,
        "queue_limit": MAX_QUEUED_RUNS,
    }


def adaptive_start_admission() -> tuple[bool, Dict]:
    profile = adaptive_start_profile()
    if not ADAPTIVE_RATE_LIMITS:
        return True, profile

    now = now_ts()
    window = int(profile["window_seconds"])
    key = client_rate_key("start")
    r = get_redis()
    r.zremrangebyscore(key, 0, now - window)
    count = int(r.zcard(key))
    limit = int(profile["limit"])
    if count >= limit:
        oldest = r.zrange(key, 0, 0, withscores=True)
        if oldest:
            retry_after = max(1, int(math.ceil((float(oldest[0][1]) + window) - now)))
        else:
            retry_after = max(1, min(15, window))
        profile["retry_after"] = retry_after
        profile["remaining"] = 0
        return False, profile
    r.zadd(key, {uuid.uuid4().hex: now})
    r.expire(key, window + 120)
    profile["retry_after"] = 0
    profile["remaining"] = max(0, limit - count - 1)
    return True, profile


def parse_gif_size(gif_size_text: str) -> tuple[int, int]:
    default = (1400, 900)
    if not gif_size_text:
        return default
    try:
        w, h = gif_size_text.lower().split("x")
        width = clamp_int(int(w), 320, 1920)
        height = clamp_int(int(h), 240, 1080)
        return (width, height)
    except Exception:
        return default


def set_run_terminal(run_id: str, state: str, status: str) -> None:
    redis_update_run(run_id, state=state, status=status, done=True, finished_at=now_ts())


def get_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_dtype(device: torch.device) -> tuple[torch.dtype, str]:
    requested = INFERENCE_DTYPE
    if requested == "auto":
        requested = "fp16" if device.type in ("cuda", "mps") else "fp32"
    if requested == "fp16":
        if device.type == "cpu":
            return torch.float32, "fp32(cpu-fallback)"
        return torch.float16, "fp16"
    if requested == "bf16":
        if device.type == "cpu":
            return torch.bfloat16, "bf16"
        if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        return torch.float32, "fp32(fallback)"
    return torch.float32, "fp32"


def format_param_count(n: int | None) -> str:
    if not n or n <= 0:
        return "unknown"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def get_model_info(model_dir: str) -> Dict:
    cached = MODEL_INFO_CACHE.get(model_dir)
    if cached:
        return cached
    info = {
        "param_count": None,
        "param_count_display": "unknown",
        "architecture": "unknown",
        "hidden_size": None,
        "num_hidden_layers": None,
    }
    try:
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        architectures = config.get("architectures") or []
        if architectures:
            info["architecture"] = str(architectures[0])
        info["hidden_size"] = config.get("hidden_size")
        info["num_hidden_layers"] = config.get("num_hidden_layers")
    except (OSError, json.JSONDecodeError):
        pass
    MODEL_INFO_CACHE[model_dir] = info
    return info


def validate_model_dir(model_dir: str) -> tuple[bool, str]:
    cached = MODEL_VALIDATION_CACHE.get(model_dir)
    if cached:
        return cached["ok"], cached["msg"]
    tokenizer = None
    tok_err = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception as e:
        tok_err = e
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        except Exception as e_slow:
            ok, msg = False, (
                f"failed loading tokenizer from {model_dir}: {e_slow}. "
                f"fast-tokenizer error was: {tok_err}"
            )
            MODEL_VALIDATION_CACHE[model_dir] = {"ok": ok, "msg": msg}
            return ok, msg

    required = ["<PROMPT>", "</PROMPT>", "<CONTEXT>", "</CONTEXT>", "<SQL>", "</SQL>"]
    vocab = tokenizer.get_vocab()
    missing = [t for t in required if t not in vocab]
    if missing:
        ok, msg = True, (
            f"tokenizer missing special tokens {missing}; "
            "inference will add them dynamically to match train.py behavior."
        )
        MODEL_VALIDATION_CACHE[model_dir] = {"ok": ok, "msg": msg}
        return ok, msg

    weight_candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]
    has_weights = any(os.path.isfile(os.path.join(model_dir, name)) for name in weight_candidates)
    if not has_weights:
        ok, msg = False, (
            f"model weights not found in {model_dir}. "
            "Expected one of: model.safetensors, pytorch_model.bin, pytorch_model.bin.index.json. "
            "Mount the trained diffusion model output directory."
        )
        MODEL_VALIDATION_CACHE[model_dir] = {"ok": ok, "msg": msg}
        return ok, msg

    ok, msg = True, "ok"
    MODEL_VALIDATION_CACHE[model_dir] = {"ok": ok, "msg": msg}
    return ok, msg


def build_should_stop_cb(run_id: str, start_time: float):
    def _should_stop() -> bool:
        run = redis_get_run(run_id)
        if not run:
            return True
        if run.get("cancel_requested"):
            return True
        if (now_ts() - start_time) > REQUEST_TIMEOUT_SECONDS:
            redis_update_run(run_id, timed_out=True, cancel_requested=True)
            return True
        return False
    return _should_stop


def process_run(run_id: str) -> None:
    run = redis_get_run(run_id)
    if not run:
        return
    start_time = now_ts()
    r = get_redis()
    r.sadd(ACTIVE_SET_KEY, run_id)
    redis_update_run(run_id, state="running", status="running (loading model)", started_at=start_time)
    try:
        payload = run["payload"]
        should_stop = build_should_stop_cb(run_id, start_time)
        first_snapshot_seen = False

        def status_cb(msg):
            redis_update_run(run_id, status=str(msg))

        def on_snapshot_callback(snapshot_obj):
            nonlocal first_snapshot_seen
            if not first_snapshot_seen:
                first_snapshot_seen = True
                first_frame_elapsed = max(0.01, now_ts() - start_time)
                redis_update_run(run_id, first_snapshot_at=now_ts())
                update_metric_average("avg_first_frame", "first_frame_count", first_frame_elapsed)
            redis_append_snapshot(run_id, snapshot_obj)

        res = run_denoising_generation_callback(
            payload["prompt"],
            payload["context"],
            payload["model_dir"],
            n_steps=payload["steps"],
            max_len=payload["max_len"],
            top_k=payload["top_k"],
            top_p=payload["top_p"],
            sql_len_request=payload["sql_len"],
            animate=True,
            on_snapshot=on_snapshot_callback,
            status_cb=status_cb,
            deterministic_seed=DEFAULT_SEED,
            should_stop=should_stop,
        )
        final_sql = strip_final_masks(res.get("sql_only", ""))
        redis_update_run(run_id, sql_only=final_sql, display=res.get("display"))

        if ENABLE_GIF and not should_stop():
            os.makedirs(OUT_DIR, exist_ok=True)
            redis_update_run(run_id, status="building GIF")
            gif_size = parse_gif_size(payload["gif_size_text"])
            _count, snaps = redis_get_snapshot_delta(run_id, 0)
            gif_bytes = build_gif_bytes_from_snapshots(snaps, size=gif_size, interval_ms=500)
            out_name = secure_filename(f"generation_{run_id}.gif")
            out_path = os.path.join(OUT_DIR, out_name)
            with open(out_path, "wb") as f:
                f.write(gif_bytes)
            redis_update_run(run_id, gif_url=f"/gif/{out_name}", status=f"GIF written to {out_path}")

        if should_stop():
            timed_out = redis_get_run(run_id).get("timed_out", False)
            if timed_out:
                set_run_terminal(run_id, state="timed_out", status="timed out")
            else:
                set_run_terminal(run_id, state="stopped", status="stopped by user")
        else:
            set_run_terminal(run_id, state="done", status="done")
            elapsed = max(0.01, now_ts() - start_time)
            update_metric_average("avg_duration", "count", elapsed)
            _snap_count, cached_snaps = redis_get_snapshot_delta(run_id, 0)
            write_result_cache(payload, final_sql, res.get("display", ""), cached_snaps)
    except TimeoutError:
        set_run_terminal(run_id, state="timed_out", status="timed out")
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        set_run_terminal(run_id, state="error", status=f"error: {e}\n{tb}")
    finally:
        r.srem(ACTIVE_SET_KEY, run_id)


def worker_loop():
    r = get_redis()
    while True:
        if active_worker_count() >= MAX_CONCURRENT_RUNS:
            time.sleep(0.5)
            continue
        job = r.blpop(QUEUE_KEY, timeout=3)
        if not job:
            continue
        _queue_name, run_id = job
        run = redis_get_run(run_id)
        if not run:
            continue
        if run.get("cancel_requested"):
            set_run_terminal(run_id, state="stopped", status="stopped while queued")
            continue
        process_run(run_id)


def cleanup_loop():
    while True:
        time.sleep(60)
        cutoff = now_ts() - RUN_TTL_SECONDS
        if ENABLE_GIF and os.path.isdir(OUT_DIR):
            for name in os.listdir(OUT_DIR):
                if not name.endswith(".gif"):
                    continue
                path = os.path.join(OUT_DIR, name)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                except OSError:
                    pass

# -------------------------
# Utilities: GIF font fitting (bigger min size)
# -------------------------
def pick_font_and_size(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int, padding: int = 28):
    fonts_to_try = ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
    font_path = None
    for p in fonts_to_try:
        try:
            ImageFont.truetype(p, size=12)
            font_path = p
            break
        except Exception:
            font_path = None

    min_size, max_size = 16, 72
    lo, hi = min_size, max_size
    best = min_size
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, size=mid) if font_path else ImageFont.load_default()
        char_w = f.getsize("M")[0] or 8
        chars_per_line = max(20, int((max_width - padding*2) / char_w))
        wrapped = textwrap.fill(text, width=chars_per_line)
        bbox = draw.multiline_textbbox((0,0), wrapped, font=f, spacing=6)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w + padding*2 <= max_width and h + padding*2 <= max_height:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, size=best) if font_path else ImageFont.load_default()

def render_text_image_for_gif(text: str, width: int, height: int, padding: int = 28):
    img = Image.new("RGB", (width, height), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    font = pick_font_and_size(draw, text, width, height, padding=padding)
    char_w = font.getsize("M")[0] or 8
    chars_per_line = max(20, int((width - padding*2) / char_w))
    wrapped = textwrap.fill(text, width=chars_per_line)
    y = padding
    draw.multiline_text((padding, y), wrapped, fill=(18,18,18), font=font, spacing=6)
    return img

def build_gif_bytes_from_snapshots(snapshots: List[Dict], size=(1400,900), interval_ms: int = 120) -> bytes:
    """
    Smooth GIF builder with blur->unblur crossfade transitions between snapshots.

    Behavior:
      - Render each snapshot to an image using `render_text_image_for_gif`.
      - For each pair (A -> B), generate `subframes_per_transition` intermediate frames where:
          frame = blend(A, GaussianBlur(B, radius=blur_radius))
        with blur_radius decreasing across the subframes (unblurring).
      - Keep a small hold at the beginning and end.

    Tunables:
      - subframes_per_transition: more frames => smoother transition.
      - blur_max_radius: initial blur radius for the unblur effect.
      - hold_frames_start / hold_frames_end: number of replicated frames at start/end.
    """
    from PIL import ImageFilter

    # tuning — change these to taste
    subframes_per_transition = 6     # frames used to animate each snapshot -> next snapshot
    blur_max_radius = 14             # maximum blur radius applied to target frame at start of transition
    hold_frames_start = 2            # hold first snapshot (makes GIF less jumpy)
    hold_frames_end = 8              # hold final snapshot longer so user can read
    # interval_ms is the time per frame in milliseconds (keeps your function signature)

    # Helper to extract the body text from snapshot (same logic as before)
    def _body_from_snap(snap):
        s = snap.get("text") if isinstance(snap, dict) else str(snap)
        start = s.rfind("<SQL>")
        if start == -1:
            body = ""
        else:
            start += len("<SQL>")
            end = s.find("</SQL>", start)
            body = s[start:end].strip() if end != -1 else s[start:].strip()
        if not body.strip():
            body = "(empty)"
        return body

    # Render base images for each snapshot
    base_imgs = []
    for snap in snapshots:
        body = _body_from_snap(snap)
        img = render_text_image_for_gif(body, width=size[0], height=size[1], padding=28)
        # make sure full-RGB mode (P conversion later)
        base_imgs.append(img.convert("RGBA"))

    frames = []

    # If no snapshots, create a blank frame
    if not base_imgs:
        blank = Image.new("RGB", size, color=(255,255,255))
        frames.append(blank.convert("P", palette=Image.ADAPTIVE))
    else:
        # hold initial snapshot
        for _ in range(hold_frames_start):
            frames.append(base_imgs[0].convert("P", palette=Image.ADAPTIVE))

        # build transitions
        for i in range(len(base_imgs) - 1):
            A = base_imgs[i]
            B = base_imgs[i + 1]

            # If A == B visually (rare), just add a single frame
            if A.tobytes() == B.tobytes():
                frames.append(B.convert("P", palette=Image.ADAPTIVE))
                continue

            # Add one frame of A (a short pause)
            frames.append(A.convert("P", palette=Image.ADAPTIVE))

            # Create subframes that unblur B from heavy blur -> sharp while crossfading
            for j in range(subframes_per_transition):
                f = (j + 1) / float(subframes_per_transition + 1)  # 0..1
                # progressive blur radius: starts at blur_max_radius and goes to 0
                blur_r = blur_max_radius * (1.0 - f)
                B_blurred = B.filter(ImageFilter.GaussianBlur(radius=blur_r))

                # blend A and B_blurred; as f -> 1 we show more of B
                blended = Image.blend(A, B_blurred, alpha=f)

                # Convert to P (palette) to keep GIF size reasonable
                frames.append(blended.convert("P", palette=Image.ADAPTIVE))

            # finally add one sharp B frame
            frames.append(B.convert("P", palette=Image.ADAPTIVE))

        # hold final snapshot a bit longer so user can read it
        for _ in range(hold_frames_end):
            frames.append(base_imgs[-1].convert("P", palette=Image.ADAPTIVE))

    # save frames to bytes
    bio = io.BytesIO()
    # Make sure we pass duration per frame; keep loop=0 to repeat forever
    frames[0].save(bio, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=interval_ms)
    bio.seek(0)
    return bio.read()


# -------------------------
# Beautiful square share-GIF (on-demand export)
# -------------------------
_FONT_CACHE: Dict[tuple, object] = {}
_MONO_FONT_PATHS = [
    ("/System/Library/Fonts/Menlo.ttc", 0),
    ("/System/Library/Fonts/SFNSMono.ttf", 0),
    ("/System/Library/Fonts/Monaco.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
    ("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", 0),
    ("DejaVuSansMono.ttf", 0),
]
_SANS_FONT_PATHS = [
    ("/System/Library/Fonts/SFNS.ttf", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ("DejaVuSans.ttf", 0),
]


def _load_font(size: int, mono: bool = True):
    size = max(8, int(size))
    key = (mono, size)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    for path, idx in (_MONO_FONT_PATHS if mono else _SANS_FONT_PATHS):
        try:
            font = ImageFont.truetype(path, size=size, index=idx)
            _FONT_CACHE[key] = font
            return font
        except Exception:
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


_GIF_THEME = {
    "bg": (11, 15, 25), "panel": (17, 23, 37), "border": (33, 42, 62),
    "text": (226, 232, 240), "muted": (120, 134, 156),
    "keyword": (96, 165, 250), "fn": (192, 132, 252), "str": (74, 222, 128),
    "punct": (148, 163, 184), "mask": (52, 66, 92), "accent": (56, 189, 248),
    "track": (30, 41, 59),
}
_GIF_KW = {"SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET",
    "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "FULL", "ON", "AND", "OR", "NOT", "IN", "AS",
    "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "DISTINCT",
    "UNION", "ALL", "ASC", "DESC", "BETWEEN", "LIKE", "IS", "NULL", "EXISTS", "CASE", "WHEN",
    "THEN", "ELSE", "END"}
_GIF_FN = {"COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "ABS", "COALESCE", "UPPER", "LOWER",
    "LENGTH", "NOW", "DATE", "CAST"}
_GIF_TOKEN_SPLIT = re.compile(r"(\s+|_{2,}|[(),*=<>!;.])")
_GIF_CLAUSE_RE = re.compile(r"\s+\b(FROM|WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|UNION(?: ALL)?|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN|OUTER JOIN|JOIN|VALUES|SET)\b", re.IGNORECASE)
_GIF_BOOL_RE = re.compile(r"\s+\b(AND|OR)\b", re.IGNORECASE)


def format_sql_lines(sql: str) -> List[str]:
    """Clause-per-line layout mirroring the live UI formatter (quote-aware)."""
    if not sql or not sql.strip():
        return ["…"]
    base = re.sub(r"\s+", " ", sql).strip()
    parts = re.split(r"('(?:[^']|'')*'?)", base)
    for i in range(0, len(parts), 2):
        seg = _GIF_CLAUSE_RE.sub(lambda m: "\n" + m.group(1), parts[i])
        seg = _GIF_BOOL_RE.sub(lambda m: "\n  " + m.group(1), seg)
        parts[i] = seg
    return "".join(parts).split("\n")


def _gif_classify(tok: str) -> str:
    t = tok.strip()
    if not t:
        return "ws"
    if re.fullmatch(r"_{2,}", t):
        return "mask"
    up = t.upper()
    if up in _GIF_KW:
        return "keyword"
    if up in _GIF_FN:
        return "fn"
    if t[0] in "'\"":
        return "str"
    if re.fullmatch(r"[(),*=<>!;.+/-]+", t):
        return "punct"
    return "text"


def _wrap_text(text: str, font, max_w: float, max_lines: int = 2) -> List[str]:
    lines, cur = [], ""
    for w in (text or "").split():
        trial = (cur + " " + w).strip()
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and font.getlength(lines[-1] + "…") > max_w:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "…"
    return lines or [""]


def _fit_mono_size(lines: List[str], max_w: float, max_h: float, lo: int = 16, hi: int = 56, leading: float = 1.5) -> int:
    longest = max((len(l) for l in lines), default=1) or 1
    n = max(len(lines), 1)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = _load_font(mid, mono=True)
        cw = f.getlength("M") or mid * 0.6
        asc, desc = f.getmetrics()
        if cw * longest <= max_w and (asc + desc) * leading * n <= max_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_share_frame(prompt: str, sql: str, step: int, total: int, size: int = 800):
    T = _GIF_THEME
    img = Image.new("RGB", (size, size), T["bg"])
    d = ImageDraw.Draw(img)
    pad = int(size * 0.085)
    d.rounded_rectangle([pad * 0.5, pad * 0.5, size - pad * 0.5, size - pad * 0.5],
                        radius=int(size * 0.035), fill=T["panel"], outline=T["border"], width=2)
    x0 = pad
    inner_w = size - 2 * pad

    # header — brand + prompt
    brand_f = _load_font(int(size * 0.026), mono=False)
    y = pad * 0.95
    d.text((x0, y), "text-diffusion", font=brand_f, fill=T["accent"])
    d.text((x0 + brand_f.getlength("text-diffusion") + 10, y + 2),
           "natural language → SQL", font=_load_font(int(size * 0.021), mono=False), fill=T["muted"])
    y += brand_f.size * 1.9
    pf = _load_font(int(size * 0.030), mono=False)
    for ln in _wrap_text(prompt or "(no prompt)", pf, inner_w, max_lines=2):
        d.text((x0, y), ln, font=pf, fill=T["text"])
        y += pf.size * 1.32
    header_bottom = y + size * 0.015
    d.line([(x0, header_bottom), (size - pad, header_bottom)], fill=T["border"], width=2)

    # footer — progress bar + step label
    foot_h = size * 0.085
    foot_top = size - pad - foot_h
    sf = _load_font(int(size * 0.023), mono=False)
    frac = max(0.0, min(1.0, (step / total) if total else 1.0))
    bar_h = max(8, int(size * 0.012))
    bar_y = size - pad - foot_h * 0.30
    d.rounded_rectangle([x0, bar_y, x0 + inner_w, bar_y + bar_h], radius=bar_h // 2, fill=T["track"])
    if frac > 0:
        d.rounded_rectangle([x0, bar_y, x0 + max(bar_h, inner_w * frac), bar_y + bar_h], radius=bar_h // 2, fill=T["accent"])
    d.text((x0, bar_y - sf.size * 1.7),
           "done" if frac >= 1 else f"denoising · step {step}/{total}", font=sf, fill=T["muted"])

    # body — formatted SQL with syntax colours + mask blocks
    body_top = header_bottom + size * 0.04
    body_h = foot_top - body_top - size * 0.02
    lines = format_sql_lines(sql)
    fsize = _fit_mono_size(lines, inner_w, body_h)
    f = _load_font(fsize, mono=True)
    cw = f.getlength("M") or fsize * 0.6
    asc, desc = f.getmetrics()
    lh = (asc + desc) * 1.5
    y = body_top
    for line in lines:
        x = x0
        for tok in _GIF_TOKEN_SPLIT.split(line):
            if tok == "":
                continue
            cls = _gif_classify(tok)
            w = cw * len(tok)
            if cls == "mask":
                d.rounded_rectangle([x + 2, y + asc * 0.16, x + w - 3, y + asc * 1.04],
                                    radius=max(3, int(fsize * 0.16)), fill=T["mask"])
            elif cls != "ws":
                d.text((x, y), tok, font=f, fill=T.get(cls, T["text"]))
            x += w
        y += lh
    return img


def build_share_gif(snapshots: List[Dict], prompt: str, size: int = 800) -> bytes:
    """Square, syntax-highlighted, shareable GIF of the denoising reveal."""
    seq, last = [], None
    for snap in snapshots:
        sql = (snap.get("sql_only") or "").strip()
        if sql == last:
            continue
        last = sql
        seq.append((sql, int(snap.get("step", 0) or 0), int(snap.get("total_steps", 1) or 1)))
    if not seq:
        seq = [("", 0, 1)]
    rgb_frames = [render_share_frame(prompt, sql, st, tot, size=size) for (sql, st, tot) in seq]
    # shared palette derived from the final (most colourful) frame -> clean, small GIF
    pal = rgb_frames[-1].convert("P", palette=Image.ADAPTIVE, colors=128)
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb_frames]
    durations = [260] * len(frames)
    durations[0] = 700        # brief intro hold
    durations[-1] = 2600      # hold final result so it's readable
    bio = io.BytesIO()
    frames[0].save(bio, format="GIF", save_all=True, append_images=frames[1:],
                   loop=0, duration=durations, disposal=2, optimize=True)
    bio.seek(0)
    return bio.read()


# -------------------------
# Model load helper (cached)
# -------------------------
def load_model_and_tokenizer(model_dir: str, max_len: int = 512):
    if MODEL_CACHE["dir"] == model_dir and MODEL_CACHE["tokenizer"] is not None and MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["tokenizer"], MODEL_CACHE["model"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    tokenizer.model_max_length = max_len
    device = get_inference_device()
    dtype, _dtype_label = resolve_model_dtype(device)
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, torch_dtype=dtype)
    except Exception:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, torch_dtype=torch.float32)
        dtype = torch.float32
    model.to(device)
    model.eval()
    MODEL_CACHE.update({"dir": model_dir, "tokenizer": tokenizer, "model": model})
    return tokenizer, model

# -------------------------
# Core denoising with callback for snapshots (reports step)
# -------------------------
def run_denoising_generation_callback(
    prompt_text: str,
    context_text: str,
    model_dir: str,
    n_steps: int = 11,
    max_len: int = 512,
    top_k: int = 1,
    top_p: float = 1,
    sql_len_request: int = 128,
    animate: bool = True,
    on_snapshot = None,
    status_cb = None,
    deterministic_seed: int = DEFAULT_SEED,
    should_stop=None,
) -> Dict:
    def log(s):
        if status_cb:
            status_cb(str(s))

    # deterministic seeds
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed)
    torch.manual_seed(deterministic_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(deterministic_seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    log(f"[INFO] Loading model/tokenizer from {model_dir}")
    tokenizer, model = load_model_and_tokenizer(model_dir, max_len)

    REQUIRED_TAGS = ["<PROMPT>", "</PROMPT>", "<CONTEXT>", "</CONTEXT>", "<SQL>", "</SQL>"]
    missing = [t for t in REQUIRED_TAGS if t not in tokenizer.get_vocab()]
    if missing:
        log(f"[INFO] Adding missing special tokens at inference-time: {missing}")
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
        log("[INFO] Resized model embeddings for added tokens")

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Build the input EXACTLY like training's encode_text (train.py): manual token
    # assembly with [CLS], single-token tags (no inter-tag spaces) and a fixed
    # mask-filled SQL window padded to max_len. The earlier string-based
    # construction did not match the training distribution and produced corrupt SQL.
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    tag_id = {t: tokenizer.convert_tokens_to_ids(t) for t in REQUIRED_TAGS}

    sql_len = max(1, min(int(sql_len_request), max_len - 9))
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    budget = max_len - sql_len - 9  # CLS + 6 tags + SEP (+1 slack); mirrors training
    if len(prompt_ids) + len(context_ids) > budget:
        context_ids = context_ids[: max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget]

    head = (
        [cls_id, tag_id["<PROMPT>"]] + prompt_ids
        + [tag_id["</PROMPT>"], tag_id["<CONTEXT>"]] + context_ids
        + [tag_id["</CONTEXT>"], tag_id["<SQL>"]]
    )
    sql_open_idx = len(head) - 1         # index of the <SQL> tag
    sql_start = len(head)                # first maskable SQL position
    ids = head + [mask_id] * sql_len + [tag_id["</SQL>"], sep_id]
    sql_close_idx = sql_start + sql_len  # index of the </SQL> tag
    attn = [1] * len(ids) + [0] * (max_len - len(ids))
    ids = ids + [pad_id] * (max_len - len(ids))

    current_ids = torch.tensor([ids], dtype=torch.long)
    current_attention = torch.tensor([attn], dtype=torch.long)
    log(f"[INFO] SQL window: {sql_len} mask tokens at positions {sql_start}..{sql_close_idx}")

    device = next(model.parameters()).device
    current_ids = current_ids.to(device)
    current_attention = current_attention.to(device)

    modifiable = torch.zeros((max_len,), dtype=torch.bool, device=device)
    modifiable[sql_open_idx + 1 : sql_close_idx] = True
    modifiable_positions = torch.nonzero(modifiable, as_tuple=False).squeeze(-1).tolist()
    if isinstance(modifiable_positions, int):
        modifiable_positions = [modifiable_positions]
    if len(modifiable_positions) == 0:
        raise RuntimeError("No modifiable SQL tokens were found (empty SQL region).")

    pad_token = tokenizer.pad_token or "<pad>"
    total_steps = max(1, min(n_steps, len(modifiable_positions)))

    def snapshot_text():
        s = tokenizer.decode(current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
        return s.replace(mask_token, " ____").replace(pad_token, "")

    def render_sql_window():
        """Animation-friendly view of the SQL span in *content space*.

        Real SQL tokens form a prefix; the SQL window is PAD-filled out to 128
        slots. Rendering all 128 means most denoising steps only commit PAD ->
        empty, producing dead frames. So we trim everything past the last
        committed real token (the "query ended here" region) and show the
        still-masked content positions as ``____`` glyphs that pop in by
        confidence — the query materialises within its real length.
        """
        window = current_ids[0, sql_open_idx + 1 : sql_close_idx].tolist()
        content_end = 0
        for i, tid in enumerate(window):
            if tid != mask_id and tid != pad_id:
                content_end = i + 1
        if content_end == 0:
            # Length not decided yet: short pending field instead of 128 slots.
            pending = min(sum(1 for tid in window if tid == mask_id), 6)
            return " ".join(["____"] * pending)
        body = [tid for tid in window[:content_end] if tid != pad_id]
        s = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        return s.replace(mask_token, " ____").strip()

    snapshots: List[Dict] = []
    last_render = None
    if animate:
        last_render = render_sql_window()
        snapshots.append({"text": snapshot_text(), "sql_only": last_render, "step": 0, "total_steps": total_steps})
        if on_snapshot:
            on_snapshot(snapshots[-1])

    log("[INFO] Starting denoising (confidence-based unmasking)")
    t0 = time.time()
    # top_k <= 1 -> fully greedy commit order; otherwise add Gumbel noise for diversity
    sample_temperature = 0.0 if int(top_k) <= 1 else 1.0
    vocab = tokenizer.get_vocab()
    forbid_ids = [vocab[t] for t in REQUIRED_TAGS if t in vocab]

    for step_idx, total_steps, current_ids in denoise_steps(
        model,
        current_ids,
        current_attention,
        modifiable_positions,
        mask_id,
        n_steps=total_steps,
        temperature=sample_temperature,
        forbid_token_ids=forbid_ids,
        should_stop=should_stop,
    ):
        if status_cb:
            status_cb(f"step {step_idx+1}/{total_steps}")
        if animate and step_idx + 1 < total_steps:
            r = render_sql_window()
            # Dedupe: skip frames where the visible query didn't change (these are
            # the PAD-only commit steps that previously rendered as no-ops).
            if r != last_render:
                last_render = r
                snapshots.append({"text": snapshot_text(), "sql_only": r, "step": step_idx+1, "total_steps": total_steps})
                if on_snapshot:
                    on_snapshot(snapshots[-1])

    final_token_slice = current_ids[0, sql_open_idx + 1 : sql_close_idx].detach().cpu().tolist()
    final_sql_only = strip_final_masks(decode_sql_from_token_ids(tokenizer, final_token_slice, mask_id=mask_id, pad_id=pad_id))
    if animate:
        s_clean_final = replace_sql_section(snapshot_text(), final_sql_only)
        snapshots.append({"text": s_clean_final, "sql_only": final_sql_only, "step": total_steps, "total_steps": total_steps})
        if on_snapshot:
            on_snapshot(snapshots[-1])

    t1 = time.time()
    log(f"[INFO] Denoising took {t1 - t0:.2f}s")

    decoded_full = tokenizer.decode(current_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    display = decoded_full.replace(mask_token, "_____").replace("<pad>", "")
    display = display.replace("_____", "")
    token_slice = current_ids[0, sql_open_idx + 1 : sql_close_idx].detach().cpu().tolist()
    sql_only = decode_sql_from_token_ids(tokenizer, token_slice, mask_id=mask_id, pad_id=pad_id)

    sql_only = strip_final_masks(sql_only)

    if not animate:
        snapshots = [{"text": display, "sql_only": sql_only, "step": total_steps, "total_steps": total_steps}]

    return {"snapshots": snapshots, "sql_only": sql_only, "display": display}

# -------------------------
# Flask endpoints
# -------------------------
@app.errorhandler(429)
def ratelimit_handler(e):
    resp = jsonify({"error": "rate_limited", "message": str(e.description)})
    retry_after = getattr(e, "retry_after", None)
    if retry_after is not None:
        try:
            resp.headers["Retry-After"] = str(int(math.ceil(float(retry_after))))
        except Exception:
            pass
    return resp, 429


@app.errorhandler(Exception)
def json_http_errors(e):
    if isinstance(e, HTTPException):
        return e
    return jsonify({"error": "internal_error", "message": str(e)}), 500


@app.route("/healthz")
def healthz():
    try:
        get_redis().ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    demand = adaptive_start_profile() if redis_ok else {}
    return jsonify({
        "ok": redis_ok,
        "redis_ok": redis_ok,
        "standalone_mode": STANDALONE_MODE,
        "rate_limit_storage": RATELIMIT_STORAGE_URI,
        "adaptive_rate_limits": ADAPTIVE_RATE_LIMITS,
        "result_cache_enabled": RESULT_CACHE_ENABLED,
        "result_cache_ttl_seconds": RESULT_CACHE_TTL_SECONDS,
        "result_cache_version": RESULT_CACHE_VERSION,
        "demand": demand.get("demand"),
        "start_limit_per_minute": demand.get("limit"),
        "active_limit": MAX_CONCURRENT_RUNS,
        "queue_limit": MAX_QUEUED_RUNS,
        "queue_length": get_redis().llen(QUEUE_KEY) if redis_ok else 0,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "avg_run_seconds": int(math.ceil(average_run_duration_seconds())) if redis_ok else None,
        "avg_first_frame_seconds": int(math.ceil(average_first_frame_seconds())) if redis_ok else None,
        "max_steps": MAX_STEPS,
        "max_max_len": MAX_MAX_LEN,
        "max_sql_len": MAX_SQL_LEN,
        "active_worker_count": active_worker_count() if redis_ok else 0,
    }), (200 if redis_ok else 503)


@app.route("/")
def index():
    default_prompt = "which pirate found the most treasure?"
    default_context = "CREATE TABLE treasure_finds (pirate TEXT, coins INT, island TEXT)"
    model_info = get_model_info(DEFAULT_MODEL_DIR)
    effective_dtype = resolve_model_dtype(get_inference_device())[1]
    static_js_path = os.path.join(os.path.dirname(__file__), "static", "inference.js")
    try:
        static_version = str(int(os.path.getmtime(static_js_path)))
    except OSError:
        static_version = str(int(time.time()))
    return render_template(
        "inference.html",
        prompt_prefill=args.prompt or default_prompt,
        context_prefill=args.context or default_context,
        model_dir=DEFAULT_MODEL_DIR or "",
        max_model_len=MAX_MAX_LEN,
        max_steps=MAX_STEPS,
        default_steps=min(24, MAX_STEPS),
        max_sql_len=MAX_SQL_LEN,
        default_sql_len=max(1, min(128, MAX_SQL_LEN)),
        max_prompt_chars=MAX_PROMPT_CHARS,
        max_context_chars=MAX_CONTEXT_CHARS,
        model_param_count=model_info.get("param_count"),
        model_param_count_display=model_info.get("param_count_display", "unknown"),
        inference_dtype_requested=INFERENCE_DTYPE,
        inference_dtype_effective=effective_dtype,
        static_version=static_version,
    )


@app.route("/start", methods=["POST"])
@limiter.limit(START_RUN_RATE_LIMIT)
def start_run():
    form = request.form
    prompt = form.get("prompt", "").strip()
    context = form.get("context", "").strip()
    model_dir = form.get("model_dir", DEFAULT_MODEL_DIR).strip() or DEFAULT_MODEL_DIR
    gif_size_text = form.get("gif_size", "").strip()
    try:
        steps = int(form.get("steps", "24"))
        max_len = int(form.get("max_len", "512"))
        sql_len = int(form.get("sql_len", "128"))
        top_k = int(form.get("top_k", "1"))
        top_p = float(form.get("top_p", "1"))
    except ValueError:
        return jsonify({"error": "invalid_input", "message": "steps/max_len/sql_len/top_k/top_p must be numeric"}), 400

    if not prompt:
        return jsonify({"error": "invalid_input", "message": "Prompt is empty"}), 400
    if len(prompt) > MAX_PROMPT_CHARS:
        return jsonify({"error": "invalid_input", "message": f"Prompt too long (>{MAX_PROMPT_CHARS} chars)"}), 400
    if len(context) > MAX_CONTEXT_CHARS:
        return jsonify({"error": "invalid_input", "message": f"Context too long (>{MAX_CONTEXT_CHARS} chars)"}), 400
    if not os.path.isdir(model_dir):
        return jsonify({"error": "invalid_input", "message": f"Model dir not found: {model_dir}"}), 400

    if steps < 1 or steps > MAX_STEPS:
        return jsonify({"error": "invalid_input", "message": f"steps must be 1..{MAX_STEPS}"}), 400
    if max_len < 8 or max_len > MAX_MAX_LEN:
        return jsonify({"error": "invalid_input", "message": f"max_len must be 8..{MAX_MAX_LEN}"}), 400
    if sql_len < 1 or sql_len > MAX_SQL_LEN:
        return jsonify({"error": "invalid_input", "message": f"sql_len must be 1..{MAX_SQL_LEN}"}), 400
    if top_k < 0 or top_k > 200:
        return jsonify({"error": "invalid_input", "message": "top_k must be 0..200"}), 400
    if top_p < 0 or top_p > 1:
        return jsonify({"error": "invalid_input", "message": "top_p must be in [0, 1]"}), 400

    limits = compute_char_heuristic_limits(max_len=max_len, sql_len=sql_len)
    prompt_cap = limits["prompt_char_cap"]
    context_cap = limits["context_char_cap"]
    combined_cap = limits["combined_char_budget"]
    if len(prompt) > prompt_cap:
        return jsonify({
            "error": "invalid_input",
            "message": f"Prompt too long for current settings ({len(prompt)}>{prompt_cap} chars; max_len={max_len}, sql_len={sql_len}).",
        }), 400
    if len(context) > context_cap:
        return jsonify({
            "error": "invalid_input",
            "message": f"Context too long for current settings ({len(context)}>{context_cap} chars; max_len={max_len}, sql_len={sql_len}).",
        }), 400
    if (len(prompt) + len(context)) > combined_cap:
        return jsonify({
            "error": "invalid_input",
            "message": (
                f"Prompt+Context too long for current settings "
                f"({len(prompt) + len(context)}>{combined_cap} chars; "
                f"prompt_cap={prompt_cap}, context_cap={context_cap}, max_len={max_len}, sql_len={sql_len})."
            ),
        }), 400

    payload = {
        "prompt": prompt,
        "context": context,
        "steps": steps,
        "max_len": max_len,
        "sql_len": sql_len,
        "top_k": top_k,
        "top_p": top_p,
        "model_dir": model_dir,
        "gif_size_text": gif_size_text,
    }
    cached = read_result_cache(payload)
    if cached:
        run_id = create_cached_run(payload, cached)
        snap_count, _snaps = redis_get_snapshot_delta(run_id, 0)
        return jsonify({
            "run_id": run_id,
            "state": "done",
            "done": True,
            "cache_hit": True,
            "snapshot_count": snap_count,
            "queue_position": None,
            "eta_seconds": 0,
            "eta_confidence": "high",
            "queue_token": run_id,
        })

    model_ok, model_msg = validate_model_dir(model_dir)
    if not model_ok:
        return jsonify({"error": "invalid_model", "message": model_msg}), 400

    admission_preview = adaptive_start_profile()
    queue_limit = int(admission_preview.get("queue_limit", MAX_QUEUED_RUNS))
    r = get_redis()
    if r.llen(QUEUE_KEY) >= queue_limit:
        retry_after = max(3, estimate_active_remaining_seconds())
        return jsonify({
            "error": "queue_full",
            "message": "Server is busy. Try again shortly.",
            "max_queued_runs": queue_limit,
            "demand": admission_preview.get("demand"),
            "retry_after": retry_after,
        }), 503

    admitted, admission = adaptive_start_admission()
    if not admitted:
        retry_after = int(admission.get("retry_after", 10))
        resp = jsonify({
            "error": "rate_limited",
            "message": "Server is busy. Please retry shortly.",
            "demand": admission.get("demand"),
            "retry_after": retry_after,
        })
        resp.headers["Retry-After"] = str(retry_after)
        return resp, 429

    run_id = uuid.uuid4().hex
    redis_set_run(run_id, {
        "state": "queued",
        "status": "queued",
        "done": False,
        "timed_out": False,
        "cancel_requested": False,
        "sql_only": "",
        "display": "",
        "gif_url": "",
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "started_at": "",
        "finished_at": "",
        "snapshot_count": 0,
        "payload": payload,
    })
    r.rpush(QUEUE_KEY, run_id)
    pos = queue_position(run_id)
    if pos is None:
        pos = 1
    timing = queue_timing_payload(pos)
    return jsonify({
        "run_id": run_id,
        "state": "queued",
        "demand": admission.get("demand"),
        "start_limit_remaining": admission.get("remaining"),
        "queue_token": run_id,
        **timing,
    })


@app.route("/stream/<run_id>")
@limiter.exempt
def stream(run_id):
    run = redis_get_run(run_id)
    if not run:
        return jsonify({"error": "not_found", "message": "run_id not found"}), 404

    def event_stream():
        cursor = 0
        last_status = ""
        done_sent = False
        last_heartbeat = time.time()
        while True:
            _, snaps = redis_get_snapshot_delta(run_id, cursor)
            for snap in snaps:
                yield f"event: snapshot\\ndata: {json.dumps(snap)}\\n\\n"
                cursor += 1

            run_now = redis_get_run(run_id)
            if not run_now:
                break
            st = run_now.get("status", "")
            if st != last_status:
                last_status = st
                yield f"event: status\\ndata: {json.dumps({'msg': st})}\\n\\n"

            is_done = run_now.get("done")
            payload = {
                "sql_only": run_now.get("sql_only"),
                "gif_url": run_now.get("gif_url"),
                "state": run_now.get("state"),
                "status": run_now.get("status"),
            }
            if is_done and not done_sent:
                yield f"event: done\\ndata: {json.dumps(payload)}\\n\\n"
                done_sent = True
                break

            time.sleep(0.08)
            now = time.time()
            if now - last_heartbeat >= 10:
                yield ": keepalive\\n\\n"
                last_heartbeat = now

    resp = Response(stream_with_context(event_stream()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    return resp


@app.route("/run/<run_id>")
@limiter.exempt
def run_state(run_id):
    after = request.args.get("after", "0")
    try:
        after_idx = max(0, int(after))
    except ValueError:
        after_idx = 0
    run = redis_get_run(run_id)
    if not run:
        return jsonify({"error": "not_found", "message": "run_id not found"}), 404
    snap_count, delta = redis_get_snapshot_delta(run_id, after_idx)
    pos = queue_position(run_id) if run.get("state") == "queued" else None
    if run.get("state") == "queued" and pos is None:
        pos = 1
    timing = queue_timing_payload(pos)
    demand = adaptive_start_profile()
    return jsonify({
        "run_id": run_id,
        "state": run.get("state"),
        "status": run.get("status"),
        "done": bool(run.get("done")),
        "sql_only": run.get("sql_only"),
        "gif_url": run.get("gif_url"),
        "cache_hit": bool(run.get("cache_hit")),
        "snapshot_count": snap_count,
        "snapshots": delta,
        "demand": demand.get("demand"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        **timing,
    })


@app.route("/queue/<run_id>")
@limiter.exempt
def queue_state(run_id):
    run = redis_get_run(run_id)
    if not run:
        return jsonify({"error": "not_found", "message": "run_id not found"}), 404
    pos = queue_position(run_id) if run.get("state") == "queued" else None
    if run.get("state") == "queued" and pos is None:
        pos = 1
    timing = queue_timing_payload(pos)
    demand = adaptive_start_profile()
    return jsonify({
        "run_id": run_id,
        "state": run.get("state"),
        "demand": demand.get("demand"),
        "active_worker_count": active_worker_count(),
        **timing,
    })


@app.route("/gif/<filename>")
def serve_gif(filename):
    if not ENABLE_GIF:
        return jsonify({"error": "gif_disabled", "message": "GIF generation is disabled on this server"}), 404
    safe_name = secure_filename(filename)
    p = os.path.join(OUT_DIR, safe_name)
    if not os.path.isfile(p):
        return jsonify({"error": "not_found", "message": "file not found"}), 404
    try:
        resp = send_file(p, mimetype="image/gif", as_attachment=True, download_name=safe_name, max_age=0)
    except TypeError:
        resp = send_file(p, mimetype="image/gif", as_attachment=True, attachment_filename=safe_name)
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.route("/export_gif/<run_id>")
@limiter.limit(EXPORT_GIF_RATE_LIMIT)
def export_gif(run_id):
    """On-demand: render the run's denoising frames into a square share-GIF.

    Independent of --enable-gif (which gates the inline worker GIF); this builds
    only when the user clicks Export, from the snapshots already stored.
    """
    run = redis_get_run(run_id)
    if not run:
        return jsonify({"error": "not_found", "message": "run_id not found"}), 404
    _, snaps = redis_get_snapshot_delta(run_id, 0)
    if not snaps:
        return jsonify({"error": "no_frames", "message": "No animation frames available for this run."}), 400
    prompt = (run.get("payload") or {}).get("prompt", "")
    try:
        gif_bytes = build_share_gif(snaps, prompt, size=800)
    except Exception as exc:
        return jsonify({"error": "render_failed", "message": f"GIF render failed: {exc}"}), 500
    name = f"text2sql-diffusion-{run_id[:8]}.gif"
    try:
        resp = send_file(io.BytesIO(gif_bytes), mimetype="image/gif", as_attachment=True, download_name=name, max_age=0)
    except TypeError:
        resp = send_file(io.BytesIO(gif_bytes), mimetype="image/gif", as_attachment=True, attachment_filename=name)
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Content-Length"] = str(len(gif_bytes))
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.route("/stop/<run_id>", methods=["POST"])
def stop_run(run_id):
    run = redis_get_run(run_id)
    if not run:
        return jsonify({"error": "not_found", "message": "run_id not found"}), 404
    r = get_redis()
    removed = r.lrem(QUEUE_KEY, 1, run_id)
    if removed > 0:
        redis_update_run(run_id, state="stopped", status="stopped while queued", cancel_requested=True, done=True, finished_at=now_ts())
    else:
        redis_update_run(run_id, state="stopping", status="stopping...", cancel_requested=True)
    return jsonify({"ok": True, "run_id": run_id})


# -------------------------
# Start server
# -------------------------
if __name__ == "__main__":
    threading.Thread(target=cleanup_loop, daemon=True).start()
    if IS_WORKER:
        print("Starting worker loop with Redis:", REDIS_URL)
        worker_loop()
        raise SystemExit(0)
    get_redis()
    if STANDALONE_MODE:
        print("Starting in-process worker (no external Redis/worker needed)")
        threading.Thread(target=worker_loop, daemon=True).start()
    url = f"http://{HOST}:{PORT}/"
    if not args.no_open:
        def _open():
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception:
                print("Open your browser at", url)
        threading.Thread(target=_open, daemon=True).start()
    print("Starting server at", url)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
