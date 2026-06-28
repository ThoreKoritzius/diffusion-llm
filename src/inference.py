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
import gc
from typing import List, Dict, Optional
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
try:
    import onnxruntime as ort
except Exception:
    ort = None

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
    parser.add_argument("--max-steps", type=int, default=env_int("MAX_STEPS", 64))
    parser.add_argument("--max-max-len", type=int, default=env_int("MAX_MAX_LEN", 512))
    parser.add_argument("--max-sql-len", type=int, default=env_int("MAX_SQL_LEN", 256))
    parser.add_argument("--enable-gif", action="store_true", help="Enable GIF generation for runs")
    parser.add_argument("--run-ttl-seconds", type=int, default=env_int("RUN_TTL_SECONDS", 900))
    parser.add_argument("--worker", action="store_true", help="Run worker loop instead of web server")
    parser.add_argument("--redis-url", type=str, default=os.environ.get("REDIS_URL", "redis://redis:6379/0"))
    parser.add_argument("--inference-dtype", type=str, default=os.environ.get("INFERENCE_DTYPE", "fp16"), choices=["fp16", "fp32", "bf16", "auto"], help="Preferred model dtype for inference")
    parser.add_argument("--confidence-stop", type=float, default=float(os.environ.get("CONFIDENCE_STOP", "0.9")), help="Adaptive early-stop: commit any masked position whose top-token probability >= this threshold, so easy queries use fewer denoising steps. 0 disables (fixed steps).")
    parser.add_argument("--inference-backend", type=str, default=os.environ.get("INFERENCE_BACKEND", "torch"), choices=["torch", "onnx", "auto"], help="Inference backend for model forwards")
    parser.add_argument("--sql-generation-mode", type=str, default=os.environ.get("SQL_GENERATION_MODE", "auto"), choices=["block", "fixed", "auto"], help="SQL denoising layout: block predicts </SQL> as a stop token; fixed matches legacy PAD-padded SQL windows; auto uses checkpoint config.")
    parser.add_argument("--sql-block-size", type=int, default=env_int("SQL_BLOCK_SIZE", 32), help="Block size for variable-length block SQL generation.")
    parser.add_argument("--eos-token-bias", type=float, default=env_float("EOS_TOKEN_BIAS", 1.5), help="Logit bias for </SQL> during block generation; 0 disables.")
    parser.add_argument("--onnx-cache-dir", type=str, default=os.environ.get("ONNX_CACHE_DIR", os.path.join(os.getcwd(), "onnx_cache")), help="Writable cache directory for exported ONNX models")
    parser.add_argument("--onnx-opset", type=int, default=env_int("ONNX_OPSET", 17), help="ONNX opset used when exporting")
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
PRELOAD_MODEL = env_bool("PRELOAD_MODEL", IS_WORKER)
PRELOAD_WARMUP_FORWARD = env_bool("PRELOAD_WARMUP_FORWARD", PRELOAD_MODEL)
REDIS_URL = args.redis_url
INFERENCE_DTYPE = (args.inference_dtype or "fp16").lower()
# Adaptive early-stop confidence threshold; <=0 or >=1 disables (fixed steps).
CONFIDENCE_STOP = args.confidence_stop if 0.0 < float(args.confidence_stop) < 1.0 else None
INFERENCE_BACKEND = (args.inference_backend or "torch").lower()
SQL_GENERATION_MODE = (args.sql_generation_mode or "block").lower()
SQL_BLOCK_SIZE = max(1, int(args.sql_block_size))
EOS_TOKEN_BIAS = float(args.eos_token_bias)
ONNX_CACHE_DIR = os.path.abspath(args.onnx_cache_dir)
ONNX_OPSET = max(13, int(args.onnx_opset))
ALLOW_FAKE_REDIS = env_bool("ALLOW_FAKE_REDIS", "REDIS_URL" not in os.environ)
RATELIMIT_STORAGE_URI = os.environ.get("RATELIMIT_STORAGE_URI") or ("memory://" if ALLOW_FAKE_REDIS else REDIS_URL)
START_RUN_RATE_LIMIT = os.environ.get("START_RUN_RATE_LIMIT", "30 per minute; 300 per hour")
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
EFFECTIVE_LEN_MULTIPLE = 64
MIN_EFFECTIVE_MAX_LEN = 64
DEFAULT_WARMUP_SEQUENCE_LENGTHS = "64,256"
ONNX_EXPORT_VERSION = "v1"
ONNX_GRAPH_OPT_LEVEL = os.environ.get("ONNX_GRAPH_OPT_LEVEL", "all").strip().lower()
ONNX_INTRA_OP_THREADS = env_int("ONNX_INTRA_OP_THREADS", env_int("TORCH_NUM_THREADS", 0))
ONNX_INTER_OP_THREADS = env_int("ONNX_INTER_OP_THREADS", 1)
ONNX_EXECUTION_MODE = os.environ.get("ONNX_EXECUTION_MODE", "sequential").strip().lower()

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
MODEL_CACHE = {"dir": None, "backend_request": None, "backend": None, "tokenizer": None, "model": None}
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


def configure_torch_cpu_threads() -> None:
    torch_threads = env_int("TORCH_NUM_THREADS", 0)
    if torch_threads > 0:
        try:
            torch.set_num_threads(torch_threads)
        except RuntimeError as exc:
            print(f"[WARN] Could not set TORCH_NUM_THREADS={torch_threads}: {exc}", flush=True)

    interop_threads = env_int("TORCH_INTEROP_THREADS", 1)
    if interop_threads > 0:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as exc:
            print(f"[WARN] Could not set TORCH_INTEROP_THREADS={interop_threads}: {exc}", flush=True)


configure_torch_cpu_threads()


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def env_int_list(name: str, default: str) -> List[int]:
    raw = os.environ.get(name, default)
    values: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue
    return values


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


def round_up_to_multiple(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return ((max(0, int(value)) + multiple - 1) // multiple) * multiple


def compute_effective_max_len(requested_max_len: int, prompt_tokens: int, context_tokens: int, sql_tokens: int) -> int:
    requested_max_len = max(1, int(requested_max_len))
    needed_len_with_slack = int(prompt_tokens) + int(context_tokens) + int(sql_tokens) + 9
    rounded_len = round_up_to_multiple(needed_len_with_slack, EFFECTIVE_LEN_MULTIPLE)
    effective_len = max(min(requested_max_len, MIN_EFFECTIVE_MAX_LEN), rounded_len)
    return min(requested_max_len, effective_len)


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
    for k in ("created_at", "updated_at", "started_at", "finished_at", "first_snapshot_at"):
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
    if out.get("step_stats"):
        try:
            out["step_stats"] = json.loads(out["step_stats"])
        except json.JSONDecodeError:
            out["step_stats"] = []
    else:
        out["step_stats"] = []
    for k in ("steps_used", "max_steps_cap"):
        try:
            out[k] = int(out[k]) if out.get(k) not in (None, "") else None
        except (ValueError, TypeError):
            out[k] = None
    try:
        out["confidence_threshold"] = float(out["confidence_threshold"]) if out.get("confidence_threshold") not in (None, "") else None
    except (ValueError, TypeError):
        out["confidence_threshold"] = None
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
        "early_stop": int(payload.get("early_stop", 1)),
        "seed": int(DEFAULT_SEED),
        "dtype": INFERENCE_DTYPE,
        "backend": INFERENCE_BACKEND,
        "sql_generation_mode": SQL_GENERATION_MODE,
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


def write_result_cache(payload: Dict, sql_only: str, display: str, snapshots: List[Dict], extra: Dict | None = None) -> None:
    if not RESULT_CACHE_ENABLED or not snapshots:
        return
    cached = {
        "payload": cache_signature_payload(payload),
        "sql_only": sql_only or "",
        "display": display or "",
        "snapshots": snapshots,
        "created_at": now_ts(),
    }
    if extra:
        cached.update(extra)
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
        "step_stats": cached.get("step_stats", []),
        "steps_used": cached.get("steps_used", ""),
        "max_steps_cap": cached.get("max_steps_cap", ""),
        "confidence_threshold": cached.get("confidence_threshold", ""),
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


def running_timing_payload(run: Dict) -> Dict:
    started_at = run.get("started_at")
    first_snapshot_at = run.get("first_snapshot_at")
    elapsed = None
    first_frame_remaining = None
    first_frame_avg = average_first_frame_seconds()

    if started_at:
        elapsed = max(0.0, now_ts() - float(started_at))
        if not first_snapshot_at and not run.get("snapshot_count"):
            first_frame_remaining = max(0, int(math.ceil(first_frame_avg - elapsed)))

    return {
        "running_elapsed_seconds": int(math.floor(elapsed)) if elapsed is not None else None,
        "first_frame_remaining_seconds": first_frame_remaining,
        "avg_first_frame_seconds": int(math.ceil(first_frame_avg)),
        "first_snapshot_at": first_snapshot_at,
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


def product_int(values) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def count_safetensors_parameters(path: str) -> int | None:
    try:
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), byteorder="little", signed=False)
            header = json.loads(f.read(header_len))
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    total = 0
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = meta.get("shape")
        if isinstance(shape, list):
            total += product_int(shape)
    return total or None


def count_model_parameters_from_weights(model_dir: str) -> tuple[int | None, str]:
    single_safetensors = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single_safetensors):
        count = count_safetensors_parameters(single_safetensors)
        if count:
            return count, "model.safetensors"

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_names = sorted(set((index.get("weight_map") or {}).values()))
        except (OSError, json.JSONDecodeError):
            shard_names = []
        total = 0
        for shard_name in shard_names:
            count = count_safetensors_parameters(os.path.join(model_dir, shard_name))
            if count:
                total += count
        if total:
            return total, "model.safetensors.index.json"

    return None, "unknown"


def configured_backend_display() -> str:
    if INFERENCE_BACKEND == "onnx":
        return "onnxruntime"
    if INFERENCE_BACKEND == "auto":
        return "onnxruntime preferred, PyTorch fallback"
    return "pytorch"


def runtime_backend_display() -> str:
    loaded = MODEL_CACHE.get("backend")
    if loaded:
        return str(loaded)
    if not IS_WORKER and not ALLOW_FAKE_REDIS:
        return f"{configured_backend_display()} (worker process)"
    return f"{configured_backend_display()} (not loaded yet)"


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
        "param_count_source": "unknown",
    }
    param_count, param_source = count_model_parameters_from_weights(model_dir)
    if param_count:
        info["param_count"] = param_count
        info["param_count_display"] = format_param_count(param_count)
        info["param_count_source"] = param_source
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
    redis_update_run(run_id, state="running", status="worker claimed (loading model)", started_at=start_time)
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
            confidence_stop=(CONFIDENCE_STOP if int(payload.get("early_stop", 1)) else None),
        )
        final_sql = strip_final_masks(res.get("sql_only", ""))
        redis_update_run(
            run_id,
            sql_only=final_sql,
            display=res.get("display"),
            step_stats=res.get("step_stats", []),
            steps_used=int(res.get("steps_used", 0) or 0),
            max_steps_cap=int(res.get("max_steps_cap", payload.get("steps", 0)) or 0),
            confidence_threshold=(CONFIDENCE_STOP if int(payload.get("early_stop", 1)) and CONFIDENCE_STOP else ""),
        )

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
            write_result_cache(payload, final_sql, res.get("display", ""), cached_snaps, extra={
                "step_stats": res.get("step_stats", []),
                "steps_used": int(res.get("steps_used", 0) or 0),
                "max_steps_cap": int(res.get("max_steps_cap", payload.get("steps", 0)) or 0),
                "confidence_threshold": (CONFIDENCE_STOP if int(payload.get("early_stop", 1)) and CONFIDENCE_STOP else None),
            })
    except TimeoutError:
        set_run_terminal(run_id, state="timed_out", status="timed out")
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        set_run_terminal(run_id, state="error", status=f"error: {e}\n{tb}")
    finally:
        r.srem(ACTIVE_SET_KEY, run_id)


def worker_loop():
    preload_worker_model()
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
        if os.path.isdir(OUT_DIR):
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
# Model load helper (cached)
# -------------------------
class MaskedLMOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class OnnxMaskedLMRunner:
    backend_label = "onnxruntime"

    def __init__(self, session, output_name: str):
        self.session = session
        self.output_name = output_name
        self.device = torch.device("cpu")

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> MaskedLMOutput:
        ort_inputs = {
            "input_ids": input_ids.detach().cpu().numpy().astype(np.int64, copy=False),
            "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int64, copy=False),
        }
        logits = self.session.run([self.output_name], ort_inputs)[0]
        return MaskedLMOutput(torch.from_numpy(logits))


class MaskedLMExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def is_onnx_runner(model) -> bool:
    return isinstance(model, OnnxMaskedLMRunner)


def model_backend_label(model) -> str:
    return getattr(model, "backend_label", "torch")


def resolve_sql_generation_mode(model) -> str:
    requested = SQL_GENERATION_MODE
    if requested in {"block", "fixed"}:
        return requested
    cfg = getattr(model, "config", None)
    mode = str(getattr(cfg, "diffusion_sql_mode", "") or "").strip().lower()
    return mode if mode in {"block", "fixed"} else "fixed"


def get_model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def load_tokenizer(model_dir: str, max_len: int):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    tokenizer.model_max_length = max_len
    return tokenizer


def load_torch_masked_lm(model_dir: str, device: torch.device | None = None, dtype: torch.dtype | None = None):
    device = device or get_inference_device()
    if dtype is None:
        dtype, _dtype_label = resolve_model_dtype(device)
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, torch_dtype=dtype)
    except Exception:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, torch_dtype=torch.float32)
        dtype = torch.float32
    model.to(device)
    model.eval()
    return model


def onnx_graph_optimization_level():
    if ort is None:
        return None
    if ONNX_GRAPH_OPT_LEVEL == "disable":
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if ONNX_GRAPH_OPT_LEVEL == "basic":
        return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if ONNX_GRAPH_OPT_LEVEL == "extended":
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.GraphOptimizationLevel.ORT_ENABLE_ALL


def model_cache_fingerprint(model_dir: str) -> str:
    h = hashlib.sha256()
    h.update(ONNX_EXPORT_VERSION.encode("utf-8"))
    h.update(str(ONNX_OPSET).encode("utf-8"))
    h.update(ONNX_GRAPH_OPT_LEVEL.encode("utf-8"))
    h.update((getattr(ort, "__version__", "no-ort") if ort else "no-ort").encode("utf-8"))
    h.update(os.path.abspath(model_dir).encode("utf-8"))
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()[:16]


def onnx_cache_paths(model_dir: str) -> tuple[str, str]:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", os.path.basename(os.path.abspath(model_dir)) or "model")
    fingerprint = model_cache_fingerprint(model_dir)
    base = f"{safe_name}-{fingerprint}-opset{ONNX_OPSET}"
    return (
        os.path.join(ONNX_CACHE_DIR, f"{base}.onnx"),
        os.path.join(ONNX_CACHE_DIR, f"{base}.optimized.onnx"),
    )


def export_onnx_model(model_dir: str, tokenizer, raw_path: str, max_len: int) -> None:
    if os.path.exists(raw_path):
        return
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    tmp_path = f"{raw_path}.{os.getpid()}.tmp"
    print(f"[INFO] Exporting ONNX model to {raw_path}", flush=True)

    model = load_torch_masked_lm(model_dir, device=torch.device("cpu"), dtype=torch.float32)
    wrapper = MaskedLMExportWrapper(model).eval()

    export_len = clamp_int(env_int("ONNX_EXPORT_SEQUENCE_LENGTH", min(max_len, 256)), 8, max_len)
    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if mask_id is None or cls_id is None or sep_id is None or pad_id is None:
        raise RuntimeError("Tokenizer is missing mask/cls/sep/pad IDs required for ONNX export.")

    ids = [cls_id, mask_id, sep_id] + [pad_id] * max(0, export_len - 3)
    attn = [1, 1, 1] + [0] * max(0, export_len - 3)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.tensor([attn], dtype=torch.long)

    export_kwargs = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        "opset_version": ONNX_OPSET,
        "do_constant_folding": True,
    }
    try:
        with torch.inference_mode():
            torch.onnx.export(wrapper, (input_ids, attention_mask), tmp_path, dynamo=False, **export_kwargs)
    except TypeError:
        with torch.inference_mode():
            torch.onnx.export(wrapper, (input_ids, attention_mask), tmp_path, **export_kwargs)
    os.replace(tmp_path, raw_path)
    del wrapper
    del model
    gc.collect()
    print("[INFO] ONNX export complete", flush=True)


def load_onnx_masked_lm(model_dir: str, tokenizer, max_len: int) -> OnnxMaskedLMRunner:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed.")
    raw_path, optimized_path = onnx_cache_paths(model_dir)
    export_onnx_model(model_dir, tokenizer, raw_path, max_len)

    sess_options = ort.SessionOptions()
    if ONNX_INTRA_OP_THREADS > 0:
        sess_options.intra_op_num_threads = ONNX_INTRA_OP_THREADS
    if ONNX_INTER_OP_THREADS > 0:
        sess_options.inter_op_num_threads = ONNX_INTER_OP_THREADS
    sess_options.execution_mode = (
        ort.ExecutionMode.ORT_PARALLEL if ONNX_EXECUTION_MODE == "parallel" else ort.ExecutionMode.ORT_SEQUENTIAL
    )
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True

    model_path = raw_path
    if os.path.exists(optimized_path):
        model_path = optimized_path
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    else:
        sess_options.graph_optimization_level = onnx_graph_optimization_level()
        sess_options.optimized_model_filepath = optimized_path

    providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in ort.get_available_providers() else None
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    output_name = session.get_outputs()[0].name
    print(f"[INFO] ONNX Runtime session ready: {model_path}", flush=True)
    return OnnxMaskedLMRunner(session, output_name)


def load_model_and_tokenizer(model_dir: str, max_len: int = 512):
    if (
        MODEL_CACHE["dir"] == model_dir
        and MODEL_CACHE["backend_request"] == INFERENCE_BACKEND
        and MODEL_CACHE["tokenizer"] is not None
        and MODEL_CACHE["model"] is not None
    ):
        return MODEL_CACHE["tokenizer"], MODEL_CACHE["model"]

    tokenizer = load_tokenizer(model_dir, max_len)
    backend = INFERENCE_BACKEND
    if backend in {"onnx", "auto"}:
        try:
            model = load_onnx_masked_lm(model_dir, tokenizer, max_len)
            MODEL_CACHE.update({
                "dir": model_dir,
                "backend_request": INFERENCE_BACKEND,
                "backend": "onnxruntime",
                "tokenizer": tokenizer,
                "model": model,
            })
            return tokenizer, model
        except Exception as exc:
            if backend == "onnx":
                raise
            print(f"[WARN] ONNX backend unavailable, falling back to PyTorch: {exc}", flush=True)

    model = load_torch_masked_lm(model_dir)
    MODEL_CACHE.update({
        "dir": model_dir,
        "backend_request": INFERENCE_BACKEND,
        "backend": "torch",
        "tokenizer": tokenizer,
        "model": model,
    })
    return tokenizer, model


def warmup_model_forward(tokenizer, model) -> None:
    if not PRELOAD_WARMUP_FORWARD:
        return

    lengths = [
        clamp_int(length, 8, MAX_MAX_LEN)
        for length in env_int_list("WARMUP_SEQUENCE_LENGTHS", DEFAULT_WARMUP_SEQUENCE_LENGTHS)
    ]
    lengths = sorted(set(lengths))
    if not lengths:
        return

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        return
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        return

    device = get_model_device(model)
    for length in lengths:
        ids = [cls_id, mask_id, sep_id] + [pad_id] * max(0, length - 3)
        attn = [1, 1, 1] + [0] * max(0, length - 3)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=device)
        with torch.inference_mode():
            _ = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, 1, :].argmax(dim=-1)


def preload_worker_model() -> None:
    if not PRELOAD_MODEL:
        return
    try:
        print(f"[INFO] Preloading {INFERENCE_BACKEND} model/tokenizer from {DEFAULT_MODEL_DIR}", flush=True)
        tokenizer, model = load_model_and_tokenizer(DEFAULT_MODEL_DIR, MAX_MAX_LEN)
        warmup_model_forward(tokenizer, model)
        print(f"[INFO] Model/tokenizer preload complete ({model_backend_label(model)})", flush=True)
    except Exception as exc:
        print(f"[WARN] Model/tokenizer preload failed: {exc}", flush=True)
        if INFERENCE_BACKEND == "onnx":
            raise


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
    confidence_stop: Optional[float] = None,
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
        if is_onnx_runner(model):
            raise RuntimeError(f"ONNX backend requires special tokens to exist before export; missing: {missing}")
        log(f"[INFO] Adding missing special tokens at inference-time: {missing}")
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
        log("[INFO] Resized model embeddings for added tokens")
    log(f"[INFO] Active inference backend: {model_backend_label(model)}")

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Build inputs with the same token-level tags as train.py. Block mode predicts
    # </SQL> as a stop token; fixed mode below preserves the legacy PAD-filled
    # SQL window contract used by older checkpoints.
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    tag_id = {t: tokenizer.convert_tokens_to_ids(t) for t in REQUIRED_TAGS}
    sql_mode = resolve_sql_generation_mode(model)
    log(f"[INFO] SQL generation mode: {sql_mode}")

    requested_max_len = max(9, int(max_len))
    sql_len = max(1, min(int(sql_len_request), requested_max_len - 9))
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    if sql_mode == "block":
        configured_block = int(getattr(model.config, "diffusion_sql_block_size", SQL_BLOCK_SIZE) or SQL_BLOCK_SIZE)
        block_len = max(1, min(configured_block, sql_len))
        budget = max(0, requested_max_len - sql_len - 8)
        if len(prompt_ids) + len(context_ids) > budget:
            if len(prompt_ids) > budget:
                prompt_ids = prompt_ids[:budget]
                context_ids = []
            else:
                context_ids = context_ids[: max(0, budget - len(prompt_ids))]

        head = (
            [cls_id, tag_id["<PROMPT>"]] + prompt_ids
            + [tag_id["</PROMPT>"], tag_id["<CONTEXT>"]] + context_ids
            + [tag_id["</CONTEXT>"], tag_id["<SQL>"]]
        )
        generated_ids: List[int] = []
        end_sql_id = int(tag_id["</SQL>"])
        pad_token = tokenizer.pad_token or "<pad>"
        sample_temperature = 0.0 if int(top_k) <= 1 else 1.0
        special_forbid = set(getattr(tokenizer, "all_special_ids", []) or [])
        special_forbid.update(tag_id.values())
        special_forbid.discard(end_sql_id)
        special_forbid.discard(int(mask_id))
        forbid_ids = list(special_forbid)
        max_blocks = int(math.ceil(sql_len / block_len))
        block_steps = max(1, min(int(n_steps), block_len))

        def block_sql_text(ids_for_display: List[int]) -> str:
            visible = []
            for tid in ids_for_display:
                tid = int(tid)
                if tid == end_sql_id:
                    break
                visible.append(tid)
            return strip_final_masks(decode_sql_from_token_ids(tokenizer, visible, mask_id=mask_id, pad_id=pad_id))

        def snapshot_text_for(ids_for_display: List[int]) -> str:
            body = tokenizer.decode(ids_for_display, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            body = body.replace(mask_token, " ____").replace(pad_token, "")
            return f"<SQL>{body}</SQL>"

        snapshots: List[Dict] = []
        step_stats: List[Dict] = []
        steps_used = 0
        last_render = None
        stopped = False

        if animate:
            initial = " ".join(["____"] * min(block_len, 6))
            snapshots.append({"text": f"<SQL>{initial}</SQL>", "sql_only": initial, "step": 0, "total_steps": block_steps * max_blocks})
            if on_snapshot:
                on_snapshot(snapshots[-1])
            last_render = initial

        log(f"[INFO] Starting block denoising: mode=block block_len={block_len} max_sql_len={sql_len}")
        t0 = time.time()
        device = get_model_device(model)

        for block_idx in range(max_blocks):
            if should_stop is not None and should_stop():
                raise TimeoutError("Run cancelled or timed out")
            remaining = sql_len - len(generated_ids)
            if remaining <= 0:
                break
            this_block_len = min(block_len, remaining)
            current_len = compute_effective_max_len(
                requested_max_len=requested_max_len,
                prompt_tokens=len(prompt_ids),
                context_tokens=len(context_ids),
                sql_tokens=len(generated_ids) + this_block_len,
            )
            ids = head + generated_ids + [mask_id] * this_block_len + [sep_id]
            pad_count = max(0, current_len - len(ids))
            attn = [1] * len(ids) + [0] * pad_count
            ids = ids + [pad_id] * pad_count
            current_ids = torch.tensor([ids], dtype=torch.long, device=device)
            current_attention = torch.tensor([attn], dtype=torch.long, device=device)
            block_start = len(head) + len(generated_ids)
            block_positions = list(range(block_start, block_start + this_block_len))
            current_steps = max(1, min(block_steps, this_block_len))

            for step_idx, total_steps, current_ids in denoise_steps(
                model,
                current_ids,
                current_attention,
                block_positions,
                mask_id,
                n_steps=current_steps,
                temperature=sample_temperature,
                forbid_token_ids=forbid_ids,
                bias_token_ids={end_sql_id: EOS_TOKEN_BIAS} if EOS_TOKEN_BIAS else None,
                should_stop=should_stop,
                confidence_stop=confidence_stop,
                on_step_stats=step_stats.append,
            ):
                steps_used += 1
                if status_cb:
                    status_cb(f"block {block_idx+1}/{max_blocks} step {step_idx+1}/{total_steps}")
                if animate and step_idx + 1 < total_steps:
                    block_now = current_ids[0, block_start : block_start + this_block_len].detach().cpu().tolist()
                    r = block_sql_text(generated_ids + block_now)
                    if r and r != last_render:
                        last_render = r
                        snapshots.append({
                            "text": snapshot_text_for(generated_ids + block_now),
                            "sql_only": r,
                            "step": steps_used,
                            "total_steps": block_steps * max_blocks,
                        })
                        if on_snapshot:
                            on_snapshot(snapshots[-1])

            block_out = current_ids[0, block_start : block_start + this_block_len].detach().cpu().tolist()
            if end_sql_id in [int(tid) for tid in block_out]:
                stop_idx = [int(tid) for tid in block_out].index(end_sql_id)
                generated_ids.extend(block_out[:stop_idx])
                stopped = True
                break
            generated_ids.extend(block_out)

        final_sql_only = block_sql_text(generated_ids)
        if animate:
            snapshots.append({"text": f"<SQL>{final_sql_only}</SQL>", "sql_only": final_sql_only, "step": steps_used, "total_steps": steps_used})
            if on_snapshot:
                on_snapshot(snapshots[-1])

        t1 = time.time()
        log(f"[INFO] Block denoising took {t1 - t0:.2f}s ({'stopped' if stopped else 'budget exhausted'})")
        display = f"<SQL>{final_sql_only}</SQL>"
        if not animate:
            snapshots = [{"text": display, "sql_only": final_sql_only, "step": steps_used, "total_steps": steps_used}]
        return {
            "snapshots": snapshots,
            "sql_only": final_sql_only,
            "display": display,
            "steps_used": steps_used,
            "max_steps_cap": block_steps * max_blocks,
            "step_stats": step_stats,
        }

    budget = max(0, requested_max_len - sql_len - 9)  # CLS + 6 tags + SEP (+1 slack); mirrors training
    if len(prompt_ids) + len(context_ids) > budget:
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[:budget]
            context_ids = []
        else:
            context_ids = context_ids[: max(0, budget - len(prompt_ids))]

    max_len = compute_effective_max_len(
        requested_max_len=requested_max_len,
        prompt_tokens=len(prompt_ids),
        context_tokens=len(context_ids),
        sql_tokens=sql_len,
    )
    if max_len < requested_max_len:
        log(f"[INFO] Effective sequence length: {max_len} tokens (requested {requested_max_len})")

    head = (
        [cls_id, tag_id["<PROMPT>"]] + prompt_ids
        + [tag_id["</PROMPT>"], tag_id["<CONTEXT>"]] + context_ids
        + [tag_id["</CONTEXT>"], tag_id["<SQL>"]]
    )
    sql_open_idx = len(head) - 1         # index of the <SQL> tag
    sql_start = len(head)                # first maskable SQL position
    ids = head + [mask_id] * sql_len + [tag_id["</SQL>"], sep_id]
    sql_close_idx = sql_start + sql_len  # index of the </SQL> tag
    pad_count = max(0, max_len - len(ids))
    attn = [1] * len(ids) + [0] * pad_count
    ids = ids + [pad_id] * pad_count

    current_ids = torch.tensor([ids], dtype=torch.long)
    current_attention = torch.tensor([attn], dtype=torch.long)
    log(f"[INFO] SQL window: {sql_len} mask tokens at positions {sql_start}..{sql_close_idx}")

    device = get_model_device(model)
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

        Real SQL tokens form a prefix; the SQL window is PAD-filled out to a
        fixed number of slots. Rendering all slots means most denoising steps only commit PAD ->
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
            # Length not decided yet: short pending field instead of the full fixed window.
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

    steps_used = 0
    step_stats: List[Dict] = []
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
        confidence_stop=confidence_stop,
        on_step_stats=step_stats.append,
    ):
        steps_used = step_idx + 1  # adaptive early-stop may finish before total_steps
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

    # Use the count of committing steps (early stop can add a final no-op yield).
    if step_stats:
        steps_used = len(step_stats)
    final_token_slice = current_ids[0, sql_open_idx + 1 : sql_close_idx].detach().cpu().tolist()
    final_sql_only = strip_final_masks(decode_sql_from_token_ids(tokenizer, final_token_slice, mask_id=mask_id, pad_id=pad_id))
    if animate:
        s_clean_final = replace_sql_section(snapshot_text(), final_sql_only)
        snapshots.append({"text": s_clean_final, "sql_only": final_sql_only, "step": steps_used, "total_steps": total_steps})
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
        snapshots = [{"text": display, "sql_only": sql_only, "step": steps_used, "total_steps": total_steps}]

    return {
        "snapshots": snapshots,
        "sql_only": sql_only,
        "display": display,
        "steps_used": steps_used,
        "max_steps_cap": total_steps,
        "step_stats": step_stats,
    }

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
        "inference_backend_requested": INFERENCE_BACKEND,
        "inference_backend_runtime": runtime_backend_display(),
        "inference_backend_loaded": MODEL_CACHE.get("backend"),
        "onnxruntime_available": ort is not None,
        "onnxruntime_version": getattr(ort, "__version__", None) if ort else None,
        "onnx_cache_dir": ONNX_CACHE_DIR,
        "onnx_opset": ONNX_OPSET,
        "onnx_graph_optimization_level": ONNX_GRAPH_OPT_LEVEL,
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
    default_prompt = "What is the highest car price for each model?"
    default_context = "CREATE TABLE cars (id INT, make VARCHAR(50), model VARCHAR(50), year INT, price DECIMAL(10,2));"
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
        model_param_count_source=model_info.get("param_count_source", "unknown"),
        inference_dtype_requested=INFERENCE_DTYPE,
        inference_dtype_effective=effective_dtype,
        inference_backend_requested=INFERENCE_BACKEND,
        inference_backend_runtime=runtime_backend_display(),
        static_version=static_version,
    )


@app.route("/start", methods=["POST"])
@limiter.limit(START_RUN_RATE_LIMIT)
def start_run():
    form = request.form
    prompt = form.get("prompt", "").strip()
    context = form.get("context", "").strip()
    model_dir = DEFAULT_MODEL_DIR
    gif_size_text = form.get("gif_size", "").strip()
    early_stop = form.get("early_stop", "1").strip().lower() not in ("0", "false", "off", "no", "")
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
        "early_stop": 1 if early_stop else 0,
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
        last_status_sent_at = 0.0
        while True:
            _, snaps = redis_get_snapshot_delta(run_id, cursor)
            for snap in snaps:
                yield f"event: snapshot\\ndata: {json.dumps(snap)}\\n\\n"
                cursor += 1

            run_now = redis_get_run(run_id)
            if not run_now:
                break
            st = run_now.get("status", "")
            state = run_now.get("state")
            snap_count = int(run_now.get("snapshot_count") or 0)
            now = time.time()
            phase_tick_due = False
            if state == "queued":
                phase_tick_due = (now - last_status_sent_at) >= 2.0
            elif state == "running" and snap_count == 0:
                phase_tick_due = (now - last_status_sent_at) >= 1.0

            if st != last_status or phase_tick_due:
                last_status = st
                last_status_sent_at = now
                status_payload = {
                    "msg": st,
                    "state": state,
                    "snapshot_count": snap_count,
                    "started_at": run_now.get("started_at"),
                }
                if state == "queued":
                    pos = queue_position(run_id)
                    if pos is None:
                        pos = 1
                    status_payload.update(queue_timing_payload(pos))
                    status_payload["demand"] = adaptive_start_profile().get("demand")
                elif state in {"running", "stopping"}:
                    status_payload.update(running_timing_payload(run_now))
                yield f"event: status\\ndata: {json.dumps(status_payload)}\\n\\n"

            is_done = run_now.get("done")
            payload = {
                "sql_only": run_now.get("sql_only"),
                "gif_url": run_now.get("gif_url"),
                "state": run_now.get("state"),
                "status": run_now.get("status"),
                "step_stats": run_now.get("step_stats"),
                "steps_used": run_now.get("steps_used"),
                "max_steps_cap": run_now.get("max_steps_cap"),
                "confidence_threshold": run_now.get("confidence_threshold"),
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
    if run.get("state") in {"running", "stopping"}:
        timing.update(running_timing_payload(run))
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
        "step_stats": run.get("step_stats"),
        "steps_used": run.get("steps_used"),
        "max_steps_cap": run.get("max_steps_cap"),
        "confidence_threshold": run.get("confidence_threshold"),
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
