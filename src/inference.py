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
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--public", action="store_true", help="Bind on 0.0.0.0 and disable browser auto-open")
    parser.add_argument("--prompt", type=str, default="", help="Prefill prompt (optional)")
    parser.add_argument("--context", type=str, default="", help="Prefill context (optional)")
    parser.add_argument("--model-dir", type=str, default="diffusion-sql", help="Default model dir (optional)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (default 42).")
    parser.add_argument("--max-concurrent-runs", type=int, default=1)
    parser.add_argument("--max-queued-runs", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=int, default=90)
    parser.add_argument("--max-prompt-chars", type=int, default=1000)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--max-max-len", type=int, default=512)
    parser.add_argument("--max-sql-len", type=int, default=128)
    parser.add_argument("--enable-gif", action="store_true", help="Enable GIF generation for runs")
    parser.add_argument("--run-ttl-seconds", type=int, default=900)
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
    storage_uri="memory://",
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


def queue_position(run_id: str) -> int | None:
    r = get_redis()
    q = r.lrange(QUEUE_KEY, 0, -1)
    try:
        return q.index(run_id) + 1
    except ValueError:
        return None


def active_worker_count() -> int:
    return int(get_redis().scard(ACTIVE_SET_KEY))


def estimate_eta_seconds(position: int | None) -> int | None:
    if position is None:
        return None
    raw = get_redis().hget(METRICS_KEY, "avg_duration")
    if raw is None:
        return None
    try:
        avg = float(raw)
    except ValueError:
        return None
    return max(1, int(round(avg * position)))


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
    }
    try:
        if MODEL_CACHE["dir"] == model_dir and MODEL_CACHE["model"] is not None:
            model = MODEL_CACHE["model"]
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_dir)
        param_count = int(sum(p.numel() for p in model.parameters()))
        info["param_count"] = param_count
        info["param_count_display"] = format_param_count(param_count)
    except Exception:
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

        def status_cb(msg):
            redis_update_run(run_id, status=str(msg))

        def on_snapshot_callback(snapshot_obj):
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
            prev_avg = r.hget(METRICS_KEY, "avg_duration")
            prev_n = r.hget(METRICS_KEY, "count")
            try:
                prev_avg_f = float(prev_avg) if prev_avg else 0.0
                prev_n_i = int(prev_n) if prev_n else 0
            except ValueError:
                prev_avg_f = 0.0
                prev_n_i = 0
            new_n = prev_n_i + 1
            new_avg = elapsed if prev_n_i == 0 else ((prev_avg_f * prev_n_i) + elapsed) / new_n
            r.hset(METRICS_KEY, mapping={"avg_duration": new_avg, "count": new_n})
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

    snapshots: List[Dict] = []
    if animate:
        s0_clean = snapshot_text()
        snapshots.append({"text": s0_clean, "sql_only": extract_sql_only_from_text(s0_clean), "step": 0, "total_steps": total_steps})
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
            s_clean = snapshot_text()
            snapshots.append({"text": s_clean, "sql_only": extract_sql_only_from_text(s_clean), "step": step_idx+1, "total_steps": total_steps})
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
    return jsonify({
        "ok": redis_ok,
        "redis_ok": redis_ok,
        "active_limit": MAX_CONCURRENT_RUNS,
        "queue_limit": MAX_QUEUED_RUNS,
        "active_worker_count": active_worker_count() if redis_ok else 0,
    }), (200 if redis_ok else 503)


@app.route("/")
def index():
    default_prompt = "how many planes are of name 747?"
    default_context = "CREATE TABLE planes (id INT, name TEXT)"
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
        model_param_count=model_info.get("param_count"),
        model_param_count_display=model_info.get("param_count_display", "unknown"),
        inference_dtype_requested=INFERENCE_DTYPE,
        inference_dtype_effective=effective_dtype,
        static_version=static_version,
    )


@app.route("/start", methods=["POST"])
@limiter.limit("5 per minute; 30 per hour")
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
    model_ok, model_msg = validate_model_dir(model_dir)
    if not model_ok:
        return jsonify({"error": "invalid_model", "message": model_msg}), 400
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

    r = get_redis()
    if r.llen(QUEUE_KEY) >= MAX_QUEUED_RUNS:
        return jsonify({
            "error": "queue_full",
            "message": "Server is busy. Try again shortly.",
            "max_queued_runs": MAX_QUEUED_RUNS,
        }), 503

    run_id = uuid.uuid4().hex
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
    eta = estimate_eta_seconds(pos)
    eta_confidence = estimate_eta_confidence(pos, eta)
    return jsonify({
        "run_id": run_id,
        "state": "queued",
        "queue_position": pos,
        "eta_seconds": eta,
        "eta_confidence": eta_confidence,
        "queue_token": run_id,
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
    eta = estimate_eta_seconds(pos)
    return jsonify({
        "run_id": run_id,
        "state": run.get("state"),
        "status": run.get("status"),
        "done": bool(run.get("done")),
        "sql_only": run.get("sql_only"),
        "gif_url": run.get("gif_url"),
        "snapshot_count": snap_count,
        "snapshots": delta,
        "queue_position": pos,
        "eta_seconds": eta,
        "eta_confidence": estimate_eta_confidence(pos, eta),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
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
    eta = estimate_eta_seconds(pos)
    return jsonify({
        "run_id": run_id,
        "state": run.get("state"),
        "queue_position": pos,
        "eta_seconds": eta,
        "eta_confidence": estimate_eta_confidence(pos, eta),
        "active_worker_count": active_worker_count(),
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
        return send_file(p, mimetype="image/gif", as_attachment=True, download_name=safe_name)
    except TypeError:
        return send_file(p, mimetype="image/gif", as_attachment=True, attachment_filename=safe_name)


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
