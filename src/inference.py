#!/usr/bin/env python3
"""
inference.py - Flask GUI with live decoding + always-generated GIF download

Usage:
  python inference.py
  python inference.py --no-open
  python inference.py --prompt "..." --context "..." --model-dir "diffusion-sql"

This version:
 - Always generates a GIF and provides a Download button (no preview image).
 - Live preview only shows content inside <SQL>...</SQL>.
 - Live preview uses a large, responsive, easy-to-read monospace display with a light-gray background.
 - GIF text sizing uses a fitting algorithm for large readable text.
 - Deterministic runs by default (seed configurable via CLI).
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
from typing import List, Dict

from flask import Flask, render_template_string, request, jsonify, Response, send_file, stream_with_context
import webbrowser
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

# -------------------------
# CLI args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
parser.add_argument("--prompt", type=str, default="", help="Prefill prompt (optional)")
parser.add_argument("--context", type=str, default="", help="Prefill context (optional)")
parser.add_argument("--model-dir", type=str, default="diffusion-sql", help="Default model dir (optional)")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed (default 1234).")
args = parser.parse_args()

HOST = args.host
PORT = args.port
DEFAULT_MODEL_DIR = args.model_dir
DEFAULT_SEED = 42

# -------------------------
# Flask app & template
# -------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>RoBERTa-diffusion — Live SQL Inference</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root{
      --accent: #1f6feb;
      --muted: #6b7280;
      --bg: #f7fafc;
      --card: #ffffff;
      --mono: 'DejaVu Sans Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
    }
    html,body { height:100%; margin:0; background:var(--bg); font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color:#0f172a; }
    .vh { height:100vh; display:flex; flex-direction:column; }
    .container { width:100%; max-width:1400px; margin:0 auto; padding:12px; box-sizing:border-box; display:flex; flex-direction:column; gap:12px; flex:1 1 auto; }
    .live-large { background:var(--card); border-radius:12px; padding:18px; box-shadow:0 8px 28px rgba(2,6,23,0.06); display:flex; gap:18px; align-items:stretch; min-height:48vh; }
    .live-left { flex:1; display:flex; flex-direction:column; gap:10px; }
    .live-header { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .live-header .title { font-weight:700; font-size:16px; color:#0b1220; }
    .step-pill { background:#eef2ff; color:var(--accent); padding:6px 10px; border-radius:999px; font-weight:700; font-size:14px; }
    /* lighter gray background for live area and nicer padding */
    .live-box { flex:1; background:#f3f4f6; border-radius:10px; padding:22px; overflow:auto; display:flex; align-items:center; justify-content:center; }
    .sql-display { font-family:var(--mono); font-weight:700; white-space:pre-wrap; word-break:break-word; color:#021024; margin:0; }
    .right-col { width:380px; display:flex; flex-direction:column; gap:12px; }
    .controls { background:var(--card); padding:14px; border-radius:10px; box-shadow:0 6px 18px rgba(2,6,23,0.04); }
    .label { font-weight:700; margin-bottom:6px; font-size:13px; color:#0b1220; display:block; }
    input[type=text], textarea, input[type=number], select { width:100%; padding:8px 10px; border-radius:8px; border:1px solid #e6eef7; font-size:14px; box-sizing:border-box; }
    textarea { min-height:80px; resize:vertical; font-family:inherit; }
    .btn { background:var(--accent); color:white; border:none; padding:12px 16px; border-radius:10px; cursor:pointer; font-weight:700; font-size:15px; width:100%; }
    .btn-ghost { background:transparent; color:var(--accent); border:1px solid #e6eef7; padding:10px 12px; border-radius:8px; cursor:pointer; }
    .small { font-size:13px; color:var(--muted); }
    .status { margin-top:6px; background:#ffffff; padding:8px; border-radius:8px; font-family:var(--mono); font-size:13px; max-height:120px; overflow:auto; white-space:pre-wrap; border:1px solid #f0f6ff; }
    .gif-area { margin-top:8px; display:flex; gap:8px; align-items:center; }
    .controls-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .footer { color:var(--muted); font-size:13px; margin-top:6px; }
    @media (max-width: 980px) {
      .live-large { flex-direction:column; min-height:55vh; }
      .right-col { width:100%; }
    }
    .status {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.4em;
  height: 4.2em;
  min-height: 4.2em;
  max-height: 4.2em;
}
  </style>
</head>
<body>
  <div class="vh">
    <div class="container">
      <div class="live-large" id="liveLarge">
        <div class="live-left">
          <div class="live-header">
            <div class="title">Lethonium - Text Diffusion Playground</div>
            <div class="step-pill" id="stepBox">Step — / —</div>
          </div>
          <div class="live-box" id="liveBox" aria-live="polite">
            <!-- liveText is large, responsive; only SQL content will be shown -->
            <pre id="liveText" class="sql-display" style="font-size:44px; margin:0; color:#0b1220;">Waiting — enter prompt/context and press Run</pre>
          </div>
        </div>

        <div class="right-col">
          <div class="controls">
            <label class="label">Controls</label>
            <form id="runForm">
              <label class="small">Prompt</label>
              <textarea name="prompt" id="prompt">{{ prompt_prefill }}</textarea>

              <label class="small" style="margin-top:8px;">Context</label>
              <textarea name="context" id="context">{{ context_prefill }}</textarea>

              <div class="controls-grid" style="margin-top:10px;">
                <div>
                  <label class="small">Steps</label>
                  <input name="steps" id="steps" type="number" value="10" min="1" max="500"/>
                </div>
                <div>
                  <label class="small">Max Len</label>
                  <input name="max_len" id="max_len" type="number" value="512" min="64" max="4096"/>
                </div>
                <div>
                  <label class="small">SQL mask len</label>
                  <input name="sql_len" id="sql_len" type="number" value="64" min="1" max="1024"/>
                </div>
                <div>
                  <label class="small">Top-k</label>
                  <input name="top_k" id="top_k" type="number" value="50" min="0" max="2000"/>
                </div>
              </div>

              <div class="controls-grid" style="margin-top:8px;">
                <div>
                  <label class="small">Top-p</label>
                  <input name="top_p" id="top_p" type="number" step="0.01" value="0.95" min="0" max="1"/>
                </div>
                <div>
                  <label class="small">Model Dir</label>
                  <input name="model_dir" id="model_dir" type="text" value="{{ model_dir }}" />
                </div>
              </div>

              <div style="margin-top:12px;">
                <button id="runBtn" class="btn" type="submit">Run Generation</button>
              </div>
            </form>

            <div id="status" class="status">Idle.</div>
          </div>

          <div>
            <div class="small" style="margin-bottom:6px;">Download GIF</div>
            <div class="gif-area">
              <a id="gifLink" class="btn-ghost" href="#" style="display:none;" download>Download GIF</a>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
let runId = null;
let es = null;
let snapshots = [];
let runInProgress = false;

const liveText = document.getElementById('liveText');
const stepBox = document.getElementById('stepBox');
const statusBox = document.getElementById('status');
const gifLink = document.getElementById('gifLink');
const runBtn = document.getElementById('runBtn');
const runForm = document.getElementById('runForm');

function setStatus(s){ statusBox.textContent = s; }
function setLiveText(txt){
  liveText.textContent = txt || "";
  fitLiveFont();
}
function fitLiveFont(){
  const container = liveText;
  const width = container.parentElement.clientWidth - 40;
  const lines = (container.textContent || "").split("\n").map(l=>l.trim());
  const longest = lines.reduce((a,b)=> Math.max(a, b.length), 0) || 40;
  const candidate = Math.floor(Math.max(18, Math.min(96, width / Math.max(6, longest) * 2.2)));
  container.style.fontSize = candidate + "px";
}
window.addEventListener('resize', fitLiveFont);

function extractSQL(s){
  const start = s.indexOf("<SQL>");
  const end = s.indexOf("</SQL>");
  if(start !== -1 && end !== -1){
    return s.substring(start+5, end).trim();
  }
  return s;
}

function setBtnStateRunning() {
  runInProgress = true;
  runBtn.textContent = "Stop Generation";
  runBtn.disabled = false;
}
function setBtnStateIdle() {
  runInProgress = false;
  runBtn.textContent = "Run Generation";
  runBtn.disabled = false;
}

async function stopRunIfRunning() {
  if (runId) {
    runBtn.disabled = true;
    setStatus("Stopping...");
    try {
      await fetch('/stop/' + runId, { method:'POST' });
    } catch(_) {}
  }
}

runBtn.onclick = async function(ev) {
  // This manually handles both submit and "stop" actions.
  if (runInProgress) {
    ev.preventDefault();
    await stopRunIfRunning();
    // Button stays disabled until stream closes.
    return;
  }
  // Otherwise, allow form submit!
};

runForm.addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  if(es){ es.close(); es=null; }
  snapshots = [];
  setStatus("Preparing run...");
  gifLink.style.display = "none";

  setBtnStateRunning();

  const form = new FormData(ev.target);
  setStatus("Sending run request to server (deterministic)...");
  try {
    const resp = await fetch('/start', { method:'POST', body: form });
    if(!resp.ok){
      const txt = await resp.text();
      setStatus("ERROR: " + resp.status + " - " + txt);
      setBtnStateIdle();
      return;
    }
    const j = await resp.json();
    runId = j.run_id;
    setStatus("Run started (id: " + runId + "). Streaming updates...");
    openStream(runId);
  } catch(e){
    setStatus("Exception starting run: " + e.toString());
    setBtnStateIdle();
  }
});

async function openStream(id){
  if(es){ es.close(); es=null; }
  es = new EventSource('/stream/' + id);
  es.onopen = ()=> console.log("SSE opened");
  es.onerror = (e)=> console.warn("SSE error", e);
  es.addEventListener('snapshot', (ev)=>{
    const obj = JSON.parse(ev.data);
    snapshots.push(obj);
    const sqlOnly = extractSQL(obj.text.replace(/____/g,'_____')) || "(empty)";
    setLiveText(sqlOnly);
    stepBox.textContent = `Step ${obj.step} / ${obj.total_steps}`;
  });
  es.addEventListener('status', (ev)=>{
    const info = JSON.parse(ev.data);
    setStatus(info.msg);
  });
  function finishBtn() {
    setBtnStateIdle();
    runId = null;
  }
  es.addEventListener('done', async (ev)=>{
    const payload = JSON.parse(ev.data);
    setStatus("Run finished.");
    if(payload.gif_url){
      gifLink.href = payload.gif_url;
      const fname = payload.gif_url.split('/').pop() || 'generation.gif';
      gifLink.download = fname;
      gifLink.style.display = "inline-block";
      gifLink.textContent = "Download GIF";
    } else {
      gifLink.style.display = "none";
    }
    if(es){ es.close(); es=null; }
    finishBtn();
  });
}
</script>
</body>
</html>
"""

# -------------------------
# Run storage / state
# -------------------------
RUNS: Dict[str, Dict] = {}  # run_id -> {snapshots:[], status:str, done:bool, sql_only, display, gif_path}
MODEL_CACHE = {"dir": None, "tokenizer": None, "model": None}

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

def build_gif_bytes_from_snapshots(snapshots: List[Dict], size=(1400,900), interval_ms: int = 500) -> bytes:
    frames = []
    for snap in snapshots:
        s = snap.get("text") if isinstance(snap, dict) else snap
        start = s.find("<SQL>")
        end = s.find("</SQL>")
        if start != -1 and end != -1:
            body = s[start+len("<SQL>"):end].strip()
        else:
            body = s
        if not body.strip():
            body = "(empty)"
        img = render_text_image_for_gif(body, width=size[0], height=size[1], padding=28)
        frames.append(img.convert("P", palette=Image.ADAPTIVE))
    if not frames:
        frames = [Image.new("RGB", size, color=(255,255,255)).convert("P", palette=Image.ADAPTIVE)]
    bio = io.BytesIO()
    frames[0].save(bio, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=interval_ms)
    bio.seek(0)
    return bio.read()

# -------------------------
# Model load helper (cached)
# -------------------------
def load_model_and_tokenizer(model_dir: str, max_len: int = 512):
    if MODEL_CACHE["dir"] == model_dir and MODEL_CACHE["tokenizer"] is not None and MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["tokenizer"], MODEL_CACHE["model"]
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
    tokenizer.model_max_length = max_len
    model = RobertaForMaskedLM.from_pretrained(model_dir)
    model.to(torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")))
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
    n_steps: int = 10,
    max_len: int = 512,
    top_k: int = 50,
    top_p: float = 0.95,
    sql_len_request: int = 64,
    animate: bool = True,
    on_snapshot = None,
    status_cb = None,
    deterministic_seed: int = DEFAULT_SEED,
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
        log(f"[INFO] Adding missing special tokens: {missing}")
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
        log("[INFO] Resized embeddings")

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    full_text = f"<PROMPT>{prompt_text}</PROMPT> <CONTEXT>{context_text}</CONTEXT> <SQL></SQL>"
    enc = tokenizer(full_text, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt", return_offsets_mapping=True)
    input_ids_full = enc["input_ids"].squeeze(0)
    offsets = enc["offset_mapping"].squeeze(0).tolist()
    attn_full = enc["attention_mask"].squeeze(0)

    sql_open_id = tokenizer.convert_tokens_to_ids("<SQL>")
    sql_close_id = tokenizer.convert_tokens_to_ids("</SQL>")

    def find_first_index(tensor, val):
        matches = (tensor == val).nonzero(as_tuple=True)[0]
        return matches[0].item() if len(matches) > 0 else None

    orig_sql_open_idx = find_first_index(input_ids_full, sql_open_id)
    orig_sql_close_idx = find_first_index(input_ids_full, sql_close_id)

    if orig_sql_open_idx is None or orig_sql_close_idx is None:
        txt = full_text
        open_char = txt.find("<SQL>")
        close_char = txt.find("</SQL>")
        if open_char == -1 or close_char == -1:
            raise RuntimeError("Could not find <SQL> or </SQL> in constructed text.")
        content_start_char = open_char + len("<SQL>")
        content_end_char = close_char
        token_start = next((i for i, (s, e) in enumerate(offsets) if e > content_start_char), None)
        tokens_covering = [i for i, (s, e) in enumerate(offsets) if s < content_end_char and e > content_start_char]
        token_end = tokens_covering[-1] + 1 if tokens_covering else None
        if token_start is None or token_end is None:
            raise RuntimeError("Failed to compute token span for SQL content (fallback).")
        orig_sql_open_idx = max(0, token_start - 1)
        orig_sql_close_idx = token_end

    sql_open_idx = int(orig_sql_open_idx)
    sql_close_idx = int(orig_sql_close_idx)
    if not (0 <= sql_open_idx < sql_close_idx <= max_len):
        if not (0 <= sql_open_idx < max_len):
            raise RuntimeError(f"Invalid SQL token indices: open={sql_open_idx}, close={sql_close_idx}")

    if sql_close_idx <= sql_open_idx + 1:
        max_possible = max_len - (sql_open_idx + 2)
        if max_possible <= 0:
            raise RuntimeError("Not enough room in sequence to insert SQL content masks.")
        sql_len = min(sql_len_request, max_possible)
        new_close_idx = sql_open_idx + 1 + sql_len
        log(f"[INFO] empty SQL region — inserting {sql_len} mask tokens")
        current_ids = torch.full((1, max_len), fill_value=mask_id, dtype=torch.long)
        current_ids[0, : sql_open_idx + 1] = input_ids_full[: sql_open_idx + 1]
        current_ids[0, sql_open_idx + 1 : new_close_idx] = mask_id
        if new_close_idx < max_len:
            current_ids[0, new_close_idx] = sql_close_id
        if orig_sql_close_idx is not None and orig_sql_close_idx != new_close_idx and orig_sql_close_idx < max_len:
            input_ids_full[orig_sql_close_idx] = pad_id
        current_attention = attn_full.unsqueeze(0).clone()
        current_attention[0, sql_open_idx + 1 : new_close_idx + 1] = 1
        sql_close_idx = new_close_idx
    else:
        current_ids = torch.full((1, max_len), fill_value=mask_id, dtype=torch.long)
        current_ids[0, : sql_open_idx + 1] = input_ids_full[: sql_open_idx + 1]
        if sql_close_idx < max_len:
            current_ids[0, sql_close_idx] = input_ids_full[sql_close_idx]
        current_attention = attn_full.unsqueeze(0).clone()

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

    mask_probs = [i / n_steps for i in range(n_steps - 1, -1, -1)]
    total_steps = len(mask_probs)
    snapshots: List[Dict] = []
    if animate:
        s0 = tokenizer.decode(current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
        snapshots.append({"text": s0.replace(mask_token, " ____"), "step": 0, "total_steps": total_steps})
        if on_snapshot:
            on_snapshot(snapshots[-1])

    log("[INFO] Starting denoising")
    t0 = time.time()
    for step_idx, p_mask in enumerate(mask_probs):
        with torch.no_grad():
            outputs = model(input_ids=current_ids, attention_mask=current_attention)
            logits = outputs.logits

        pred_ids = current_ids.clone()
        for pos in modifiable_positions:
            logit_vec = logits[0, pos, :]
            filtered = top_k_top_p_filtering(logit_vec, top_k=top_k, top_p=top_p, filter_value=-float("Inf"))
            probs = torch.softmax(filtered, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            pred_ids[0, pos] = sampled

        if status_cb:
            status_cb(f"step {step_idx+1}/{total_steps}")

        if math.isclose(p_mask, 0.0):
            new_ids = current_ids.clone()
            new_ids[0, sql_open_idx + 1 : sql_close_idx] = pred_ids[0, sql_open_idx + 1 : sql_close_idx]
            current_ids = new_ids
            if animate:
                s = tokenizer.decode(current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
                snapshots.append({"text": s.replace(mask_token, " ____"), "step": step_idx+1, "total_steps": total_steps})
                if on_snapshot:
                    on_snapshot(snapshots[-1])
            break

        rand = torch.rand((max_len,), device=device)
        remask = (rand < p_mask) & modifiable

        next_ids = current_ids.clone()
        for pos in modifiable_positions:
            if remask[pos]:
                next_ids[0, pos] = mask_id
            else:
                next_ids[0, pos] = pred_ids[0, pos]

        current_ids = next_ids
        if animate:
            s = tokenizer.decode(current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            snapshots.append({"text": s.replace(mask_token, " ____"), "step": step_idx+1, "total_steps": total_steps})
            if on_snapshot:
                on_snapshot(snapshots[-1])

    t1 = time.time()
    log(f"[INFO] Denoising took {t1 - t0:.2f}s")

    decoded_full = tokenizer.decode(current_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    display = decoded_full.replace(mask_token, "_____")
    try:
        start_marker = "<SQL>"
        end_marker = "</SQL>"
        start_idx = display.index(start_marker) + len(start_marker)
        end_idx = display.index(end_marker, start_idx)
        sql_only = display[start_idx:end_idx].strip()
    except ValueError:
        token_slice = current_ids[0, sql_open_idx + 1 : sql_close_idx].detach().cpu().tolist()
        sql_only = tokenizer.decode(token_slice, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    if not animate:
        snapshots = [{"text": display, "step": total_steps, "total_steps": total_steps}]

    return {"snapshots": snapshots, "sql_only": sql_only, "display": display}

# -------------------------
# top-k / top-p helper
# -------------------------
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, filter_value: float = -float("Inf")) -> torch.Tensor:
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k)[0][..., -1]
        indices_to_remove = logits < kth_value
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# -------------------------
# Flask endpoints
# -------------------------
@app.route("/")
def index():
    default_prompt = "how many planes are of name a380?"
    default_context = "CREATE TABLE planes (id INT, name TEXT)"
    return render_template_string(
        TEMPLATE,
        prompt_prefill=args.prompt or default_prompt,
        context_prefill=args.context or default_context,
        model_dir=DEFAULT_MODEL_DIR or "",
    )
@app.route("/start", methods=["POST"])
def start_run():
    form = request.form
    prompt = form.get("prompt", "").strip()
    context = form.get("context", "").strip()
    steps = int(form.get("steps", "10"))
    max_len = int(form.get("max_len", "512"))
    sql_len = int(form.get("sql_len", "64"))
    top_k = int(form.get("top_k", "50"))
    top_p = float(form.get("top_p", "0.95"))
    model_dir = form.get("model_dir", DEFAULT_MODEL_DIR).strip() or DEFAULT_MODEL_DIR
    animate = True
    gif_size_text = form.get("gif_size", "").strip()
    interval_ms = 500

    if not prompt:
        return "Prompt is empty", 400
    if not os.path.isdir(model_dir):
        return jsonify({"error": f"Model dir not found: {model_dir}"}), 400

    run_id = uuid.uuid4().hex
    RUNS[run_id] = {"snapshots": [], "status": "queued", "done": False, "sql_only": None, "display": None, "gif_url": None}

    def status_cb(msg):
        RUNS[run_id]["status"] = msg

    def on_snapshot_callback(snapshot_obj):
        RUNS[run_id]["snapshots"].append(snapshot_obj)

    def worker():
        try:
            RUNS[run_id]["status"] = "running (loading model)"
            res = run_denoising_generation_callback(prompt, context, model_dir,
                                                    n_steps=steps, max_len=max_len,
                                                    top_k=top_k, top_p=top_p,
                                                    sql_len_request=sql_len, animate=animate,
                                                    on_snapshot=on_snapshot_callback, status_cb=status_cb,
                                                    deterministic_seed=DEFAULT_SEED)
            RUNS[run_id]["sql_only"] = res.get("sql_only")
            RUNS[run_id]["display"] = res.get("display")
            RUNS[run_id]["status"] = "generation complete"

            # always build GIF
            if gif_size_text:
                try:
                    w,h = gif_size_text.lower().split('x')
                    gif_size = (int(w), int(h))
                except Exception:
                    gif_size = (1400,900)
            else:
                gif_size = (1400,900)

            RUNS[run_id]["status"] = "building GIF"
            try:
                gif_bytes = build_gif_bytes_from_snapshots(RUNS[run_id]["snapshots"], size=gif_size, interval_ms=interval_ms)
                out_name = f"generation_{run_id}.gif"
                out_path = os.path.join(os.path.abspath("."), out_name)
                with open(out_path, "wb") as f:
                    f.write(gif_bytes)
                RUNS[run_id]["gif_url"] = "/gif/" + out_name
            except Exception as e:
                RUNS[run_id]["status"] = f"GIF build failed: {e}"

            RUNS[run_id]["done"] = True
            RUNS[run_id]["status"] = "done"
        except Exception as e:
            RUNS[run_id]["status"] = f"error: {e}"
            RUNS[run_id]["done"] = True

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return jsonify({"run_id": run_id})

@app.route("/stream/<run_id>")
def stream(run_id):
    if run_id not in RUNS:
        return "Not found", 404

    def event_stream():
        sent = 0
        last_status = ""
        while True:
            snaps = RUNS[run_id]["snapshots"]
            while sent < len(snaps):
                s = snaps[sent]
                yield f"event: snapshot\ndata: {json.dumps(s)}\n\n"
                sent += 1
            st = RUNS[run_id].get("status", "")
            if st != last_status:
                last_status = st
                yield f"event: status\ndata: {json.dumps({'msg': st})}\n\n"
            if RUNS[run_id].get("done"):
                payload = {"sql_only": RUNS[run_id].get("sql_only"), "gif_url": RUNS[run_id].get("gif_url")}
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                break
            time.sleep(0.12)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route("/gif/<filename>")
def serve_gif(filename):
    p = os.path.join(os.path.abspath("."), filename)
    if not os.path.isfile(p):
        return "Not found", 404
    return send_file(p, mimetype="image/gif")

@app.route("/stop/<run_id>", methods=["POST"])
def stop_run(run_id):
    if run_id in RUNS:
        RUNS[run_id]["status"] = "stopped by user"
        RUNS[run_id]["done"] = True
        return "ok"
    return "not found", 404

# -------------------------
# Start server
# -------------------------
if __name__ == "__main__":
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
