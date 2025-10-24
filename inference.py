#!/usr/bin/env python3
"""
inference.py - Flask GUI with live decoding + GIF export

Run:
    python inference.py         # opens a browser GUI
    python inference.py --no-open   # do not auto-open browser
    python inference.py --prompt "..." --context "..."  # prefill GUI

Features:
 - Live decoding (server -> browser) via Server-Sent Events (SSE)
 - Inputs: Prompt, Context, Steps, Max-Len, Top-k, Top-p, SQL mask len, Model dir
 - Live text area (big, responsive) at the top that scales to screen sizes
 - Small live animation preview that cycles incoming snapshots as they arrive
 - Export GIF button (appears when generation finishes and GIF built)
 - Font-size fitting for GIF frames so text is large and readable
 - Preserves original denoising / masking algorithm; minimal internal changes to allow callbacks
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
from typing import List, Dict

from flask import Flask, render_template_string, request, jsonify, Response, send_file, stream_with_context
import webbrowser
from PIL import Image, ImageDraw, ImageFont

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
args = parser.parse_args()

HOST = args.host
PORT = args.port

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
      --accent: #2563eb;
      --muted: #6b7280;
      --bg: #fafafa;
      --card: #ffffff;
      --mono: 'DejaVu Sans Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
    }
    html,body { height:100%; margin:0; background:var(--bg); font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color:#111827; }
    .container { max-width:1200px; margin:18px auto; padding:12px; }
    .header { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    h1 { margin:0; font-size:20px; }
    .layout { display:grid; grid-template-columns: 1fr 360px; gap:18px; margin-top:14px; }
    .top-live { background:var(--card); border-radius:10px; padding:12px; box-shadow:0 6px 18px rgba(15,23,42,0.06); display:flex; flex-direction:column; }
    .live-text { flex:1; overflow:auto; padding:8px; border-radius:8px; background:#fff; border:1px solid #eee; }
    .controls { background:var(--card); padding:12px; border-radius:10px; box-shadow:0 6px 18px rgba(15,23,42,0.04); }
    label { font-weight:600; font-size:13px; color:#111827; display:block; margin-bottom:6px; }
    input[type=text], textarea, input[type=number], select { width:100%; padding:8px 10px; border-radius:8px; border:1px solid #e6e6ee; font-size:14px; box-sizing:border-box; }
    textarea { min-height:100px; resize:vertical; font-family:inherit; }
    .small { font-size:13px; color:var(--muted); }
    .row { display:flex; gap:8px; align-items:center; }
    .btn { background:var(--accent); color:white; border:none; padding:10px 14px; border-radius:8px; cursor:pointer; font-weight:600; }
    .btn-ghost { background:transparent; color:var(--accent); border:1px solid #e6eefc; }
    .status { margin-top:10px; background:#1118270f; padding:8px; border-radius:8px; font-family:var(--mono); font-size:13px; max-height:160px; overflow:auto; white-space:pre-wrap; }
    .preview { margin-top:12px; border-radius:8px; padding:10px; background:#fff; border:1px solid #eee; text-align:center; min-height:120px; }
    .field-grid { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
    .footer { margin-top:12px; color:var(--muted); font-size:13px; }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .container { padding:8px; }
    }
    /* big responsive SQL area styles */
    .sql-display { font-family: var(--mono); font-weight:600; line-height:1.1; white-space:pre-wrap; word-break:break-word; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>RoBERTa-diffusion — Live SQL Inference</h1>
      <div class="small">Live decoding on top • Responsive preview & GIF export</div>
    </div>


    <div class="layout" style="margin-top:18px;">
      <div class="controls">
        <form id="runForm">
          <label>Prompt</label>
          <textarea name="prompt" id="prompt">{{ prompt_prefill }}</textarea>

          <label style="margin-top:8px;">Context (schema / info)</label>
          <textarea name="context" id="context">{{ context_prefill }}</textarea>

          <div style="margin-top:8px;" class="field-grid">
            <div>
              <label>Steps</label>
              <input type="number" name="steps" id="steps" value="10" min="1" max="500"/>
            </div>
            <div>
              <label>Max Len</label>
              <input type="number" name="max_len" id="max_len" value="512" min="64" max="4096"/>
            </div>
            <div>
              <label>SQL mask len</label>
              <input type="number" name="sql_len" id="sql_len" value="64" min="1" max="1024"/>
            </div>
            <div>
              <label>Top-k</label>
              <input type="number" name="top_k" id="top_k" value="50" min="0" max="2000"/>
            </div>
          </div>

          <div style="margin-top:8px;" class="field-grid">
            <div>
              <label>Top-p</label>
              <input type="number" step="0.01" name="top_p" id="top_p" value="0.95" min="0" max="1"/>
            </div>
            <div>
              <label>Model Dir</label>
              <input type="text" name="model_dir" id="model_dir" value="{{ model_dir }}" />
            </div>
          </div>

          <div style="display:flex; gap:8px; margin-top:10px;">
            <label style="display:flex; align-items:center; gap:8px;"><input type="checkbox" name="animate" checked /> Collect snapshots</label>
            <label style="display:flex; align-items:center; gap:8px;"><input type="checkbox" name="gif_fullscreen" /> GIF fullscreen (1920x1080)</label>
          </div>

          <div style="margin-top:8px;">
            <label>GIF Width x Height (e.g. 1200x800, overrides fullscreen)</label>
            <input name="gif_size" id="gif_size" type="text" placeholder="1200x800" />
          </div>

          <div style="margin-top:12px; display:flex; gap:8px;">
            <button id="runBtn" class="btn" type="submit">Run Generation</button>
            <button id="resetBtn" class="btn-ghost" type="button">Reset</button>
            <div style="flex:1"></div>
            <div class="small">Model dir must contain tokenizer & RobertaForMaskedLM files.</div>
          </div>
        </form>

        <div id="status" class="status" style="margin-top:12px;">Idle.</div>
      </div>

      <div>
        <div style="background:var(--card); padding:12px; border-radius:10px; box-shadow:0 6px 18px rgba(15,23,42,0.04);">
          <h3 style="margin:0 0 8px 0;">Live Model Generation</h3>
          <div id="liveBox" class="live-text sql-display" style="font-size:28px;">
            <div style="color:#9ca3af;">Waiting for run — enter prompt/context in the form and click Run</div>
          </div>
                <a id="gifLink" class="btn" style="display:none; margin-top:12px;" download>Download GIF</a>

            <button id="stopBtn" class="btn-ghost" style="display:none;">Stop</button>

        </div>
      </div>
    </div>
  </div>

<script>
let runId = null;
let es = null;
let snapshots = [];
let animTimer = null;
let animIndex = 0;

const liveBox = document.getElementById('liveBox');
const animInner = document.getElementById('animInner');
const animPreview = document.getElementById('animPreview');
const statusBox = document.getElementById('status');
const gifLink = document.getElementById('gifLink');
const stopBtn = document.getElementById('stopBtn');

function setStatus(s){ statusBox.textContent = s; }

function setLiveText(txt){
  // set raw text, then adjust font-size heuristically to fit container
  liveBox.textContent = txt || "";
  fitLiveFont();
}

function fitLiveFont(){
  // dynamic font sizing: compute longest line length and container width
  const container = liveBox;
  const width = container.clientWidth - 24;
  const lines = (container.textContent || "").split("\n").map(l=>l.trim());
  const longest = lines.reduce((a,b)=> Math.max(a, b.length), 0) || 40;
  // heuristic: char width ≈ 0.6 * fontSize in px for monospace; adjust:
  const candidate = Math.floor(Math.max(12, Math.min(56, width / Math.max(10, longest) * 1.6)));
  container.style.fontSize = candidate + "px";
}

window.addEventListener('resize', fitLiveFont);

function startAnimatingPreview(){
  if(animTimer) clearInterval(animTimer);
  animIndex = 0;
  if(snapshots.length === 0){
    animInner.textContent = "No frames yet";
    return;
  }
  animTimer = setInterval(()=>{
    animInner.textContent = snapshotToPlain(snapshots[animIndex]);
    animIndex = (animIndex + 1) % snapshots.length;
  }, 500);
}

function stopAnimatingPreview(){
  if(animTimer){ clearInterval(animTimer); animTimer = null; }
}

function snapshotToPlain(s){
  // extract SQL content between tags if present, otherwise whole snapshot
  const start = s.indexOf("<SQL>");
  const end = s.indexOf("</SQL>");
  if(start !== -1 && end !== -1){
    return s.substring(start+5, end).trim();
  }
  return s;
}

document.getElementById('runForm').addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  // clear previous
  stopRun();
  snapshots = [];
  setLiveText("Starting run...");
  gifLink.innerHTML = "";
  gifLink.style.display = "none";
  gifLink.href = "#";
  gifLink.textContent = "";   
  setStatus("Preparing run...");

  const form = new FormData(ev.target);
  // start run (immediate response with run_id)
  setStatus("Sending run request to server...");
  try {
    const resp = await fetch('/start', { method:'POST', body: form });
    if(!resp.ok){
      const txt = await resp.text();
      setStatus("ERROR: " + resp.status + " - " + txt);
      return;
    }
    const j = await resp.json();
    runId = j.run_id;
    setStatus("Run started (id: " + runId + "). Streaming live updates...");
    // open SSE
    openStream(runId);
    // show stop button
    stopBtn.style.display = "inline-block";
    stopBtn.onclick = () => {
      fetch('/stop/' + runId, {method:'POST'});
      stopBtn.style.display = "none";
    };
  } catch(e){
    setStatus("Exception starting run: " + e.toString());
  }
});

function stopRun(){
  if(es){ es.close(); es = null; }
  stopAnimatingPreview();
  stopBtn.style.display = "none";
}

async function openStream(id){
  if(es){ es.close(); es = null; }
  es = new EventSource('/stream/' + id);
  es.onopen = ()=> console.log("SSE opened");
  es.onerror = (e) => {
    console.warn("SSE error", e);
  };
  es.addEventListener('snapshot', (ev)=>{
    const txt = JSON.parse(ev.data);
    snapshots.push(txt);
    // update live text to last snapshot
    setLiveText(snapshotToPlain(txt));
    // update mini preview frames array and start preview
    startAnimatingPreview();
  });
  es.addEventListener('status', (ev)=>{
    const info = JSON.parse(ev.data);
    setStatus(info.msg);
  });
  es.addEventListener('done', async (ev)=>{
    const payload = JSON.parse(ev.data);
    setStatus("Run finished.");
    
      if(payload.gif_url){
    gifLink.href = payload.gif_url;
    gifLink.download = ''; // trigger download
    gifLink.style.display = "inline-block";
    gifLink.textContent = "Download GIF";
    }
    es.close();
    es = null;
    stopBtn.style.display = "none";
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
# Utilities: font fitting for GIF frames
# -------------------------
def pick_font_and_size(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int, padding: int = 24):
    """
    Pick the largest integer font size (within bounds) that makes wrapped text fit within max_width x max_height.
    Uses DejaVuSansMono if available else default.
    """
    # try mono font
    fonts_to_try = ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
    base_font = None
    for p in fonts_to_try:
        try:
            base_font = ImageFont.truetype(p, size=12)
            font_path = p
            break
        except Exception:
            base_font = None
    if base_font is None:
        # fallback
        font_path = None

    # measure longest reasonable width using monospaced assumption
    max_chars = max(40, int((max_width - padding * 2) / 8))
    wrapped = textwrap.fill(text, width=max_chars)
    # candidate sizes
    min_size, max_size = 12, 54
    # binary search for largest size that fits
    lo, hi = min_size, max_size
    best = min_size
    while lo <= hi:
        mid = (lo + hi) // 2
        if font_path:
            f = ImageFont.truetype(font_path, size=mid)
        else:
            f = ImageFont.load_default()
        # recompute wrapping at this font: approximate chars per line using width / (font.getsize('M')[0])
        char_w = f.getsize("M")[0] or 8
        chars_per_line = max(20, int((max_width - padding*2) / char_w))
        wrapped_mid = textwrap.fill(text, width=chars_per_line)
        bbox = draw.multiline_textbbox((0,0), wrapped_mid, font=f, spacing=4)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w + padding*2 <= max_width and h + padding*2 <= max_height:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if font_path:
        return ImageFont.truetype(font_path, size=best)
    else:
        return ImageFont.load_default()

def render_text_image_for_gif(text: str, width: int, height: int, padding: int = 24):
    img = Image.new("RGB", (width, height), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    font = pick_font_and_size(draw, text, width, height, padding=padding)
    # compute char width to determine wrap
    char_w = font.getsize("M")[0] or 8
    chars_per_line = max(20, int((width - padding*2) / char_w))
    wrapped = textwrap.fill(text, width=chars_per_line)
    # compute top-left to center vertically a bit
    bbox = draw.multiline_textbbox((0,0), wrapped, font=font, spacing=6)
    text_h = bbox[3] - bbox[1]
    y = padding
    draw.multiline_text((padding, y), wrapped, fill=(20,20,20), font=font, spacing=6)
    return img

def build_gif_bytes_from_snapshots(snapshots: List[str], size=(1200,800), interval_ms: int = 550) -> bytes:
    frames = []
    for snap in snapshots:
        start = snap.find("<SQL>")
        end = snap.find("</SQL>")
        if start != -1 and end != -1:
            body = snap[start+len("<SQL>"):end].strip()
        else:
            body = snap
        if not body.strip():
            body = "(empty)"
        img = render_text_image_for_gif(body, width=size[0], height=size[1], padding=28)
        frames.append(img.convert("P", palette=Image.ADAPTIVE))
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
# Core denoising with callback for snapshots
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
    on_snapshot = None,   # callable(snapshot_str) -> None called when a snapshot is produced (for live streaming)
    status_cb = None,     # callable(message) -> None used for minor status updates
) -> Dict:
    def log(s):
        if status_cb:
            status_cb(str(s))
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

    # prepare mask region
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
    snapshots: List[str] = []
    if animate:
        s0 = tokenizer.decode(current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
        snapshots.append(s0.replace(mask_token, " ____"))
        if on_snapshot:
            on_snapshot(snapshots[-1])

    log("[INFO] Starting denoising")
    t0 = time.time()
    for p_mask in mask_probs:
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

        if math.isclose(p_mask, 0.0):
            new_ids = current_ids.clone()
            new_ids[0, sql_open_idx + 1 : sql_close_idx] = pred_ids[0, sql_open_idx + 1 : sql_close_idx]
            current_ids = new_ids
            if animate:
                s = tokenizer.decode(current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
                snapshots.append(s.replace(mask_token, " ____"))
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
            snapshots.append(s.replace(mask_token, " ____"))
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
        snapshots = [display]

    return {"snapshots": snapshots, "sql_only": sql_only, "display": display}

# -------------------------
# top-k / top-p helper (same as your original)
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
    return render_template_string(TEMPLATE, prompt_prefill=args.prompt or "", context_prefill=args.context or "", model_dir=args.model_dir or "")

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
    model_dir = form.get("model_dir", args.model_dir).strip() or args.model_dir
    animate = form.get("animate") is not None
    export_gif = True
    gif_fullscreen = form.get("gif_fullscreen") is not None
    gif_size_text = form.get("gif_size", "").strip()
    interval_ms = 550

    if not prompt:
        return "Prompt is empty", 400
    if not os.path.isdir(model_dir):
        # FIXED escape bug here (no odd escaping)
        return jsonify({"error": f"Model dir not found: {model_dir}"}), 400

    run_id = uuid.uuid4().hex
    RUNS[run_id] = {"snapshots": [], "status": "queued", "done": False, "sql_only": None, "display": None, "gif_url": None}

    def status_cb(msg):
        RUNS[run_id]["status"] = msg

    def on_snapshot_callback(snapshot_str):
        RUNS[run_id]["snapshots"].append(snapshot_str)

    def worker():
        try:
            RUNS[run_id]["status"] = "running (loading model)"
            res = run_denoising_generation_callback(prompt, context, model_dir,
                                                    n_steps=steps, max_len=max_len,
                                                    top_k=top_k, top_p=top_p,
                                                    sql_len_request=sql_len, animate=animate,
                                                    on_snapshot=on_snapshot_callback, status_cb=status_cb)
            RUNS[run_id]["sql_only"] = res.get("sql_only")
            RUNS[run_id]["display"] = res.get("display")
            RUNS[run_id]["status"] = "generation complete"
            # optionally build GIF if requested
            if export_gif:
                if gif_fullscreen:
                    gif_size = (1920,1080)
                elif gif_size_text:
                    try:
                        w,h = gif_size_text.lower().split('x')
                        gif_size = (int(w), int(h))
                    except Exception:
                        gif_size = (1200,800)
                else:
                    gif_size = (1200,800)
                RUNS[run_id]["status"] = "building GIF"
                try:
                    gif_bytes = build_gif_bytes_from_snapshots(res["snapshots"], size=gif_size, interval_ms=interval_ms)
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
            # send any new snapshots
            snaps = RUNS[run_id]["snapshots"]
            while sent < len(snaps):
                s = snaps[sent]
                # send snapshot event
                yield f"event: snapshot\ndata: {json.dumps(s)}\n\n"
                sent += 1
            # send status periodically if it changed
            st = RUNS[run_id].get("status", "")
            if st != last_status:
                last_status = st
                yield f"event: status\ndata: {json.dumps({'msg': st})}\n\n"
            if RUNS[run_id].get("done"):
                # final payload includes sql_only and gif_url if present
                payload = {"sql_only": RUNS[run_id].get("sql_only"), "gif_url": RUNS[run_id].get("gif_url")}
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                break
            time.sleep(0.18)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route("/gif/<filename>")
def serve_gif(filename):
    p = os.path.join(os.path.abspath("."), filename)
    if not os.path.isfile(p):
        return "Not found", 404
    return send_file(p, mimetype="image/gif")

@app.route("/stop/<run_id>", methods=["POST"])
def stop_run(run_id):
    # best-effort: mark as done; we cannot easily cancel torch inference safely here
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
