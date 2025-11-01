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
from werkzeug.utils import secure_filename

from flask import Flask, render_template_string, request, jsonify, Response, send_file, stream_with_context
import webbrowser
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from queue import Queue, Empty

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
  <title>Diffusion LLM Playground</title>
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
.sql-display {
  transition: filter 20ms cubic-bezier(.2,.9,.2,1), opacity 20ms cubic-bezier(.2,.9,.2,1);
  will-change: filter, opacity;
}

.live-box {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}
#liveTextWrapper {
  width: 100%;
  min-height: 100px;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}
#liveTextWrapper pre {
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
}

#liveTextWrapper pre {
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  width: 100%;
}
/* But on desktop (>=700px), center vertically: */
@media (min-width: 701px) {
  #liveTextWrapper pre {
    align-items: center;
    justify-content: center;
  }
}
@media (max-width: 700px) {
  .user-info-dialog {
    width: 96vw;
    padding: 16px;
    font-size: 15px;
  }
}
.tab-header {
  background: #f5faff;
  border: 1.1px solid #dde5f7;
  margin-bottom:8px;
}
.tab-btn {
  flex: 1 1 0;
  padding: 12px 0;
  border: none;
  font-size: 15px;
  background: none;
  outline: none;
  color: #5a5f6c;
  font-weight: 600;
  cursor: pointer;
  transition: background .12s, color .12s;
  border-bottom: 2.5px solid transparent;
}
.tab-btn-active, .tab-btn:focus {
  background: #eaf3fe;
  color: #1e4aa6;
  border-bottom: 2.5px solid var(--accent, #1f6feb);
  z-index: 1;
}
.tab-panes {
  min-height: 110px;
}
.tab-pane {
  display: none;
  padding: 0;
}
.tab-pane-active {
  display: block;
}
#csvTableWrapper table {
  border-collapse: collapse;
  width: 100%;
  font-family: var(--mono), monospace;
  font-size: 13.5px;
}
#csvTableWrapper th, #csvTableWrapper td {
  border: 1px solid #e4e6f2;
  padding: 4px 9px;
  text-align: left;
  background: #fcfcfd;
  color: #34425a;
  max-width: 180px;
  overflow-wrap: break-word;
}
#csvTableWrapper th {
  background: #f4f6fb;
  color: #294688;
  font-weight: 800;
  position: sticky; top: 0; z-index: 2;
}
@media (max-width:850px) {
  .right-col {
    width: 100%;
  }
}

/* Animated mask / pad runs to smoothly shrink/expand token runs */
.sql-display .mask-run,
.sql-display .pad-run,
.sql-display .char {
  display: inline-block;
  white-space: pre;
  transform-origin: left center;
  transition: transform 420ms cubic-bezier(.2,.9,.2,1), opacity 420ms cubic-bezier(.2,.9,.2,1);
  will-change: transform, opacity;
}

/* Represent a run of underscore mask characters; width based on character count */
.sql-display .mask-run {
  --len: 4;
  width: calc(var(--len) * 0.62ch); /* monospace; adjust multiplier if needed */
  overflow: hidden;
}

/* A pad run (represents removed/padded space) — we animate width too */
.sql-display .pad-run {
  --len: 1;
  width: calc(var(--len) * 0.62ch);
  overflow: hidden;
  opacity: 0.0;
}

/* initial collapsed state for new elements (so they can expand) */
.sql-display .collapsed {
  transform: scaleX(0);
  opacity: 0;
}

/* explicit shrink for elements that should collapse */
.sql-display .shrink {
  transform: scaleX(0);
  opacity: 0;
}

/* ensure normal state is visible */
.sql-display .visible {
  transform: scaleX(1);
  opacity: 1;
}

/* keep the pre editable for innerHTML usage */
.sql-display { white-space: pre-wrap; word-break: break-word; }

  </style>
</head>
<body>
  <div class="vh">
    <div class="container">
      <div class="live-large" id="liveLarge">
        <div class="live-left">
          <div class="live-header">
            <div class="title">Text Diffusion Playground</div>
            <div class="step-pill" id="stepBox">Step — / —</div>
          </div>
          <div class="live-box" id="liveBox" aria-live="polite">
           <div id="liveTextWrapper" style="width:100%; position:relative; display:flex; align-items:center; justify-content:center; min-height: 100px">
                <pre id="liveTextFront" class="sql-display" style="position:absolute; inset:0; margin:0; font-size:44px; color:#0b1220; z-index:2;">Waiting — enter prompt/context and press Run</pre>
                <pre id="liveTextBack" class="sql-display" style="position:absolute; inset:0; margin:0; font-size:44px; color:#0b1220; z-index:1; opacity:0; filter:blur(12px);"> </pre>
            </div>
          </div>
          <div id="sliderRow" style="width:100%; margin-top:18px; display:none; flex-direction:column; align-items:center; gap:6px;">
            <input type="range" min="0" value="0" id="snapSlider" style="width:100%;">
            <div class="small" id="snapSliderLabel"></div>
         </div>
        </div>
        <div class="right-col">
          <div class="controls">
            <label class="label">Controls</label>
           <form id="runForm" autocomplete="off">
    <!-- Prompt is always visible above the tabs -->
    <label class="small" for="prompt"><b>Prompt</b></label>
    <textarea name="prompt" id="prompt" placeholder="e.g. how many planes are of name a380?" style="margin-bottom:10px;">{{ prompt_prefill }}</textarea>

    <!-- Tab header -->
    <div class="tab-header" style="display:flex; border-radius:10px; overflow:hidden; margin-bottom:14px; box-shadow:0 2px 12px #eef2f4;">
      <button type="button" class="tab-btn tab-btn-active" data-tab="data">Data</button>
      <button type="button" class="tab-btn" data-tab="advanced">Advanced</button>
    </div>
    <!-- Tab content -->
    <div class="tab-panes">
      <!-- DATA TAB -->
      <div class="tab-pane tab-pane-active" id="tab-data">
        <!-- CSV Upload -->
        <div style="margin-bottom:16px;">
          <label class="small" for="csvInput"><b>Upload CSV</b></label>
          <input type="file" id="csvInput" accept=".csv" style="width:100%;margin-top:5px;"/>
        </div>
        <!-- Table preview -->
        <div id="csvTableWrapper" style="background:#f5f9ff;border:1px solid #e5e9f2; border-radius:8px; margin-bottom:14px;max-height:200px;overflow:auto;"></div>
        <div class="small" style="color:#56677a;">
          Tip: Table schema is auto-populated in Advanced tab.<br>
          Only the first 30 rows are previewed.
        </div>
      </div>
      <!-- ADVANCED TAB -->
      <div class="tab-pane" id="tab-advanced">
        <label class="small" for="context" style="margin-top:8px;"><b>Context</b></label>
        <textarea name="context" id="context" placeholder="Table schema will appear here">{{ context_prefill }}</textarea>

        <div class="controls-grid" style="margin-top:12px;">
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
            <input name="top_k" id="top_k" type="number" value="1" min="0" max="2000"/>
          </div>
        </div>

        <div class="controls-grid" style="margin-top:10px;">
          <div>
            <label class="small">Top-p</label>
            <input name="top_p" id="top_p" type="number" step="0.01" value="1" min="0" max="1"/>
          </div>
          <div>
            <label class="small">Model Dir</label>
            <input name="model_dir" id="model_dir" type="text" value="{{ model_dir }}" />
          </div>
        </div>
      </div>
    </div>
    <!-- Run button ALWAYS shown -->
    <div style="margin-top:16px;">
      <button id="runBtn" class="btn" type="submit" style="width:100%;font-size:15.5px;font-weight:700;">Run Generation</button>
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
const sliderRow = document.getElementById('sliderRow');
const snapSlider = document.getElementById('snapSlider');
const snapSliderLabel = document.getElementById('snapSliderLabel');

function setStatus(s){ statusBox.textContent = s; }
const liveFront = document.getElementById('liveTextFront');
const liveBack = document.getElementById('liveTextBack');

function fitLiveFontForElem(elem) {
  const container = elem.parentElement;
  let width = container.clientWidth - 40;
  if (window.innerWidth <= 600) {
    width = Math.max(180, width * 0.9);
  }
  const lines = (elem.textContent || "").split("\n");
  const longest = lines.reduce((a,b)=> Math.max(a, b.length), 0) || 40;

  // Responsive formula: larger min on desktop, smaller min on mobile.
  let minFont = window.innerWidth < 600 ? 14 : 24;
  let maxFont = window.innerWidth < 600 ? 28 : 96;
  let candidate = Math.floor(Math.max(
    minFont, 
    Math.min(maxFont, width / Math.max(8, longest) * 2.2)
  ));
  elem.style.fontSize = candidate + "px";
}

// global helper used by resize
function fitLiveFont(){
  fitLiveFontForElem(liveFront);
  fitLiveFontForElem(liveBack);
}

window.addEventListener('resize', fitLiveFont);

// animate blur/unblur transition
let animating = false;
function animateBlurToText(newText){
  if(animating){
    // quick fallback: stop current animation and show text
    liveFront.style.transition = "";
    liveBack.style.transition = "";
    liveFront.textContent = newText;
    fitLiveFontForElem(liveFront);
    // restore transitions
    setTimeout(()=> {
      liveFront.style.transition = "";
      liveBack.style.transition = "";
    }, 20);
    animating = false;
    return;
  }

  // prepare back with the new text
  liveBack.textContent = newText || "";
  fitLiveFontForElem(liveBack);

  // initial visual state for back
  const blurMax = 4;
  liveBack.style.filter = `blur(${blurMax}px)`;
  liveBack.style.opacity = "0";
  // ensure ordering
  liveBack.style.zIndex = 2;
  liveFront.style.zIndex = 1;

  // trigger reflow so transition occurs
  // then animate: back -> blur 0, opacity 1 ; front -> opacity 0
  requestAnimationFrame(()=> {
    liveBack.style.transition = "filter 420ms cubic-bezier(.2,.9,.2,1), opacity 420ms cubic-bezier(.2,.9,.2,1)";
    liveFront.style.transition = "opacity 420ms cubic-bezier(.2,.9,.2,1)";
    // target states
    liveBack.style.filter = "blur(0px)";
    liveBack.style.opacity = "1";
    liveFront.style.opacity = "0";
    animating = true;

    // after transition, swap content & reset styles
    setTimeout(()=> {
      // put new text into front and reset back
      liveFront.textContent = newText || "";
      fitLiveFontForElem(liveFront);
      liveFront.style.opacity = "1";
      liveFront.style.transition = "";
      liveFront.style.filter = "none";
      liveFront.style.zIndex = 2;

      liveBack.style.opacity = "0";
      liveBack.style.filter = `blur(${blurMax}px)`;
      liveBack.style.transition = "";
      liveBack.style.zIndex = 1;
      animating = false;
    }, 460);
  });
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
  sliderRow.style.display = "none";

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
    animateBlurToText(sqlOnly);

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

  if (snapshots.length > 1) {
    snapSlider.max = (snapshots.length - 1).toString();
    snapSlider.value = snapSlider.max;
    sliderRow.style.display = "flex";
    updateSliderLabel();
  } else {
    sliderRow.style.display = "none";
  }
  });
}

function updateSliderLabel() {
  if (!snapshots.length) {
    snapSliderLabel.innerText = "";
    return;
  }
  const idx = parseInt(snapSlider.value);
  const snap = snapshots[idx] || {};
  snapSliderLabel.innerText = `Step ${snap.step || (idx+1)} / ${snap.total_steps || snapshots.length}`;
}

// User moves slider => show that snapshot
snapSlider.addEventListener('input', ()=>{
  if (!snapshots.length) return;
  const idx = parseInt(snapSlider.value);
  const snap = snapshots[idx];
  const sqlOnly = extractSQL((snap.text || "").replace(/____/g,'_____')) || "(empty)";
  animateBlurToText(sqlOnly);
  stepBox.textContent = `Step ${snap.step} / ${snap.total_steps}`;
  updateSliderLabel();
});
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.onclick = function() {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-btn-active'));
    btn.classList.add('tab-btn-active');
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('tab-pane-active'));
    document.getElementById('tab-' + btn.dataset.tab).classList.add('tab-pane-active');
  }
});

document.getElementById('csvInput').addEventListener('change', function(ev){
  const file = ev.target.files[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = function(e){
    processCSV(e.target.result);
  };
  reader.readAsText(file);
});

// Guess data types from first nonempty data row
function guessType(val){
  if(/^[\d]+$/.test(val)) return "INT";
  if(/^[\d\.]+$/.test(val) && val.includes('.')) return "FLOAT";
  return "TEXT";
}
function processCSV(text){
  const lines = text.trim().split(/\r?\n/);
  if(!lines.length) return;
  const headers = lines[0].split(",");
  // Find first "valid" data row for type inference
  let dataRow = null;
  for(let i=1;i<lines.length;i++){
    if(lines[i].trim()) {
      dataRow = lines[i].split(",");
      break;
    }
  }
  if(!dataRow) dataRow = headers.map(x=>"");
  let types = dataRow.map(guessType);
  // Table preview (first 30, min 5 rows)
  let dataRows = lines.slice(1).map(r=>r.split(",")).slice(0,30);
  let html = '<table><thead><tr>' +
    headers.map(h=>`<th>${h}</th>`).join("") + '</tr></thead><tbody>';
  if(!dataRows.length){html+='<tr><td colspan="'+headers.length+'"><em>No data rows</em></td></tr>';}
  dataRows.forEach(row=>{
    html+='<tr>'+headers.map((_,i)=>`<td>${row[i]!==undefined?row[i]:""}</td>`).join("")+'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('csvTableWrapper').innerHTML = html;

  // CREATE TABLE SQL (not displayed, but set in context)
  let colDefs = headers.map((h,i)=>`  ${h} ${types[i]||"TEXT"}`).join(",\n");
  let create = `CREATE TABLE table_name (\n${colDefs}\n);`
  if(document.getElementById('context')) {
    document.getElementById('context').value = create;
  }
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
        start = s.find("<SQL>")
        end = s.find("</SQL>")
        if start != -1 and end != -1:
            body = s[start+len("<SQL>"):end].strip()
        else:
            body = s
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
    top_k: int = 1,
    top_p: float = 1,
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
        snapshots.append({"text": s0.replace(mask_token, " ____").replace("<pad>", ""), "step": 0, "total_steps": total_steps})
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
                snapshots.append({"text": s.replace(mask_token, " ____").replace("<pad>", ""), "step": step_idx+1, "total_steps": total_steps})
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
            snapshots.append({"text": s.replace(mask_token, " ____").replace("<pad>", ""), "step": step_idx+1, "total_steps": total_steps})
            if on_snapshot:
                on_snapshot(snapshots[-1])

    t1 = time.time()
    log(f"[INFO] Denoising took {t1 - t0:.2f}s")

    decoded_full = tokenizer.decode(current_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    display = decoded_full.replace(mask_token, "_____").replace("<pad>", "")
    display = display.replace("_____", "")
    try:
        start_marker = "<SQL>"
        end_marker = "</SQL>"
        start_idx = display.index(start_marker) + len(start_marker)
        end_idx = display.index(end_marker, start_idx)
        sql_only = display[start_idx:end_idx].strip()
    except ValueError:
        token_slice = current_ids[0, sql_open_idx + 1 : sql_close_idx].detach().cpu().tolist()
        sql_only = tokenizer.decode(token_slice, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        sql_only = sql_only.replace("____", "")


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
    top_k = int(form.get("top_k", "1"))
    top_p = float(form.get("top_p", "1"))
    model_dir = form.get("model_dir", DEFAULT_MODEL_DIR).strip() or DEFAULT_MODEL_DIR
    animate = True
    gif_size_text = form.get("gif_size", "").strip()
    interval_ms = 500

    if not prompt:
        return "Prompt is empty", 400
    if not os.path.isdir(model_dir):
        return jsonify({"error": f"Model dir not found: {model_dir}"}), 400

    run_id = uuid.uuid4().hex
    RUNS[run_id] = {
        "snapshots": [],
        "snapshot_queue": Queue(),
        "status": "queued",
        "done": False,
        "sql_only": None,
        "display": None,
        "gif_url": None
    }
    def status_cb(msg):
        RUNS[run_id]["status"] = msg

    def on_snapshot_callback(snapshot_obj):
        RUNS[run_id]["snapshots"].append(snapshot_obj)
        RUNS[run_id]["snapshot_queue"].put(snapshot_obj)

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
            # ensure output dir
            out_dir = os.path.join(os.path.abspath("."), "generated_gifs")
            os.makedirs(out_dir, exist_ok=True)

            # always build GIF
            RUNS[run_id]["status"] = "building GIF"
            try:
                if gif_size_text:
                    try:
                        w,h = gif_size_text.lower().split('x')
                        gif_size = (int(w), int(h))
                    except Exception:
                        gif_size = (1400,900)
                else:
                    gif_size = (1400,900)

                gif_bytes = build_gif_bytes_from_snapshots(RUNS[run_id]["snapshots"], size=gif_size, interval_ms=interval_ms)
                out_name = f"generation_{run_id}.gif"
                # safe filename
                out_name = secure_filename(out_name)
                out_path = os.path.join(out_dir, out_name)
                with open(out_path, "wb") as f:
                    f.write(gif_bytes)
                # use a URL path that maps to our /gif/<filename> endpoint
                RUNS[run_id]["gif_url"] = f"/gif/{out_name}"
                RUNS[run_id]["status"] = f"GIF written to {out_path}"
            except Exception as e:
                # store traceback so UI shows reason
                RUNS[run_id]["status"] = f"GIF build failed: {e}\n{tb}"

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
        q = RUNS[run_id]["snapshot_queue"]
        last_status = ""
        done_sent = False
        while True:
            try:
                snap = q.get(timeout=0.4)  # Non-polling, gets data as soon as available
                yield f"event: snapshot\ndata: {json.dumps(snap)}\n\n"
            except Empty:
                pass

            st = RUNS[run_id].get("status", "")
            if st != last_status:
                last_status = st
                yield f"event: status\ndata: {json.dumps({'msg': st})}\n\n"

            if RUNS[run_id].get("done") and not done_sent:
                payload = {"sql_only": RUNS[run_id].get("sql_only"), "gif_url": RUNS[run_id].get("gif_url")}
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                done_sent = True
                break
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

from werkzeug.utils import secure_filename
from flask import abort

@app.route("/gif/<filename>")
def serve_gif(filename):
    # prevent path traversal
    safe_name = secure_filename(filename)
    out_dir = os.path.join(os.path.abspath("."), "generated_gifs")
    p = os.path.join(out_dir, safe_name)
    if not os.path.isfile(p):
        return "Not found", 404
    # Force as-attachment download (Flask 2.0+ uses download_name)
    try:
        return send_file(p, mimetype="image/gif", as_attachment=True, download_name=safe_name)
    except TypeError:
        # fallback for older Flask versions
        return send_file(p, mimetype="image/gif", as_attachment=True, attachment_filename=safe_name)


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
