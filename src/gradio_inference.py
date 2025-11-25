#!/usr/bin/env python3
"""
Gradio-based inference playground for Diffusion-LLM.

This keeps the core denoising pipeline from the Flask app but exposes it as a
Gradio Blocks UI with streaming updates, optional GIF export, and a CSV helper
to auto-generate a table schema for the context field.
"""
import argparse
import csv
import html
import io
import math
import os
import random
import textwrap
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple
from queue import Queue, Empty

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from transformers import RobertaTokenizerFast, RobertaForMaskedLM


DEFAULT_MODEL_DIR = "diffusion-sql"
DEFAULT_SEED = 42
GIF_DIR = os.path.join(os.path.abspath("."), "generated_gifs")
MODEL_CACHE = {"dir": None, "tokenizer": None, "model": None}


# -------------------------
# Utility helpers
# -------------------------
def parse_gif_size(text: str, fallback: Tuple[int, int] = (1400, 900)) -> Tuple[int, int]:
    try:
        if not text:
            return fallback
        w, h = text.lower().split("x")
        return max(64, int(w)), max(64, int(h))
    except Exception:
        return fallback


def extract_sql_only(s: str) -> str:
    start = s.find("<SQL>")
    end = s.find("</SQL>")
    if start != -1 and end != -1:
        body = s[start + len("<SQL>") : end].strip()
        return body or "(empty)"
    return s.strip() or "(empty)"


UI_CSS = """
#live-sql-box textarea {
  font-size: 30px;
  letter-spacing: 0.01em;
  line-height: 1.55;
  font-family: 'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace;
  min-height: 320px;
  max-height: 320px;
  height: 320px;
  resize: none;
}
"""

def format_status(text: str) -> str:
    safe = html.escape(text)
    return safe


# -------------------------
# GIF helpers (copied from Flask UI)
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
        # Pillow 10 removed getsize on FreeTypeFont; use getbbox instead.
        try:
            char_w = f.getsize("M")[0]  # type: ignore[attr-defined]
        except Exception:
            bbox_m = f.getbbox("M") if hasattr(f, "getbbox") else (0, 0, 8, 0)
            char_w = (bbox_m[2] - bbox_m[0]) or 8
        chars_per_line = max(20, int((max_width - padding * 2) / char_w))
        wrapped = textwrap.fill(text, width=chars_per_line)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=f, spacing=6)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w + padding * 2 <= max_width and h + padding * 2 <= max_height:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, size=best) if font_path else ImageFont.load_default()


def render_text_image_for_gif(text: str, width: int, height: int, padding: int = 28):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = pick_font_and_size(draw, text, width, height, padding=padding)
    try:
        char_w = font.getsize("M")[0]  # type: ignore[attr-defined]
    except Exception:
        bbox_m = font.getbbox("M") if hasattr(font, "getbbox") else (0, 0, 8, 0)
        char_w = (bbox_m[2] - bbox_m[0]) or 8
    chars_per_line = max(20, int((width - padding * 2) / char_w))
    wrapped = textwrap.fill(text, width=chars_per_line)
    y = padding
    draw.multiline_text((padding, y), wrapped, fill=(18, 18, 18), font=font, spacing=6)
    return img


def build_gif_bytes_from_snapshots(snapshots: List[Dict], size=(1400, 900), interval_ms: int = 120) -> bytes:
    subframes_per_transition = 6
    blur_max_radius = 14
    hold_frames_start = 2
    hold_frames_end = 8

    def _body_from_snap(snap):
        s = snap.get("text") if isinstance(snap, dict) else str(snap)
        start = s.find("<SQL>")
        end = s.find("</SQL>")
        if start != -1 and end != -1:
            body = s[start + len("<SQL>") : end].strip()
        else:
            body = s
        if not body.strip():
            body = "(empty)"
        return body

    base_imgs = []
    for snap in snapshots:
        body = _body_from_snap(snap)
        img = render_text_image_for_gif(body, width=size[0], height=size[1], padding=28)
        base_imgs.append(img.convert("RGBA"))

    frames = []
    if not base_imgs:
        blank = Image.new("RGB", size, color=(255, 255, 255))
        frames.append(blank.convert("P", palette=Image.ADAPTIVE))
    else:
        for _ in range(hold_frames_start):
            frames.append(base_imgs[0].convert("P", palette=Image.ADAPTIVE))

        for i in range(len(base_imgs) - 1):
            a_img = base_imgs[i]
            b_img = base_imgs[i + 1]
            if a_img.tobytes() == b_img.tobytes():
                frames.append(b_img.convert("P", palette=Image.ADAPTIVE))
                continue

            frames.append(a_img.convert("P", palette=Image.ADAPTIVE))
            for j in range(subframes_per_transition):
                f = (j + 1) / float(subframes_per_transition + 1)
                blur_r = blur_max_radius * (1.0 - f)
                b_blurred = b_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
                blended = Image.blend(a_img, b_blurred, alpha=f)
                frames.append(blended.convert("P", palette=Image.ADAPTIVE))
            frames.append(b_img.convert("P", palette=Image.ADAPTIVE))

        for _ in range(hold_frames_end):
            frames.append(base_imgs[-1].convert("P", palette=Image.ADAPTIVE))

    bio = io.BytesIO()
    frames[0].save(bio, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=interval_ms)
    bio.seek(0)
    return bio.read()


# -------------------------
# Core model helpers
# -------------------------
def load_model_and_tokenizer(model_dir: str, max_len: int = 512):
    if MODEL_CACHE["dir"] == model_dir and MODEL_CACHE["tokenizer"] is not None and MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["tokenizer"], MODEL_CACHE["model"]
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
    tokenizer.model_max_length = max_len
    model = RobertaForMaskedLM.from_pretrained(model_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")
    )
    model.to(device)
    model.eval()
    MODEL_CACHE.update({"dir": model_dir, "tokenizer": tokenizer, "model": model})
    return tokenizer, model


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
    on_snapshot=None,
    status_cb=None,
    deterministic_seed: int = DEFAULT_SEED,
) -> Dict:
    def log(s):
        if status_cb:
            status_cb(str(s))

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

    required_tags = ["<PROMPT>", "</PROMPT>", "<CONTEXT>", "</CONTEXT>", "<SQL>", "</SQL>"]
    missing = [t for t in required_tags if t not in tokenizer.get_vocab()]
    if missing:
        log(f"[INFO] Adding missing special tokens: {missing}")
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
        log("[INFO] Resized embeddings")

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    full_text = f"<PROMPT>{prompt_text}</PROMPT> <CONTEXT>{context_text}</CONTEXT> <SQL></SQL>"
    enc = tokenizer(
        full_text, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt", return_offsets_mapping=True
    )
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
            status_cb(f"step {step_idx + 1}/{total_steps}")

        if math.isclose(p_mask, 0.0):
            new_ids = current_ids.clone()
            new_ids[0, sql_open_idx + 1 : sql_close_idx] = pred_ids[0, sql_open_idx + 1 : sql_close_idx]
            current_ids = new_ids
            if animate:
                s = tokenizer.decode(current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
                snapshots.append({"text": s.replace(mask_token, " ____").replace("<pad>", ""), "step": step_idx + 1, "total_steps": total_steps})
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
            snapshots.append({"text": s.replace(mask_token, " ____").replace("<pad>", ""), "step": step_idx + 1, "total_steps": total_steps})
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
# Gradio wiring
# -------------------------
def stream_generation(
    prompt: str,
    context: str,
    steps: int,
    max_len: int,
    sql_len: int,
    top_k: int,
    top_p: float,
    model_dir: str,
    seed: int,
    gif_size_text: str,
    make_gif: bool,
):
    snapshots_state: List[Dict] = []
    snapshot_queue: Queue = Queue()
    status_queue: Queue = Queue()
    done = threading.Event()
    holder: Dict[str, Optional[Dict]] = {}

    def on_snapshot(snap):
        snapshot_queue.put(snap)

    def status_cb(msg):
        status_queue.put(str(msg))

    def worker():
        try:
            res = run_denoising_generation_callback(
                prompt_text=prompt,
                context_text=context,
                model_dir=model_dir,
                n_steps=steps,
                max_len=max_len,
                top_k=top_k,
                top_p=top_p,
                sql_len_request=sql_len,
                animate=True,
                on_snapshot=on_snapshot,
                status_cb=status_cb,
                deterministic_seed=seed,
            )
            holder["res"] = res
        except Exception as exc:
            holder["error"] = str(exc)
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()

    live_sql = ""
    status_raw = "Preparing run..."
    status_text = format_status(status_raw)
    slider_update = gr.update(visible=False, maximum=max(1, steps), value=0, step=1)
    slider_label = "Scrub diffusion steps"
    gif_file = None
    yield live_sql, slider_update, gif_file, snapshots_state

    while True:
        try:
            snap = snapshot_queue.get(timeout=0.15)
        except Empty:
            snap = None

        try:
            status_raw = status_queue.get_nowait()
        except Empty:
            pass

        if snap:
            snapshots_state.append(snap)
            live_sql = extract_sql_only(snap.get("text", ""))
            slider_update = gr.update(visible=True, maximum=max(1, steps), value=len(snapshots_state) - 1, step=1)

        status_text = format_status(status_raw)
        if snap or status_raw:
            yield live_sql, slider_update, gif_file, snapshots_state

        if done.is_set() and snapshot_queue.empty():
            break

    if "error" in holder:
        status_raw = f"Error: {holder['error']}"
        status_text = format_status(status_raw)
        yield live_sql, slider_update, gif_file, snapshots_state
        return

    res = holder.get("res") or {}

    gif_path = None
    if make_gif and res.get("snapshots"):
        os.makedirs(GIF_DIR, exist_ok=True)
        gif_bytes = build_gif_bytes_from_snapshots(res["snapshots"], size=parse_gif_size(gif_size_text), interval_ms=120)
        gif_path = os.path.join(GIF_DIR, f"generation_{uuid.uuid4().hex}.gif")
        with open(gif_path, "wb") as f:
            f.write(gif_bytes)
    gif_file = gif_path
    status_raw = "Done"
    status_text = format_status(status_raw)
    if res.get("snapshots"):
        last = res["snapshots"][-1]
        live_sql = extract_sql_only(last.get("text", live_sql))
    yield live_sql, slider_update, gif_file, snapshots_state


def snapshot_at_index(idx: int, snapshots: List[Dict]):
    if snapshots is None or len(snapshots) == 0:
        return gr.update()
    idx = int(idx)
    if idx < 0:
        idx = 0
    if idx >= len(snapshots):
        idx = len(snapshots) - 1
    snap = snapshots[idx]
    return extract_sql_only(snap.get("text", ""))


def infer_context_from_csv(file_path: Optional[str]):
    if not file_path:
        return gr.update()
    try:
        with open(file_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return gr.update()
        headers = rows[0]
        data_row = None
        for r in rows[1:]:
            if any(cell.strip() for cell in r):
                data_row = r
                break
        if data_row is None:
            data_row = [""] * len(headers)

        def guess_type(val: str):
            if val.strip().isdigit():
                return "INT"
            try:
                float(val)
                if "." in val:
                    return "FLOAT"
            except Exception:
                pass
            return "TEXT"

        types = [guess_type(v) for v in data_row]
        col_defs = ",\n".join([f"  {h} {t}" for h, t in zip(headers, types)])
        create_stmt = f"CREATE TABLE table_name (\n{col_defs}\n);"
        return gr.update(value=create_stmt)
    except Exception as exc:
        return gr.update(value=f"-- Failed to parse CSV: {exc}")


def build_interface(default_prompt: str, default_context: str, default_model_dir: str, default_seed: int, default_gif_size: str, enable_gif: bool):
    with gr.Blocks(title="Diffusion LLM (Gradio)") as demo:
        gr.HTML(f"<style>{UI_CSS}</style>")
        gr.Markdown(
            "### Diffusion LLM — Gradio Inference\n"
            "Stream text-to-SQL diffusion, watch each denoising step, and export animations."
        )

        snapshots_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                prompt_in = gr.Textbox(label="Prompt", value=default_prompt, lines=3)

                with gr.Tabs():
                    with gr.TabItem("Manual context"):
                        context_in = gr.Textbox(label="Context (database definition)", value=default_context, lines=6)
                    with gr.TabItem("CSV upload"):
                        csv_in = gr.File(label="CSV for schema (auto-fills context)", file_types=[".csv"], type="filepath")
                        gr.Markdown("Upload a CSV to auto-build a CREATE TABLE stub.")
                csv_in.change(fn=infer_context_from_csv, inputs=csv_in, outputs=[context_in], queue=False)

                with gr.Accordion("Advanced controls", open=False):
                    steps_in = gr.Slider(1, 50, value=10, step=1, label="Steps (max 50)")
                    max_len_in = gr.Slider(64, 4096, value=512, step=64, label="Max length")
                    sql_len_in = gr.Slider(1, 1024, value=64, step=1, label="SQL mask length")
                    top_k_in = gr.Slider(0, 2000, value=1, step=1, label="Top-k (0 = off)")
                    top_p_in = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p (0 = off)")
                    model_dir_in = gr.Textbox(label="Model directory", value=default_model_dir)
                    seed_in = gr.Number(label="Seed", value=default_seed, precision=0)

                run_btn = gr.Button("Run Diffusion", variant="primary")
            with gr.Column(scale=2):
                live_sql_out = gr.Textbox(label="Live SQL", lines=9, elem_id="live-sql-box")
               # status_out = gr.Textbox(label="", value="Idle", elem_id="status-box", lines=2)
                replay_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0, visible=False, label="Scrub diffusion steps", interactive=True)
                gif_out = gr.File(label="GIF", visible=False)

        run_btn.click(
            fn=stream_generation,
            inputs=[
                prompt_in,
                context_in,
                steps_in,
                max_len_in,
                sql_len_in,
                top_k_in,
                top_p_in,
                model_dir_in,
                seed_in
            ],
            outputs=[live_sql_out, replay_slider, gif_out, snapshots_state],
            queue=True,
        )

        replay_slider.change(
            fn=snapshot_at_index,
            inputs=[replay_slider, snapshots_state],
            outputs=[live_sql_out],
            queue=False,
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio inference UI for Diffusion-LLM")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", type=str, default="how many planes are of name a380?")
    parser.add_argument("--context", type=str, default="CREATE TABLE planes (id INT, name TEXT, weight NUMBER)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gif-size", type=str, default="1400x900")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF building and hide related controls")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_interface(
        default_prompt=args.prompt,
        default_context=args.context,
        default_model_dir=args.model_dir,
        default_seed=args.seed,
        default_gif_size=args.gif_size,
        enable_gif=not args.no_gif,
    )
    demo.queue(api_open=False)
    demo.launch(server_name=args.host,
                server_port=args.port,
                share=args.share,
                inbrowser=not args.share,
                theme=gr.themes.Ocean())


if __name__ == "__main__":
    main()
