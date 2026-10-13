"""Execution-reward RL for a masked-diffusion SQL model (GRPO-style, on-policy, no value model).

Per prompt: sample G completions with the model's own diffusion sampler, run each against the example's schema in
SQLite, reward = 1 if it returns the gold result (+ VALID_BONUS if it at least executes). Advantage = reward minus
the group mean (no std scaling). The sequence log-likelihood a masked diffusion model has no closed form for is
estimated with coupled complementary masks (DiffuCoder's coupled-GRPO): one random mask over the window and its
complement, so every completion token is scored exactly once per sample. One gradient step per batch, so the
policy ratio is 1 and needs no clipping.

Env vars (defaults in brackets):
  INIT_CHECKPOINT  diffusion checkpoint to start from (required)      OUTPUT_DIR  where to save [checkpoints/rl]
  STEPS [300]  PROMPTS_PER_STEP [8]  GROUP [8]  LR [1e-6]  WINDOW [64]  SAMPLE_STEPS [8]  TEMPERATURE [1.0]
  VALID_BONUS [0.2]  EVAL_EVERY [50]  EVAL_SIZE [200]  KL_BETA [0.0] (k3 KL to the initial model on scored tokens)
  TRAIN_ROWS [100000] (prompts come from these train rows; eval uses the next EVAL_SIZE rows, never the test split)
  DEVICE [cuda if available]  DTYPE [bfloat16 on cuda, float32 on cpu] (autocast; weights stay fp32)  SEED [0]
"""

import json
import math
import os
import random
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.common import compiles, run_sql  # noqa: E402
from prompt_isolation import is_isolated_checkpoint, isolate_prompt, set_prompt_len  # noqa: E402

env = os.environ.get
INIT = env("INIT_CHECKPOINT")
OUT = env("OUTPUT_DIR", "checkpoints/rl")
STEPS = int(env("STEPS", "300"))
PROMPTS = int(env("PROMPTS_PER_STEP", "8"))
G = int(env("GROUP", "8"))
LR = float(env("LR", "1e-6"))
W = int(env("WINDOW", "64"))
SAMPLE_STEPS = int(env("SAMPLE_STEPS", "8"))
TEMP = float(env("TEMPERATURE", "1.0"))
VALID_BONUS = float(env("VALID_BONUS", "0.2"))
EVAL_EVERY = int(env("EVAL_EVERY", "50"))
EVAL_SIZE = int(env("EVAL_SIZE", "200"))
KL_BETA = float(env("KL_BETA", "0.0"))
TRAIN_ROWS = int(env("TRAIN_ROWS", "100000"))
DEVICE = torch.device(env("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DTYPE = getattr(torch, env("DTYPE", "bfloat16" if DEVICE.type == "cuda" else "float32"))
TAGS = ["<PROMPT>", "</PROMPT>", "<CONTEXT>", "</CONTEXT>", "<SQL>", "</SQL>"]
MAX_LEN = 512


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Policy:
    def __init__(self, path: str, trainable: bool):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tid = {t: self.tok.convert_tokens_to_ids(t) for t in TAGS}
        # fp32 master weights for the policy (LR ~1e-6 updates vanish in bf16); bf16 autocast for the forward
        self.model = AutoModelForMaskedLM.from_pretrained(path, attn_implementation="sdpa",
                                                          torch_dtype=torch.float32 if trainable else DTYPE)
        self.isolated = is_isolated_checkpoint(self.model.config)
        if self.isolated:
            isolate_prompt(self.model)
        self.model.to(DEVICE).train(trainable)
        if not trainable:
            self.model.requires_grad_(False)
        self.mask, self.pad = self.tok.mask_token_id, self.tok.pad_token_id
        self.forbid = torch.tensor(sorted(set(self.tid.values()) | {self.mask} | set(self.tok.all_special_ids)
                                          - {self.pad}), device=DEVICE)

    def encode(self, prompt: str, context: str):
        """[CLS] <PROMPT> p </PROMPT> <CONTEXT> c </CONTEXT> <SQL> [MASK]*W </SQL> [SEP] (as in bench/backends.py)."""
        t, tok = self.tid, self.tok
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(context, add_special_tokens=False)["input_ids"]
        budget = MAX_LEN - W - 9
        if len(p) + len(c) > budget:
            c = c[: max(0, budget - len(p))]
            p = p[:budget]
        ids = [tok.cls_token_id, t["<PROMPT>"]] + p + [t["</PROMPT>"], t["<CONTEXT>"]] + c + [t["</CONTEXT>"], t["<SQL>"]]
        lo = len(ids)
        return ids + [self.mask] * W + [t["</SQL>"], tok.sep_token_id], lo

    def window_logits(self, ids: torch.Tensor, lo: int) -> torch.Tensor:
        if self.isolated:
            set_prompt_len(self.model, lo)
        with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DTYPE != torch.float32):
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, lo:lo + W].float()
        logits[..., self.forbid] = -float("inf")
        return logits

    @torch.no_grad()
    def sample(self, prompt: str, context: str, n: int, temperature: float):
        """n completions in parallel: each step samples every masked position (temperature, top-k 20, or greedy),
        then commits the most confident ones on the cosine schedule (same schedule as src/denoising.py)."""
        ids, lo = self.encode(prompt, context)
        x = torch.tensor([ids] * n, device=DEVICE)
        for step in range(SAMPLE_STEPS):
            masked = x[:, lo:lo + W] == self.mask
            if not masked.any():
                break
            logits = self.window_logits(x, lo)
            if temperature > 0:
                vals, idx = torch.topk(logits / temperature, 20, dim=-1)
                choice = torch.distributions.Categorical(logits=vals).sample()
                pred = idx.gather(-1, choice[..., None]).squeeze(-1)
            else:
                pred = logits.argmax(-1)
            conf = torch.log_softmax(logits, -1).gather(-1, pred[..., None]).squeeze(-1)
            conf = conf.masked_fill(~masked, -float("inf"))
            keep_masked = 0 if step + 1 >= SAMPLE_STEPS else int(W * math.cos(math.pi / 2 * (step + 1) / SAMPLE_STEPS))
            n_commit = (masked.sum(1) - keep_masked).clamp(min=1)
            order = conf.argsort(dim=1, descending=True)
            rank = order.argsort(dim=1)
            commit = (rank < n_commit[:, None]) & masked
            window = x[:, lo:lo + W]
            x[:, lo:lo + W] = torch.where(commit, pred, window)
        return x, lo

    def decode(self, x: torch.Tensor, lo: int):
        out = []
        for row in x[:, lo:lo + W].tolist():
            toks = [t for t in row if t not in (self.pad, self.mask)]
            out.append(self.tok.decode(toks, skip_special_tokens=True).strip())
        return out

    def coupled_logp(self, x: torch.Tensor, lo: int, gen: torch.Generator):
        """Per-sample sum of log p(token) over the window, each position scored once: a random mask and its
        complement (two forwards). Returns (logp [n], per-token logp [n, W] for the KL term, masks used)."""
        n = x.shape[0]
        target = x[:, lo:lo + W]
        t = torch.rand(n, 1, generator=gen, device="cpu").clamp(0.2, 0.8).to(DEVICE)
        m = torch.rand(n, W, generator=gen, device="cpu").to(DEVICE) < t
        tok_lp = torch.zeros(n, W, device=DEVICE)
        for mask in (m, ~m):
            inp = x.clone()
            inp[:, lo:lo + W] = torch.where(mask, torch.full_like(target, self.mask), target)
            lp = torch.log_softmax(self.window_logits(inp, lo), -1).gather(-1, target[..., None]).squeeze(-1)
            tok_lp = tok_lp + torch.where(mask, lp, torch.zeros_like(lp))
        return tok_lp.sum(1), tok_lp, m


def reward(context: str, gold_res, pred: str) -> float:
    try:
        res = run_sql(context, pred)
    except Exception:
        return 0.0
    return VALID_BONUS + (1.0 if res == gold_res else 0.0)


def load_rows():
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
    rows = [{"prompt": e["sql_prompt"], "context": e["sql_context"] or "", "sql": e["sql"]}
            for e in ds.select(range(min(len(ds), TRAIN_ROWS + EVAL_SIZE)))]
    return rows[:TRAIN_ROWS], rows[TRAIN_ROWS:TRAIN_ROWS + EVAL_SIZE]


def with_gold(row):
    try:
        return run_sql(row["context"], row["sql"])
    except Exception:
        return None


@torch.no_grad()
def evaluate(policy: Policy, rows) -> dict:
    policy.model.eval()
    ex = valid = n = 0
    for r in rows:
        gold = with_gold(r)
        if gold is None:
            continue
        x, lo = policy.sample(r["prompt"], r["context"], 1, 0.0)
        pred = policy.decode(x, lo)[0]
        n += 1
        valid += compiles(r["context"], pred)
        try:
            ex += run_sql(r["context"], pred) == gold
        except Exception:
            pass
    policy.model.train()
    return {"exec": ex / max(n, 1), "valid": valid / max(n, 1), "n": n}


def main():
    assert INIT, "set INIT_CHECKPOINT"
    random.seed(int(env("SEED", "0")))
    gen = torch.Generator().manual_seed(int(env("SEED", "0")))
    os.makedirs(OUT, exist_ok=True)
    policy = Policy(INIT, trainable=True)
    ref = Policy(INIT, trainable=False) if KL_BETA > 0 else None
    opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    train_rows, eval_rows = load_rows()
    log(f"rl from {INIT}: {len(train_rows)} train prompts, {len(eval_rows)} eval, device {DEVICE} {DTYPE}")
    metrics_path = os.path.join(OUT, "rl_metrics.jsonl")
    best = evaluate(policy, eval_rows)
    log(f"step 0 eval {best}")
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"step": 0, **best}) + "\n")
    policy.model.save_pretrained(OUT)
    policy.tok.save_pretrained(OUT)
    order = list(range(len(train_rows)))
    random.shuffle(order)
    cursor = 0
    for step in range(1, STEPS + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        rewards, used, losses = [], 0, []
        while used < PROMPTS:
            r = train_rows[order[cursor % len(order)]]
            cursor += 1
            gold = with_gold(r)
            if gold is None:
                continue
            policy.model.eval()
            x, lo = policy.sample(r["prompt"], r["context"], G, TEMP)
            policy.model.train()
            rew = torch.tensor([reward(r["context"], gold, p) for p in policy.decode(x, lo)], device=DEVICE)
            rewards.extend(rew.tolist())
            used += 1
            adv = rew - rew.mean()
            if adv.abs().max() < 1e-6:  # all samples equally good: no signal
                continue
            logp, tok_lp, _ = policy.coupled_logp(x, lo, gen)
            loss = -(adv * logp).sum() / (G * W * PROMPTS)
            if ref is not None:
                with torch.no_grad():
                    gen_state = gen.get_state()
                    _, ref_tok_lp, _ = ref.coupled_logp(x, lo, gen)
                    gen.set_state(gen_state)
                d = ref_tok_lp - tok_lp
                loss = loss + KL_BETA * (torch.exp(d) - d - 1).sum() / (G * W * PROMPTS)
            loss.backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
        opt.step()
        mean_r = sum(rewards) / len(rewards)
        log(f"step {step}: reward {mean_r:.3f}, updates {len(losses)}/{PROMPTS}, {time.time() - t0:.1f}s")
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "reward": mean_r, "updates": len(losses)}) + "\n")
        if step % EVAL_EVERY == 0 or step == STEPS:
            m = evaluate(policy, eval_rows)
            log(f"step {step} eval {m} (best exec {best['exec']:.3f})")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, **m}) + "\n")
            if m["exec"] > best["exec"]:
                best = m
                policy.model.save_pretrained(OUT)
                policy.tok.save_pretrained(OUT)
                log(f"saved {OUT}")
    log(f"done, best eval {best}")


if __name__ == "__main__":
    main()
