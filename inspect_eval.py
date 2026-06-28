"""Qualitative eval for the SQL masked-diffusion model.

Loads a checkpoint and generates SQL for (a) held-out gretelai test examples and
(b) a set of hard-coded simple sanity cases, printing pred vs gold so we can tell
whether a low exact-match is a strict-string-metric artifact or genuinely weak
generation. Runs on GPU if available, else CPU.

Usage: python3 inspect_eval.py [CKPT_DIR] [N_DATASET] [GEN_STEPS] [SQL_LEN]
"""
import os, re, sys
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_FLAX", "0")

import math
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from denoising import denoise_steps

CKPT = sys.argv[1] if len(sys.argv) > 1 else "diffusion-sql-modernbert"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
GEN_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
MAX_LEN = int(os.environ.get("INSPECT_MAX_LEN", "512"))
SQL_WINDOW = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("INSPECT_SQL_LEN", "128"))
BLOCK_SIZE = max(1, int(os.environ.get("INSPECT_BLOCK_SIZE", "32")))
EOS_BIAS = float(os.environ.get("INSPECT_EOS_BIAS", "1.5"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(CKPT)
# sdpa avoids flash-attn's fp32 restriction; works on GPU and CPU.
model = AutoModelForMaskedLM.from_pretrained(CKPT, attn_implementation="sdpa").eval().to(DEV)
SQL_MODE = str(getattr(model.config, "diffusion_sql_mode", os.environ.get("SQL_GENERATION_MODE", "fixed"))).lower()
if SQL_MODE not in {"block", "fixed"}:
    SQL_MODE = "fixed"
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
TID = {t: tok.convert_tokens_to_ids(t) for t in TAGS}
CLS, SEP = tok.cls_token_id, tok.sep_token_id
PAD, MASK = tok.pad_token_id, tok.mask_token_id


def encode(prompt, context, sql=""):
    pi = tok(prompt, add_special_tokens=False)["input_ids"]
    ci = tok(context, add_special_tokens=False)["input_ids"]
    si = tok(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
    si = si + [PAD] * (SQL_WINDOW - len(si))
    budget = MAX_LEN - SQL_WINDOW - 9
    if len(pi) + len(ci) > budget:
        ci = ci[: max(0, budget - len(pi))]; pi = pi[:budget]
    ids = [CLS, TID['<PROMPT>']] + pi + [TID['</PROMPT>'], TID['<CONTEXT>']] + ci + [TID['</CONTEXT>'], TID['<SQL>']]
    lo = len(ids); ids = ids + si + [TID['</SQL>'], SEP]; hi = lo + SQL_WINDOW
    attn = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD] * (MAX_LEN - len(ids))
    return ids, attn, lo, hi


def norm(s):
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


def trim_sql(s):
    s = re.sub(r"\s+", " ", (s or "").replace("</SQL>", " ")).strip()
    semi = s.find(";")
    if semi >= 0:
        s = s[:semi]
    return s.strip()


@torch.no_grad()
def generate_fixed(prompt, context):
    ids, attn, lo, hi = encode(prompt, context)
    ids = torch.tensor([ids], device=DEV); attn = torch.tensor([attn], device=DEV)
    ids[0, lo:hi] = MASK
    for _ in denoise_steps(model, ids, attn, list(range(lo, hi)), MASK,
                           n_steps=GEN_STEPS, forbid_token_ids=list(TID.values())):
        pass
    out = [t for t in ids[0, lo:hi].tolist() if t not in (PAD, MASK)]
    return trim_sql(tok.decode(out, skip_special_tokens=True))


@torch.no_grad()
def generate_block(prompt, context):
    pi = tok(prompt, add_special_tokens=False)["input_ids"]
    ci = tok(context, add_special_tokens=False)["input_ids"]
    budget = MAX_LEN - SQL_WINDOW - 8
    if len(pi) + len(ci) > budget:
        ci = ci[: max(0, budget - len(pi))]; pi = pi[:budget]
    head = [CLS, TID['<PROMPT>']] + pi + [TID['</PROMPT>'], TID['<CONTEXT>']] + ci + [TID['</CONTEXT>'], TID['<SQL>']]
    generated = []
    end_id = TID['</SQL>']
    forbid = set(tok.all_special_ids or [])
    forbid.update(TID.values())
    forbid.discard(end_id)
    forbid.discard(MASK)
    max_blocks = int(math.ceil(SQL_WINDOW / BLOCK_SIZE))
    for _block in range(max_blocks):
        block_len = min(BLOCK_SIZE, SQL_WINDOW - len(generated))
        if block_len <= 0:
            break
        ids = head + generated + [MASK] * block_len + [SEP]
        attn = [1] * len(ids)
        if len(ids) < MAX_LEN:
            attn += [0] * (MAX_LEN - len(ids))
            ids += [PAD] * (MAX_LEN - len(ids))
        ids = torch.tensor([ids[:MAX_LEN]], device=DEV)
        attn = torch.tensor([attn[:MAX_LEN]], device=DEV)
        start = len(head) + len(generated)
        positions = list(range(start, min(start + block_len, MAX_LEN - 1)))
        for _ in denoise_steps(
            model, ids, attn, positions, MASK,
            n_steps=min(GEN_STEPS, len(positions)),
            forbid_token_ids=list(forbid),
            bias_token_ids={end_id: EOS_BIAS} if EOS_BIAS else None,
        ):
            pass
        block = [int(t) for t in ids[0, start:start + len(positions)].tolist()]
        if end_id in block:
            generated.extend(block[:block.index(end_id)])
            break
        generated.extend(block)
        text = trim_sql(tok.decode([t for t in generated if t not in (PAD, MASK)], skip_special_tokens=True))
        if text.endswith(";"):
            break
    return trim_sql(tok.decode([t for t in generated if t not in (PAD, MASK)], skip_special_tokens=True))


def generate(prompt, context):
    if SQL_MODE == "block":
        return generate_block(prompt, context)
    return generate_fixed(prompt, context)


def report(items, header):
    """items: list of (prompt, context, gold). Returns (exact, soft) fractions."""
    print(f"\n{'='*70}\n{header}\n{'='*70}")
    exact = soft = 0
    for i, (p, c, gold) in enumerate(items):
        pred = generate(p, c)
        em = norm(pred) == norm(gold)
        a, b = set(norm(pred).split()), set(norm(gold).split())
        jacc = len(a & b) / max(1, len(a | b))
        exact += em; soft += jacc >= 0.9
        flag = "✅" if em else ("≈" if jacc >= 0.9 else "❌")
        print(f"\n[{i}] {flag} EM={em} jacc={jacc:.2f}")
        print(f"  Q:    {p[:120]}")
        print(f"  GOLD: {norm(gold)[:170]}")
        print(f"  PRED: {norm(pred)[:170]}")
    n = max(1, len(items))
    print(f"\n--- {header}: exact={exact/n:.2f}  soft(jacc>=.9)={soft/n:.2f}  (n={len(items)}) ---")
    return exact / n, soft / n


# ---------------------------------------------------------------------------
# Hard-coded sanity cases: minimal schemas + unambiguous questions.
# ---------------------------------------------------------------------------
SIMPLE_TESTS = [
    ("How many users are there?",
     "CREATE TABLE users (id INT, name TEXT, age INT);",
     "SELECT count(*) FROM users"),
    ("List all product names.",
     "CREATE TABLE products (id INT, name TEXT, price INT);",
     "SELECT name FROM products"),
    ("What is the average price of products?",
     "CREATE TABLE products (id INT, name TEXT, price INT);",
     "SELECT avg(price) FROM products"),
    ("What is the maximum salary of employees?",
     "CREATE TABLE employees (id INT, name TEXT, salary INT, department TEXT);",
     "SELECT max(salary) FROM employees"),
    ("Show the names of employees in the Sales department.",
     "CREATE TABLE employees (id INT, name TEXT, salary INT, department TEXT);",
     "SELECT name FROM employees WHERE department = 'Sales'"),
    ("How many orders were placed by customer 5?",
     "CREATE TABLE orders (id INT, customer_id INT, amount INT);",
     "SELECT count(*) FROM orders WHERE customer_id = 5"),
    ("What is the total amount of all orders?",
     "CREATE TABLE orders (id INT, customer_id INT, amount INT);",
     "SELECT sum(amount) FROM orders"),
    ("Delete all orders with amount less than 100.",
     "CREATE TABLE orders (id INT, customer_id INT, amount INT);",
     "DELETE FROM orders WHERE amount < 100"),
]


if __name__ == "__main__":
    print(f"checkpoint={CKPT}  device={DEV}  mode={SQL_MODE}  gen_steps={GEN_STEPS}  sql_len={SQL_WINDOW}  block_size={BLOCK_SIZE}  eos_bias={EOS_BIAS}")
    report(SIMPLE_TESTS, "HARD-CODED SANITY CASES")

    ds = load_dataset("gretelai/synthetic_text_to_sql", split="test")
    dataset_items = [(ds[i]["sql_prompt"], ds[i].get("sql_context", ""), ds[i].get("sql", ""))
                     for i in range(N)]
    report(dataset_items, f"GRETELAI TEST SET (first {N})")
