"""Target verification cost vs. width at realistic cached contexts (Qwen2.5-Coder-1.5B-Instruct, MPS)."""
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

HERE = Path(__file__).resolve().parent
WIDTHS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]


@torch.inference_mode()
def main():
    m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', torch_dtype=torch.float16,
                                             attn_implementation='sdpa').eval().to('mps')
    rows = [json.loads(l) for l in (HERE / 'targets.jsonl').open()]
    rng = random.Random(7)
    sample = rng.sample([r for r in rows if len(r['out_ids']) > 100], 16)
    out = []
    for r in sample:
        ctx = r['prompt_ids'] + r['out_ids'][:len(r['out_ids']) // 2]
        cache = DynamicCache()
        m(input_ids=torch.tensor([ctx], device='mps'), past_key_values=cache, use_cache=True)
        base = len(ctx)
        jobs = [w for w in WIDTHS for _ in range(5)]
        rng.shuffle(jobs)
        for w in [1, 16] + jobs:  # first two are warm-up
            x = torch.randint(100, 30000, (1, w), device='mps')
            torch.mps.synchronize(); t0 = time.perf_counter()
            o = m(input_ids=x, past_key_values=cache, use_cache=True)
            o.logits.argmax(-1).cpu()
            dt = (time.perf_counter() - t0) * 1e3
            cache.crop(base)
            out.append(dict(id=r['id'], ctx=base, w=w, ms=dt))
        out = out[:-len(jobs) - 2] + out[-len(jobs):]
    (HERE / 'cost.json').write_text(json.dumps(out))
    import numpy as np
    for w in WIDTHS:
        print(w, round(float(np.median([o['ms'] for o in out if o['w'] == w])), 2))


if __name__ == '__main__':
    main()
