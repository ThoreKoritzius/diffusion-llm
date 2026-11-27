"""Sync the Qwen3-8B text-to-SQL runs from the GPU box to W&B (project sql-diffusion), from this Mac.

The box has no W&B login, so training ran without a logger; this pulls the logs and results over ssh and replays
them: one run per drafter fine-tune (every logged step: loss, per-position acceptance, expected accepted length,
learning rate, throughput) and one run with the benchmark table (exec accuracy, latency, speedup, tokens per call).
Re-runnable: runs resume by fixed id and only steps not yet sent are logged.

Usage: python wandb_sync.py HOST     e.g. python wandb_sync.py ubuntu@<box-ip>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
HOST = sys.argv[1]
BOX = '~/2026-11-23_generalization'
LOCAL = HERE / 'results' / 'qwen3-8b' / 'box_sync'
STATE = LOCAL / 'wandb_state.json'
PROJECT = 'sql-diffusion'
SSH = ['ssh', '-i', str(Path.home() / '.ssh/lambda'), '-o', 'ConnectTimeout=15']
KV = re.compile(r'([A-Za-z_][\w/]*)=(-?[0-9][0-9.]*(?:e[-+]?[0-9]+)?)')
TRAIN = {  # run id -> (log on the box, run name, run.yaml on the box)
    'q8b-sql-dflash-ft': ('logs/train_dflash.log', 'qwen3-8b SQL: DFlash drafter fine-tune', 'spec-sql8b/drafter/run.yaml'),
    'q8b-sql-eagle3-ft': ('logs/train_eagle3.log', 'qwen3-8b SQL: EAGLE-3 drafter fine-tune', 'spec-sql8b/eagle/run.yaml'),
}


def pull(remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(['rsync', '-az', '-e', ' '.join(SSH), f'{HOST}:{BOX}/{remote}', str(local)], capture_output=True)
    return r.returncode == 0 and local.exists()


def steps(log_text):
    """Yield (global_step, metrics) for each training log block (rich prints one key=value block per step)."""
    text = log_text.replace('\r', '\n')
    start = text.find('Training epoch 1/')
    if start < 0:
        return
    acc = {}
    for line in text[start:].splitlines():
        for k, v in KV.findall(line):
            if k == 'global_step':
                if acc:
                    yield int(v), acc
                acc = {}
            elif '/' in k or k == 'epoch':
                acc[k] = float(v)


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for rid, (log, name, yml) in TRAIN.items():
        lp = LOCAL / Path(log).name
        if not pull(log, lp):
            continue
        rows = list(steps(lp.read_text(errors='replace')))
        last = state.get(rid, -1)
        new = [(s, m) for s, m in rows if s > last]
        if not new:
            continue
        cfg = {}
        if pull(yml, LOCAL / f'{rid}.yaml'):
            cfg = {'run_yaml': (LOCAL / f'{rid}.yaml').read_text()}
        cfg.update(target='Qwen/Qwen3-8B', data='12k gretelai/synthetic_text_to_sql train rows answered by Qwen3-8B',
                   epochs=3, lr=1e-4, total_seq_len=2048)
        run = wandb.init(project=PROJECT, id=rid, name=name, resume='allow', config=cfg, reinit='finish_previous')
        for s, m in new:
            run.log(m, step=s)
        run.finish()
        state[rid] = new[-1][0]
        print(f'{rid}: logged steps {new[0][0]}..{new[-1][0]}')
    # benchmark table
    if pull('results/qwen3-8b/sql1000/summary.json', LOCAL / 'summary.json'):
        res = json.loads((LOCAL / 'summary.json').read_text())
        cols = ['method', 'exec', 'valid', 'exact', 'lat_mean_ms', 'lat_p50_ms', 'speedup_vs_ar', 'tokens_per_call',
                'same_as_ar', 'n', 'n_exec']
        tab = wandb.Table(columns=cols, data=[[k.removeprefix('vllm_')] + [v.get(c) for c in cols[1:]] for k, v in res.items()])
        run = wandb.init(project=PROJECT, id='q8b-sql-bench', name='qwen3-8b SQL: vLLM benchmark (H100 PCIe, batch 1)',
                         resume='allow', reinit='finish_previous',
                         config=dict(target='Qwen/Qwen3-8B', queries=1000, exec_rows=777, gpu='H100 PCIe', batch=1, greedy=True))
        run.log({'bench': tab, 'speedup_vs_ar': wandb.plot.bar(tab, 'method', 'speedup_vs_ar', title='Speedup vs plain decoding'),
                 'latency_ms': wandb.plot.bar(tab, 'method', 'lat_mean_ms', title='Mean latency per query (ms)')})
        for k, v in res.items():
            for c in ('exec', 'lat_mean_ms', 'speedup_vs_ar', 'tokens_per_call'):
                if v.get(c) is not None:
                    run.summary[f"{k.removeprefix('vllm_')}/{c}"] = v[c]
        run.finish()
        print('bench: logged', ', '.join(res))
    STATE.write_text(json.dumps(state))


if __name__ == '__main__':
    main()
