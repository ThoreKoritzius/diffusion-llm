"""Cohorts, prompts, stop rules and target configs shared by every script of the generalization run."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COHORTS = ('edit261', 'ceb150', 'diff104')
FENCE_LANG = {'python': 'python', 'cpp': 'cpp', 'java': 'java', 'javascript': 'javascript', 'go': 'go',
              'typescript': 'typescript', 'rust': 'rust', 'c': 'c', 'php': 'php', 'ruby': 'ruby'}
LANG_NAME = {'python': 'Python', 'cpp': 'C++', 'java': 'Java', 'javascript': 'JavaScript', 'go': 'Go',
             'typescript': 'TypeScript', 'rust': 'Rust', 'c': 'C', 'php': 'PHP', 'ruby': 'Ruby'}
MAX_MODEL_LEN = 8192
DIFF_HEAD = '<<<<<<< SEARCH\n'


# Text-to-SQL drafters for Qwen3-8B (README: "Text-to-SQL on Qwen3-8B"). name -> (dir in spec-sql8b/, method, k).
# The *_conv / *_l1 / causal entries are the diagnosis of the causal-mask bug; dflash_sql_nc and eagle_sql are the
# results. Layer ids are hidden_states indices (0 = embeddings), the convention of Speculators and vLLM.
SQL_DRAFTERS = {
    'dflash_conv': ('zlab', 'dflash', 15),            # z-lab drafter converted to Speculators format: served causal
    'dflash_conv_l1': ('zlab_l1', 'dflash', 15),      # same, aux layer ids [1 9 17 25 33]
    'dflash_conv_nc': ('zlab_nc', 'dflash', 15),      # same weights, sliding_window_non_causal=true: bidirectional
    'dflash_sql': ('drafter', 'dflash', 15),          # SQL fine-tune, served as converted (causal, ids [2 10 ..])
    'dflash_sql_l1': ('drafter_l1', 'dflash', 15),    # SQL fine-tune, causal, on its training layers [1 9 ..]
    'dflash_sql_nc': ('drafter_nc', 'dflash', 15),    # SQL fine-tune, bidirectional, training layers: the result
    'eagle_sql': ('eagle', 'eagle3', 7),              # RedHatAI EAGLE-3 fine-tuned on the same data and recipe
}


def targets():
    return json.loads((HERE / 'targets.json').read_text())


def vllm_configs(key, extra=False):
    """Method name -> vLLM speculative config (None = plain AR)."""
    t = targets()[key]
    cfg = {'ar': None}
    for k in (7, 15, 31):
        cfg[f'ngram{k}'] = {'method': 'ngram', 'num_speculative_tokens': k, 'prompt_lookup_min': 2, 'prompt_lookup_max': 5}
    cfg['draft7'] = {'method': 'draft_model', 'model': t['draft'], 'num_speculative_tokens': 7}
    if t.get('eagle'):
        cfg['eagle7'] = {'method': 'eagle3', 'model': t['eagle'], 'num_speculative_tokens': 7}
    cfg['dflash'] = {'method': 'dflash', 'model': t['dflash'], 'num_speculative_tokens': t['dflash_k']}
    if key == 'qwen3-8b':  # text-to-SQL drafters built by train_sql_drafters.sh (only listed once they exist)
        for name, (sub, method, k) in SQL_DRAFTERS.items():
            p = HERE / 'spec-sql8b' / sub
            if p.exists():
                ck = p / 'checkpoint_best' if (p / 'checkpoint_best').exists() else p
                cfg[name] = {'method': method, 'model': str(ck.resolve()), 'num_speculative_tokens': k}
    if extra:  # copy + DFlash router in vLLM (plugin ./copyflash_plugin), and DFlash padded to the same verify width
        opts = {'dflash_k': t['dflash_k'], 'router_min': 8, 'max_n': 24}
        cfg['copyflash'] = {'method': 'copyflash', 'model': t['dflash'], 'num_speculative_tokens': 31,
                            'speculator_options': opts}
        cfg['dflash_pad31'] = {'method': 'copyflash', 'model': t['dflash'], 'num_speculative_tokens': 31,
                               'speculator_options': {**opts, 'mode': 'dflash_only'}}
        for K in (47, 63):  # wider copy window (63 + 1 = 64 tokens is vLLM's largest captured CUDA graph size)
            cfg[f'copyflash{K}'] = {**cfg['copyflash'], 'num_speculative_tokens': K}
        # same configs under new names, for runs with VLLM_BATCH_INVARIANT=1 (bitwise-reproducible kernels): lossless
        # verification must then give output identical to AR on every request
        # prompt lookup with the same 63-token window: the copy-only ("speculative edits") baseline for fast apply
        cfg['ngram63'] = {**cfg['ngram31'], 'num_speculative_tokens': 63}
        cfg['ar_bi'], cfg['copyflash63_bi'] = None, cfg['copyflash63']
        cfg['dflash_pad63'] = {**cfg['dflash_pad31'], 'num_speculative_tokens': 63}
        for rm in (5, 12):  # router threshold ablation (63-token window)
            cfg[f'copyflash63_rm{rm}'] = {**cfg['copyflash63'], 'speculator_options': {**opts, 'router_min': rm}}
    return cfg


def cohort(name, limit=None):
    items = json.loads((HERE / 'data' / f'{name}.json').read_text())
    return items[:limit] if limit else items


def prefill(item):
    return DIFF_HEAD if item['format'] == 'diff' else f"```{FENCE_LANG[item['language']]}\n"


def stop_strings(item):
    return [] if item['format'] == 'diff' else ['```']


def prompt_text(tok, item):
    lang, fence = LANG_NAME[item['language']], FENCE_LANG[item['language']]
    what = {'function': 'function', 'file': 'file', 'program': 'program'}[item['regime']]
    code = f"```{fence}\n{item['old'].rstrip()}\n```"
    if item['format'] == 'apply':  # fast-apply: merge an edit snippet into the full file
        user = (f'Apply the edit below to the following {lang} file. The edit shows only the changed parts; lines marked '
                f'"... existing code ..." stand for unchanged code that must be kept exactly as it is. Reply with only the '
                f'complete updated file in a single {lang} code block.\n\nChange: {item["instruction"]}\n\nOriginal file:\n{code}'
                f"\n\nEdit:\n```{fence}\n{item['edit'].rstrip()}\n```")
    elif item['format'] == 'diff':
        user = (f'Edit the following {lang} {what} according to the change request. Reply only with SEARCH/REPLACE '
                'blocks, one per change. SEARCH must copy the original lines exactly; REPLACE holds the new lines:\n'
                '<<<<<<< SEARCH\n(original lines)\n=======\n(new lines)\n>>>>>>> REPLACE\n\n'
                f"Change request: {item['instruction']}\n\n{code}")
    else:
        user = (f'Revise the following {lang} {what} according to the change request. '
                f'Reply with only the complete revised {what} in a single {lang} code block.\n\n'
                f"Change request: {item['instruction']}\n\n{code}")
    text = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return text + prefill(item)


def max_new_tokens(tok, item):
    n = len(tok.encode(item['old'], add_special_tokens=False))
    if item['format'] == 'apply':  # whole file back; stay inside the context window
        room = MAX_MODEL_LEN - len(tok.encode(prompt_text(tok, item), add_special_tokens=False)) - 16
        return min(room, int(1.3 * n + 256))
    return min(1200, int(0.9 * n + 160)) if item['format'] == 'diff' else min(1400, int(1.6 * n + 96))


def group(item):
    if item['cohort'] == 'edit261':
        if item['regime'] == 'file':
            return 'full file'
        return 'function (CanItEdit)' if item['source'] == 'canitedit' else 'function (EditPackFT)'
    if item['cohort'] == 'ceb150':
        return f"bug fix ({LANG_NAME[item['language']]})"
    if item['cohort'] == 'apply100':
        return f"fast apply ({LANG_NAME[item['language']]})"
    return 'search/replace diff'


def stopped(tok, item, out, eos):
    if out[-1] in eos:
        return True
    return item['format'] != 'diff' and '```' in tok.decode(out[-6:])
