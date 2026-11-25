"""Cohort, prompts and limits shared by every H100 script (identical to the Mac runs in 2026-11-21_dflash-mlx)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FENCE = '```python\n'


def cohort(limit=None):
    items = json.loads((HERE / 'data' / 'cohort.json').read_text())
    return items[:limit] if limit else items


def prompt_text(tok, item):
    what = 'function' if item['regime'] == 'function' else 'file'
    user = (f'Revise the following Python {what} according to the change request. '
            f'Reply with only the complete revised {what} in a single Python code block.\n\n'
            f"Change request: {item['instruction']}\n\n```python\n{item['old'].rstrip()}\n```")
    text = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return text + FENCE


def max_new_tokens(tok, item):
    return min(1400, int(1.6 * len(tok.encode(item['old'], add_special_tokens=False)) + 96))


def group(item):
    if item['regime'] == 'file':
        return 'full file'
    return 'function (CanItEdit)' if item['source'] == 'canitedit' else 'function (EditPackFT)'
