"""Lexical SQL constraint for masked diffusion decoding (DINGO-style).

Each denoising step, instead of taking the per-position argmax, a dynamic program finds the most probable
token sequence for the whole SQL window that a small automaton accepts. Committed tokens are fixed; masked
positions use the model's log-probabilities. The automaton checks what the error analysis showed going wrong:

- every identifier-like word (run of letter/digit/underscore tokens) is an SQL keyword or function, a word
  from the schema or prompt, a word frequent in training SQL, a short alias (<= SHORT_ALIAS chars), a number,
  or the name right after AS;
- parentheses balance (outside string literals), string literals close, and only padding follows the end.

Words are matched on their canonical tokenization (with and without a leading space), which is what the model
was trained on.
"""

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SHORT_ALIAS = 4
AS_NAME_TOKENS = 5
MAX_DEPTH = 4
NEG = -1e30

KEYWORDS = """
abort action add after all alter always analyze and as asc attach autoincrement before begin between by cascade
case cast check collate column commit conflict constraint create cross current current_date current_time
current_timestamp database default deferrable deferred delete desc detach distinct do drop each else end escape
except exclude exclusive exists explain fail filter first following for foreign from full generated glob group
groups having if ignore immediate in index indexed initially inner insert instead intersect into is isnull join
key last left like limit match materialized natural no not nothing notnull null nulls of offset on or order
others outer over partition plan pragma preceding primary query raise range recursive references regexp reindex
release rename replace restrict returning right rollback row rows savepoint select set table temp temporary then
ties to transaction trigger unbounded union unique update using vacuum values view virtual when where window with
without true false interval top ilike similar lateral fetch next only percent
count sum avg min max round abs cast coalesce ifnull nullif length lower upper trim ltrim rtrim substr substring
instr replace printf format date time datetime julianday strftime unixepoch random total group_concat string_agg
row_number rank dense_rank percent_rank cume_dist ntile lag lead first_value last_value nth_value
extract year month day hour minute second week quarter dayofweek dayofyear weekday now curdate curtime
current_date date_sub date_add datediff date_trunc date_part dateadd getdate timestampdiff to_date to_char
concat concat_ws left right position char_length stddev stddev_pop stddev_samp variance var_pop var_samp median
percentile_cont percentile_disc within array_agg ceil ceiling floor power sqrt exp ln log log10 mod sign
greatest least iif ifnull isnull convert decode nvl listagg
int integer smallint bigint tinyint float real double precision decimal numeric text varchar char boolean bool
blob timestamp serial
""".split()


def _case_variants(word: str) -> List[str]:
    return sorted({word, word.lower(), word.upper(), word[:1].upper() + word[1:].lower()})


class TokenClasses:
    """Per-token facts the automaton needs, for the whole vocabulary."""

    def __init__(self, tok, vocab_size: int):
        self.vocab_size = vocab_size
        n = vocab_size
        self.space = np.zeros(n, bool)
        self.kind = np.full(n, "x", dtype="<U1")  # w: letters/underscore, d: digits, p: other text, x: forbidden
        self.length = np.zeros(n, np.int32)
        self.pad = tok.pad_token_id
        special = set(tok.all_special_ids) | {i for i in range(len(tok)) if tok.convert_ids_to_tokens(i) is None}
        self.text: List[str] = [""] * n
        for i in range(min(n, len(tok))):
            if i in special:
                continue
            piece = tok.convert_ids_to_tokens(i)
            if piece is None or (piece.startswith("<") and piece.endswith(">") and piece in tok.get_vocab()
                                 and i >= len(tok) - 16):
                continue
            text = tok.convert_tokens_to_string([piece])
            self.text[i] = text
            body = text[1:] if text.startswith(" ") else text
            self.space[i] = text.startswith(" ") and body != ""
            if re.fullmatch(r"[A-Za-z_]+", body):
                self.kind[i] = "w"
            elif re.fullmatch(r"[0-9]+", body):
                self.kind[i] = "d"
            elif text:
                self.kind[i] = "p"
            self.length[i] = len(body)
        # outcome of a text token on (in_quote, depth): computed per start state, grouped later
        self.free_fn: Dict[int, Tuple] = {}
        self.quote_fn: Dict[int, Tuple] = {}
        for i in np.flatnonzero(self.kind != "x"):
            text = self.text[i]
            if self.kind[i] == "p":
                self.free_fn[i] = tuple(self._run(text, False, d) for d in range(MAX_DEPTH + 1))
            self.quote_fn[i] = tuple(self._run(text, True, d) for d in range(MAX_DEPTH + 1))

    @staticmethod
    def _run(text: str, in_quote: bool, depth: int):
        for ch in text:
            if ch == "'":
                in_quote = not in_quote
            elif not in_quote and ch == "(":
                depth += 1
            elif not in_quote and ch == ")":
                depth -= 1
            if depth < 0 or depth > MAX_DEPTH:
                return None
        return (in_quote, depth)


class Groups:
    """Token groups whose per-position max (and argmax) the DP needs."""

    def __init__(self, members: Dict[object, List[int]]):
        self.keys = list(members)
        self.idx = [np.asarray(members[k], dtype=np.int64) for k in self.keys]

    def best(self, lp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """lp: (W, V) -> (W, G) max values and (W, G) argmax token ids."""
        vals = np.empty((lp.shape[0], len(self.keys)))
        args = np.empty((lp.shape[0], len(self.keys)), dtype=np.int64)
        for g, idx in enumerate(self.idx):
            sub = lp[:, idx]
            a = sub.argmax(1)
            vals[:, g] = sub[np.arange(lp.shape[0]), a]
            args[:, g] = idx[a]
        return vals, args


class SqlGrammar:
    """Shared (per tokenizer) part: token classes, keyword tokenizations, token groups."""

    def __init__(self, tok, vocab_size: int, extra_words: Sequence[str] = ()):
        self.tok = tok
        self.tc = TokenClasses(tok, vocab_size)
        self._seq_cache: Dict[str, List[Tuple[int, ...]]] = {}
        self.global_words = set()
        for w in list(KEYWORDS) + list(extra_words):
            self.global_words.update(_case_variants(w))
        tc = self.tc
        D = MAX_DEPTH + 1
        # free-context punctuation: group by its effect on every depth
        free: Dict[object, List[int]] = {}
        for i, fn in tc.free_fn.items():
            free.setdefault(fn, []).append(i)
        self.free_groups = Groups(free)
        quote: Dict[object, List[int]] = {}
        for i, fn in tc.quote_fn.items():
            quote.setdefault(fn, []).append(i)
        self.quote_groups = Groups(quote)
        # word-token groups: (space, kind, length capped) for alias / number / free-name transitions
        words: Dict[object, List[int]] = {}
        for i in np.flatnonzero((tc.kind == "w") | (tc.kind == "d")):
            words.setdefault((bool(tc.space[i]), tc.kind[i], min(int(tc.length[i]), SHORT_ALIAS + 1)), []).append(i)
        self.word_groups = Groups(words)
        self.D = D

    def sequences(self, word: str) -> List[Tuple[int, ...]]:
        if word not in self._seq_cache:
            seqs = []
            for text in (word, " " + word):
                ids = tuple(self.tok(text, add_special_tokens=False)["input_ids"])
                if ids and all(i < self.tc.vocab_size and self.tc.kind[i] in "wd" for i in ids):
                    seqs.append(ids)
            self._seq_cache[word] = seqs
        return self._seq_cache[word]

    def for_query(self, prompt: str, context: str) -> "QueryGrammar":
        local = set()
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", context + " " + prompt):
            local.update({w, w.lower()})
        return QueryGrammar(self, self.global_words | local)


class QueryGrammar:
    """Automaton for one query: a token trie of allowed words plus a few free states, per paren depth."""

    # free (non-trie) states, each per depth
    F, Q, NUM, END = 0, 1, 2, 3         # after punctuation / in string / number / padding (depth 0 only)
    SW = 4                               # SW + k - 1: short alias with k characters so far (k = 1..SHORT_ALIAS)
    AS = SW + SHORT_ALIAS                # AS + k - 1: name after AS with k tokens so far (k = 1..AS_NAME_TOKENS)

    def __init__(self, g: SqlGrammar, words):
        self.g = g
        self.children: List[Dict[int, int]] = [{}]
        self._parent, self._token, self._terminal, self._is_as = [-1], [-1], [False], [False]
        self.words = set()
        self.add_words(words)

    def add_words(self, words) -> None:
        """Allow more words (e.g. aliases the query defines with AS). Rebuilds the arrays only if needed."""
        new = [w for w in words if w not in self.words]
        if not new and hasattr(self, "token"):
            return
        children, parent, token, terminal, is_as = self.children, self._parent, self._token, self._terminal, self._is_as
        for w in new:
            self.words.add(w)
            for seq in self.g.sequences(w):
                node = 0
                for t in seq:
                    nxt = children[node].get(t)
                    if nxt is None:
                        nxt = len(parent)
                        children[node][t] = nxt
                        children.append({})
                        parent.append(node)
                        token.append(t)
                        terminal.append(False)
                        is_as.append(False)
                    node = nxt
                terminal[node] = True
                is_as[node] = is_as[node] or w.lower() == "as"
        self.parent = np.asarray(parent[1:]) - 1    # trie node n (0-based, root dropped); -1 = starts a word
        self.token = np.asarray(token[1:])
        self.terminal = np.asarray(terminal[1:])
        self.is_as = np.asarray(is_as[1:])
        self.start_space = self.g.tc.space[self.token]
        self.n_free = self.AS + AS_NAME_TOKENS
        self.word_ends = [self.F, self.NUM] + list(range(self.SW, self.n_free))  # states a word may end in

    def decode(self, fixed: Sequence[Optional[int]], logp: np.ndarray) -> Optional[List[int]]:
        """fixed: per window position a committed token id or None. logp: (W, V) log-probs (rows of fixed
        positions are ignored). Returns the most probable accepted sequence, or None if nothing is accepted."""
        g, tc = self.g, self.g.tc
        W, V = logp.shape
        D = g.D
        lp = logp.copy()
        for i, t in enumerate(fixed):
            if t is not None:
                lp[i] = NEG
                lp[i, t] = 0.0
        pad_lp = lp[:, tc.pad].copy()
        lp[:, tc.kind == "x"] = NEG
        fv, fa = g.free_groups.best(lp)
        qv, qa = g.quote_groups.best(lp)
        wv, wa = g.word_groups.best(lp)
        trie_lp = lp[:, self.token] if len(self.token) else np.zeros((W, 0))

        nf, nt = self.n_free, len(self.token)
        val = np.full((nf, D), NEG)
        tval = np.full((nt, D), NEG)
        val[self.F, 0] = 0.0
        # back pointers: per position, for each state the (previous state id, token); state id = s * D + d,
        # trie states offset by nf * D
        back_s = np.empty((W, (nf + nt) * D), dtype=np.int64)
        back_t = np.empty((W, (nf + nt) * D), dtype=np.int64)
        wkeys = g.word_groups.keys
        for i in range(W):
            nval = np.full((nf, D), NEG)
            ntval = np.full((nt, D), NEG)
            bs = np.full((nf, D), -1, dtype=np.int64)
            bt = np.full((nf, D), -1, dtype=np.int64)

            def offer(s, d, v, prev, t):
                if v > nval[s, d]:
                    nval[s, d] = v
                    bs[s, d] = prev
                    bt[s, d] = t

            # boundary: states after which a new word or punctuation may follow
            term_d = np.where(self.terminal[:, None], tval, NEG)
            bnd_trie = term_d.argmax(0) if nt else np.zeros(D, dtype=np.int64)
            cand_ids = self.word_ends
            cands = [val[s] for s in cand_ids]
            if nt:
                cands.append(term_d[bnd_trie, np.arange(D)])
            stack = np.stack(cands)
            barg = stack.argmax(0)
            bval = stack.max(0)
            bprev = np.array([(cand_ids[barg[d]] * D + d) if barg[d] < len(cand_ids)
                              else ((nf + bnd_trie[d]) * D + d) for d in range(D)])
            as_d = np.where((self.terminal & self.is_as)[:, None], tval, NEG)
            as_arg = as_d.argmax(0) if nt else np.zeros(D, dtype=np.int64)
            as_val = as_d.max(0) if nt else np.full(D, NEG)

            # punctuation from a boundary, and every token inside a string
            for gi, fn in enumerate(g.free_groups.keys):
                for d in range(D):
                    out = fn[d]
                    if out is not None and bval[d] > NEG / 2:
                        q, d2 = out
                        offer(self.Q if q else self.F, d2, bval[d] + fv[i, gi], bprev[d], fa[i, gi])
            for gi, fn in enumerate(g.quote_groups.keys):
                for d in range(D):
                    out = fn[d]
                    if out is not None and val[self.Q, d] > NEG / 2:
                        q, d2 = out
                        offer(self.Q if q else self.F, d2, val[self.Q, d] + qv[i, gi], self.Q * D + d, qa[i, gi])
            # padding ends the SQL
            offer(self.END, 0, max(bval[0], val[self.END, 0]) + pad_lp[i],
                  bprev[0] if bval[0] >= val[self.END, 0] else self.END * D, tc.pad)
            # word tokens: start a short alias / number / free name, or continue one
            for gi, (space, kind, ln) in enumerate(wkeys):
                v_tok, t_tok = wv[i, gi], wa[i, gi]
                for d in range(D):
                    start_v, start_p = (bval[d], bprev[d]) if space else (val[self.F, d], self.F * D + d)
                    if kind == "d":
                        offer(self.NUM, d, start_v + v_tok, start_p, t_tok)
                    elif ln <= SHORT_ALIAS:
                        offer(self.SW + ln - 1, d, start_v + v_tok, start_p, t_tok)
                    if space and as_val[d] > NEG / 2:
                        offer(self.AS, d, as_val[d] + v_tok, (nf + as_arg[d]) * D + d, t_tok)
                    if not space:
                        if kind == "d":
                            offer(self.NUM, d, val[self.NUM, d] + v_tok, self.NUM * D + d, t_tok)
                        for k in range(1, AS_NAME_TOKENS):
                            offer(self.AS + k, d, val[self.AS + k - 1, d] + v_tok, (self.AS + k - 1) * D + d, t_tok)
                        for k in range(1, SHORT_ALIAS + 1):
                            if k + ln <= SHORT_ALIAS:
                                offer(self.SW + k + ln - 1, d, val[self.SW + k - 1, d] + v_tok,
                                      (self.SW + k - 1) * D + d, t_tok)
            # trie: start a word or continue it
            tb_s = np.full((nt, D), -1, dtype=np.int64)
            if nt:
                root = self.parent < 0
                start = np.where(self.start_space[:, None], bval[None, :], val[self.F][None, :])
                start_p = np.where(self.start_space[:, None], bprev[None, :],
                                   (self.F * D + np.arange(D))[None, :])
                prev = np.where(root[:, None], start, tval[np.maximum(self.parent, 0)])
                ntval = prev + trie_lp[i][:, None]
                tb_s = np.where(root[:, None], start_p,
                                (nf + np.maximum(self.parent, 0))[:, None] * D + np.arange(D)[None, :])
            back_s[i, : nf * D] = bs.reshape(-1)
            back_t[i, : nf * D] = bt.reshape(-1)
            back_s[i, nf * D:] = tb_s.reshape(-1)
            back_t[i, nf * D:] = np.repeat(self.token, D) if nt else []
            val, tval = nval, np.maximum(ntval, NEG)

        # accepted ends: padding, or the window ends on a complete word / punctuation at depth 0
        finals = [(val[self.END, 0], self.END * D)]
        for s in self.word_ends:
            finals.append((val[s, 0], s * D))
        if nt:
            t_end = np.where(self.terminal, tval[:, 0], NEG)
            n = int(t_end.argmax())
            finals.append((t_end[n], (nf + n) * D))
        best_v, state = max(finals)
        if best_v < NEG / 2:
            return None
        out = []
        for i in range(W - 1, -1, -1):
            out.append(int(back_t[i, state]))
            state = int(back_s[i, state])
        return out[::-1]


def train_sql_words(min_count: int = 20, rows: int = 100_000) -> List[str]:
    """Identifier-like words frequent in training SQL (aliases, dialect functions), cached next to the vocab."""
    path = os.path.join(os.path.dirname(__file__), ".onnx_cache", f"sql_words-{min_count}-{rows}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from collections import Counter

    from datasets import load_dataset

    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train").select(range(rows))
    counts = Counter()
    for sql in list(ds["sql"]):
        counts.update(w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.sub(r"'[^']*'", "''", sql)))
    words = sorted(w for w, c in counts.items() if c >= min_count)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(words, f)
    return words


AS_ALIAS = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)


def as_aliases(sql: str) -> List[str]:
    return AS_ALIAS.findall(sql)
