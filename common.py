import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/dataset.csv")
RESULTS = Path("results")
LABELS = ["claude", "chatgpt", "gemini"]
L2I = {l: i for i, l in enumerate(LABELS)}
PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"

# words, newlines, and single punctuation / markdown symbols are all tokens,
# so formatting (**, #, -, newlines, ...) is visible to the models.
TOKEN_RE = re.compile(r"\w+|\n|[^\w\s]")
# tokens removed by --strip_format (used for the RQ4 ablation)
FORMAT_TOKENS = {"\n", "*", "#", "-", "`", "|", ">", "_", "~"}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["y"] = df["llm_name"].map(L2I)
    return df


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def get_tokens(inp, out, mode, strip_format=False, in_max=40):
    """mode: 'input' | 'output' | 'both' (input truncated to in_max tokens + <sep> + output)."""
    t_in, t_out = tokenize(inp), tokenize(out)
    if strip_format:
        t_in = [t for t in t_in if t not in FORMAT_TOKENS]
        t_out = [t for t in t_out if t not in FORMAT_TOKENS]
    if mode == "input":
        return t_in
    if mode == "output":
        return t_out
    return t_in[:in_max] + [SEP] + t_out


def build_vocab(token_lists, min_freq=2, max_size=20000):
    counts = Counter(t for toks in token_lists for t in toks)
    itos = [PAD, UNK] + [t for t, c in counts.most_common(max_size) if c >= min_freq]
    return {t: i for i, t in enumerate(itos)}


def encode(tokens, stoi):
    return [stoi.get(t, 1) for t in tokens]


def make_folds(df, k=5, seed=0):
    """k-fold CV split by prompt_id, stratified by category. Returns [(train_idx, test_idx), ...]."""
    rng = np.random.RandomState(seed)
    fold_of = {}
    for _, g in df.groupby("category"):
        pids = np.array(sorted(g.prompt_id.unique()))
        rng.shuffle(pids)
        for i, p in enumerate(pids):
            fold_of[p] = i % k
    f = df.prompt_id.map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def make_domain_folds(df):
    """Leave-one-category-out (RQ3). Returns [(train_idx, test_idx, held_out_category), ...]."""
    out = []
    for c in sorted(df.category.unique()):
        out.append((np.where(df.category != c)[0], np.where(df.category == c)[0], c))
    return out


def split_val(train_idx, df, frac=0.15, seed=0):
    """Hold out a validation set (by prompt_id) from the training indices, for early stopping."""
    rng = np.random.RandomState(seed)
    sub = df.iloc[train_idx]
    val_prompts = set()
    for _, g in sub.groupby("category"):
        pids = np.array(sorted(g.prompt_id.unique()))
        rng.shuffle(pids)
        n = max(1, int(round(len(pids) * frac)))
        val_prompts.update(pids[:n].tolist())
    is_val = sub.prompt_id.isin(val_prompts).values
    return train_idx[~is_val], train_idx[is_val]
