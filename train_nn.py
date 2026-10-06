import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (RESULTS, build_vocab, encode, get_tokens, load_data,
                    make_domain_folds, make_folds, split_val)
from models import build_model


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class SeqDataset(Dataset):
    def __init__(self, seqs, ys, max_len):
        self.seqs, self.ys, self.max_len = seqs, ys, max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        ids = self.seqs[i][: self.max_len]
        x = ids + [0] * (self.max_len - len(ids))   # always pad to max_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(int(self.ys[i]), dtype=torch.long)


@torch.no_grad()
def logits_for(model, ds, device):
    model.eval()
    out = []
    for xb, _ in DataLoader(ds, batch_size=128):
        out.append(model(xb.to(device)).cpu())
    return torch.cat(out)


def run_one(model_name, tr_idx, te_idx, tokens, df, args, seed, device):
    """Train on tr_idx (minus a validation slice), early-stop on validation, predict te_idx."""
    tr_idx, va_idx = split_val(tr_idx, df, frac=0.15, seed=seed)
    # vocabulary is built from training data only (no test leakage)
    stoi = build_vocab([tokens[i] for i in np.concatenate([tr_idx, va_idx])], min_freq=args.min_freq)
    ys = df.y.values

    def make_ds(idx):
        return SeqDataset([encode(tokens[i], stoi) for i in idx], ys[idx], args.max_len)

    ds_tr, ds_va, ds_te = make_ds(tr_idx), make_ds(va_idx), make_ds(te_idx)

    set_seed(seed)
    model = build_model(model_name, len(stoi)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    y_va = torch.tensor(ys[va_idx], dtype=torch.long)

    best_acc, best_loss, best_state, bad = -1.0, float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        lg = logits_for(model, ds_va, device)
        v_loss = lossf(lg, y_va).item()
        v_acc = (lg.argmax(1) == y_va).float().mean().item()
        if v_acc > best_acc or (v_acc == best_acc and v_loss < best_loss):
            best_acc, best_loss, bad = v_acc, v_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    return logits_for(model, ds_te, device).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["rq1", "rq2", "rq3"], required=True)
    ap.add_argument("--model", choices=["cnn", "lstm", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=3, help="random restarts per fold")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--strip_format", action="store_true",
                    help="remove newlines and markdown symbols (RQ4 ablation)")
    ap.add_argument("--quick", action="store_true", help="1 fold, 1 seed, 3 epochs (smoke test)")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs = 1, 3

    df = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = {"rq1": ["output"], "rq2": ["input", "output", "both"], "rq3": ["output"]}[args.exp]
    models = ["cnn", "lstm"] if args.model == "both" else [args.model]
    variant = "nofmt" if args.strip_format else "full"
    print(f"device={device}  exp={args.exp}  variant={variant}  models={models}  modes={modes}")

    rows = []
    for mode in modes:
        tokens = [get_tokens(i, o, mode, args.strip_format) for i, o in zip(df.input, df.output)]
        if args.exp == "rq3":
            splits = make_domain_folds(df)
        else:
            splits = [(tr, te, f) for f, (tr, te) in enumerate(make_folds(df, args.folds, seed=0))]
        if args.quick:
            splits = splits[:1]
        for model_name in models:
            for seed in range(args.seeds):
                for tr, te, fold in splits:
                    t0 = time.time()
                    pred = run_one(model_name, tr, te, tokens, df, args, seed, device)
                    acc = float((pred == df.y.values[te]).mean())
                    print(f"{args.exp} {variant} mode={mode:6s} {model_name:4s} seed={seed} "
                          f"fold={fold} acc={acc:.3f} ({time.time() - t0:.0f}s)")
                    for i, p in zip(te, pred):
                        rows.append(dict(exp=args.exp, variant=variant, model=model_name, mode=mode,
                                         seed=seed, fold=fold, test_idx=int(i), y_true=int(df.y[i]),
                                         y_pred=int(p), category=df.category[i]))

    RESULTS.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""
    path = RESULTS / f"preds_{args.exp}_nn_{variant}{tag}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
