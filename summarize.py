import glob

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from common import LABELS, RESULTS


def load_all():
    files = [f for f in sorted(glob.glob(str(RESULTS / "preds_*.csv"))) if "_quick" not in f]
    if not files:
        raise SystemExit("No results/preds_*.csv files found. Run baseline.py / train_nn.py first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def per_fold_scores(df):
    rows = []
    keys = ["exp", "variant", "model", "mode", "seed", "fold"]
    for k, g in df.groupby(keys):
        rows.append(dict(zip(keys, k),
                         acc=(g.y_true == g.y_pred).mean(),
                         macro_f1=f1_score(g.y_true, g.y_pred, average="macro")))
    m = pd.DataFrame(rows)
    # average over random restarts first, so each fold counts once
    return m.groupby(["exp", "variant", "model", "mode", "fold"], as_index=False)[["acc", "macro_f1"]].mean()


def save_confusions(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping confusion matrix images)")
        return
    sub = df[(df.exp == "rq1") & (df.variant == "full") & (df["mode"] == "output")]
    for model, g in sub.groupby("model"):
        cm = confusion_matrix(g.y_true, g.y_pred, labels=range(len(LABELS)))
        cm = cm / cm.sum(axis=1, keepdims=True)           # row-normalised = per-class recall
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{model} (RQ1, output only)")
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black")
        fig.tight_layout()
        fig.savefig(RESULTS / f"confusion_rq1_{model}.png", dpi=200)
        plt.close(fig)
        print(f"saved results/confusion_rq1_{model}.png")


def main():
    df = load_all()
    folds = per_fold_scores(df)

    def agg(x):
        return pd.Series({"acc_mean": x.acc.mean(), "acc_std": x.acc.std(),
                          "f1_mean": x.macro_f1.mean(), "f1_std": x.macro_f1.std(), "n_folds": len(x)})

    pd.set_option("display.width", 200)
    for exp in ["rq1", "rq2"]:
        t = folds[folds.exp == exp].groupby(["variant", "model", "mode"]).apply(agg).round(3)
        if len(t):
            print(f"\n=== {exp.upper()} (mean/std over CV folds) ===\n{t.to_string()}")
            t.to_csv(RESULTS / f"summary_{exp}.csv")

    r3 = folds[folds.exp == "rq3"]
    if len(r3):
        t = r3.pivot_table(index=["variant", "model"], columns="fold", values="acc").round(3)
        t["mean"] = t.mean(axis=1).round(3)
        print(f"\n=== RQ3 accuracy when the listed task is held out of training ===\n{t.to_string()}")
        t.to_csv(RESULTS / "summary_rq3.csv")

    # RQ1 per-task accuracy: is some kind of prompt harder?
    r1 = df[(df.exp == "rq1") & (df["mode"] == "output")].copy()
    if len(r1):
        r1["correct"] = (r1.y_true == r1.y_pred).astype(float)
        t = r1.pivot_table(index=["variant", "model"], columns="category", values="correct").round(3)
        print(f"\n=== RQ1 accuracy by task type ===\n{t.to_string()}")
        t.to_csv(RESULTS / "summary_rq1_by_category.csv")

        r1["true_llm"] = r1.y_true.map(dict(enumerate(LABELS)))
        t = r1.pivot_table(index=["variant", "model"], columns="true_llm", values="correct").round(3)
        print(f"\n=== RQ1 recall by LLM ===\n{t.to_string()}")
        t.to_csv(RESULTS / "summary_rq1_by_llm.csv")

    save_confusions(df)


if __name__ == "__main__":
    main()
