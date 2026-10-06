import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import LABELS, RESULTS, load_data, make_folds, tokenize

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.M)
HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s", re.M)

GROUPS = {
    "length": ["n_chars", "n_words", "n_lines", "blank_lines"],
    "formatting": ["bold_count", "header_count", "bullet_lines", "code_blocks", "latex"],
    "lexical": ["avg_word_len", "ttr", "avg_sent_len"],
    "punctuation": ["em_dash", "exclaim", "question", "commas_per_word", "colon_count",
                    "digit_frac", "emoji"],
}


def features(text):
    words = re.findall(r"[A-Za-z']+", text)
    nw = max(len(words), 1)
    lines = text.split("\n")
    nonempty = [l for l in lines if l.strip()]
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_lines": len(nonempty),
        "blank_lines": len(lines) - len(nonempty),
        "bold_count": text.count("**") // 2,
        "header_count": len(HEADER_RE.findall(text)),
        "bullet_lines": len(BULLET_RE.findall(text)),
        "code_blocks": text.count("```") // 2,
        "latex": text.count("\\(") + text.count("\\[") + text.count("$"),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "ttr": len({w.lower() for w in words}) / nw,   # lexical diversity (type-token ratio)
        "avg_sent_len": nw / max(len(sents), 1),
        "em_dash": text.count("\u2014"),
        "exclaim": text.count("!"),
        "question": text.count("?"),
        "commas_per_word": text.count(",") / nw,
        "colon_count": text.count(":"),
        "digit_frac": sum(c.isdigit() for c in text) / max(len(text), 1),
        "emoji": len(EMOJI_RE.findall(text)),
    }


def cv_acc(make_model, X, y, folds):
    accs = []
    for tr, te in folds:
        m = make_model().fit(X[tr], y[tr])
        accs.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def tfidf_lr():
    return make_pipeline(
        TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                        ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10, max_iter=3000))


def main():
    df = load_data()
    y = df.y.values
    folds = make_folds(df, 5, seed=0)
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    # ---- A. feature means -------------------------------------------------
    F = pd.DataFrame([features(t) for t in df.output])
    F["llm"] = df.llm_name.values
    means = F.groupby("llm").mean().T.round(3)[LABELS]
    print("=== A. Mean stylometric features per LLM ===")
    print(means.to_string())
    means.to_csv(RESULTS / "rq4_feature_means.csv")
    wc = df.assign(n_words=F.n_words.values).pivot_table(
        index="llm_name", columns="category", values="n_words", aggfunc="median").round(0)
    print("\nMedian response length (words) by LLM x task:\n", wc.to_string())
    wc.to_csv(RESULTS / "rq4_length_by_task.csv")

    # ---- B. feature-group ablation ---------------------------------------
    X = F.drop(columns="llm")
    rf = lambda: RandomForestClassifier(n_estimators=300, random_state=0)
    rows = []
    allc = [c for g in GROUPS.values() for c in g]
    m, s = cv_acc(rf, X[allc].values, y, folds)
    rows.append(("all features", m, s))
    for g, cols in GROUPS.items():
        rest = [c for c in allc if c not in cols]
        rows.append((f"drop {g}", *cv_acc(rf, X[rest].values, y, folds)))
    for g, cols in GROUPS.items():
        rows.append((f"only {g}", *cv_acc(rf, X[cols].values, y, folds)))
    B = pd.DataFrame(rows, columns=["features", "acc_mean", "acc_std"]).round(3)
    print("\n=== B. Random forest on handcrafted features (5-fold CV, chance = 0.333) ===")
    print(B.to_string(index=False))
    B.to_csv(RESULTS / "rq4_feature_group_ablation.csv", index=False)

    # ---- C. text ablations -----------------------------------------------
    variants = {
        "full text": lambda t: t,
        "no markdown/newlines": lambda t: re.sub(r"[*#`|>_~\-]", " ", t).replace("\n", " "),
        "no punctuation, letters+digits only": lambda t: re.sub(r"[^A-Za-z0-9\s]", " ", t),
        "first 30 tokens only": lambda t: " ".join(tokenize(t)[:30]),
        "last 30 tokens only": lambda t: " ".join(tokenize(t)[-30:]),
    }
    rows = []
    for name, fn in variants.items():
        text = df.output.map(fn)
        accs = []
        for tr, te in folds:
            clf = tfidf_lr().fit(text.iloc[tr], y[tr])
            accs.append((clf.predict(text.iloc[te]) == y[te]).mean())
        rows.append((name, np.mean(accs), np.std(accs)))
    C = pd.DataFrame(rows, columns=["text variant", "acc_mean", "acc_std"]).round(3)
    print("\n=== C. TF-IDF + LR on ablated text (5-fold CV) ===")
    print(C.to_string(index=False))
    C.to_csv(RESULTS / "rq4_text_ablation.csv", index=False)

    # ---- D. top tokens ---------------------------------------------------
    pipe = tfidf_lr().fit(df.output, y)
    vec, lr = pipe[0], pipe[1]
    names = np.array(vec.get_feature_names_out())
    print("\n=== D. Most indicative tokens per LLM (largest LR weights) ===")
    top = {}
    for ci, label in enumerate(LABELS):
        idx = np.argsort(lr.coef_[ci])[::-1][:15]
        top[label] = [repr(t) for t in names[idx]]
        print(f"{label:8s}: {', '.join(top[label])}")
    pd.DataFrame(top).to_csv(RESULTS / "rq4_top_tokens.csv", index=False)


if __name__ == "__main__":
    main()
