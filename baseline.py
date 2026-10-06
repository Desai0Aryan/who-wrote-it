import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import (RESULTS, load_data, make_domain_folds, make_folds,
                    tokenize)


def make_text(df, mode):
    if mode == "input":
        return df.input
    if mode == "output":
        return df.output
    return df.input + " \n <sep> \n " + df.output


def make_clf():
    return make_pipeline(
        TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                        ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10, max_iter=3000),
    )


def main():
    df = load_data()
    RESULTS.mkdir(exist_ok=True)
    plan = {"rq1": ["output"], "rq2": ["input", "output", "both"], "rq3": ["output"]}
    for exp, modes in plan.items():
        rows = []
        if exp == "rq3":
            splits = make_domain_folds(df)
        else:
            splits = [(tr, te, f) for f, (tr, te) in enumerate(make_folds(df, 5, seed=0))]
        for mode in modes:
            text = make_text(df, mode)
            for tr, te, fold in splits:
                clf = make_clf().fit(text.iloc[tr], df.y.values[tr])
                pred = clf.predict(text.iloc[te])
                acc = (pred == df.y.values[te]).mean()
                print(f"{exp} mode={mode:6s} tfidf_lr fold={fold} acc={acc:.3f}")
                for i, p in zip(te, pred):
                    rows.append(dict(exp=exp, variant="full", model="tfidf_lr", mode=mode, seed=0,
                                     fold=fold, test_idx=int(i), y_true=int(df.y[i]),
                                     y_pred=int(p), category=df.category[i]))
        pd.DataFrame(rows).to_csv(RESULTS / f"preds_{exp}_baseline_full.csv", index=False)
    print("saved results/preds_*_baseline_full.csv")


if __name__ == "__main__":
    main()
