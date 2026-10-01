"""Step 3: parse the raw chatbot replies into data/dataset.csv.

Run:  python parse_responses.py
Reads data/raw/<model>/batch_XX.txt, writes data/dataset.csv with columns
  llm_name, prompt_id, category, input, output
and prints a summary plus any problems (missing/empty answers).
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA = Path("data")
MODELS = ["claude", "chatgpt", "gemini"]
MIN_CHARS = 20
# Matches "=== ANSWER 3 ===" even when wrapped in markdown like **...** or ## ...
MARK = re.compile(r"^[^\w\n]*=+\s*ANSWER\s*(\d+)\s*=+[^\w\n]*$", re.I | re.M)


def split_answers(text):
    ms = list(MARK.finditer(text))
    out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        body = text[m.end():end].strip()
        body = re.sub(r"\n[-*_ ]{3,}\s*$", "", body).strip()  # trailing --- separators
        out[int(m.group(1))] = body
    return out


def main():
    prompts = pd.read_csv(DATA / "prompts.csv").set_index("prompt_id")
    batches = json.loads((DATA / "batches.json").read_text())
    rows, problems = [], []

    for model in MODELS:
        for b, pids in batches.items():
            f = DATA / "raw" / model / f"batch_{b}.txt"
            if not f.exists():
                problems.append(f"missing file: {f}")
                continue
            ans = split_answers(f.read_text(encoding="utf-8"))
            for k, pid in enumerate(pids, 1):
                out = ans.get(k, "")
                if len(out) < MIN_CHARS:
                    problems.append(f"{model} batch {b} answer {k}: missing or empty")
                    continue
                rows.append({
                    "llm_name": model,
                    "prompt_id": pid,
                    "category": prompts.loc[pid, "category"],
                    "input": prompts.loc[pid, "prompt"],
                    "output": out,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No responses found yet. Save replies to data/raw/<model>/batch_XX.txt first.")
        return
    df.to_csv(DATA / "dataset.csv", index=False)

    df["n_words"] = df["output"].str.split().str.len()
    print(f"Saved {len(df)} rows to data/dataset.csv\n")
    print("Rows per model:\n", df["llm_name"].value_counts().to_string(), "\n")
    print("Rows per model x category:\n", pd.crosstab(df["llm_name"], df["category"]).to_string(), "\n")
    print("Mean output length (words) per model:\n",
          df.groupby("llm_name")["n_words"].mean().round(1).to_string(), "\n")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
    else:
        print("No problems found.")


if __name__ == "__main__":
    main()
