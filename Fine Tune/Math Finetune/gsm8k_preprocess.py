# filter_gsm8k.py
# Build train/val from GSM8K 'train' split. Targets end with \boxed{number}.
# If TEST_CSV exists, questions in it are excluded for safety.
# No EOS here (training code appends it).

import os
import re
import json
import random
import pandas as pd
from datasets import load_dataset

# === CONFIG ===
TEST_CSV        = os.path.join(".", "test_set.csv")    # optional; excluded if exists
OUT_DIR         = os.path.join(".", "dataset_splits_boxed")
VAL_RATIO       = 0.10
SEED            = 42

# Possible question column names if TEST_CSV is present
QUESTION_COL_CANDIDATES = ["prompt", "question", "Question", "problem", "Problem", "input"]

# Normalize final answer to \boxed{...}
NUM_RE = r"[-+]?(?:\d+/\d+|\d+(?:\.\d+)?)"
PAT_HASH_TAIL     = re.compile(rf"(.*?)(####\s*)({NUM_RE})(\s*)$", flags=re.S)
PAT_BRACE_TAIL    = re.compile(rf"(.*)\{{\s*({NUM_RE})\s*\}}(\s*)$", flags=re.S)
PAT_LAST_NUM_TAIL = re.compile(rf"(.*)({NUM_RE})(\s*)$", flags=re.S)

def to_boxed_answer(a: str) -> str:
    if not isinstance(a, str):
        return str(a)
    s = a.rstrip()

    # Already boxed?
    if re.search(rf"\\boxed\{{\s*{NUM_RE}\s*\}}\s*$", s):
        return s + ("\n" if not s.endswith("\n") else "")

    m = PAT_BRACE_TAIL.match(s)
    if m:
        pre, num, _ = m.groups()
        return pre.rstrip() + "\n\n\\boxed{" + num + "}\n"

    m = PAT_HASH_TAIL.match(s)
    if m:
        pre, _hashes, num, _ = m.groups()
        return pre.rstrip() + "\n\n\\boxed{" + num + "}\n"

    m = PAT_LAST_NUM_TAIL.match(s)
    if m:
        pre, num, _ = m.groups()
        return pre.rstrip() + "\n\n\\boxed{" + num + "}\n"

    return s + ("\n" if not s.endswith("\n") else "")

def detect_question_column(df: pd.DataFrame) -> str:
    for c in QUESTION_COL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]

def load_test_questions_if_any(path: str) -> set:
    if not os.path.exists(path):
        print("No TEST_CSV found — skipping exclusion step.")
        return set()
    print(f"Loading test CSV from: {path}")
    df = pd.read_csv(path)
    qcol = detect_question_column(df)
    test_qs = set(df[qcol].astype(str).str.strip().tolist())
    print(f"Detected question column: '{qcol}' | Test rows: {len(df)} | Unique: {len(test_qs)}")
    return test_qs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Safety: exclude any questions present in test CSV (even though official test != train)
    test_qs = load_test_questions_if_any(TEST_CSV)

    print("Loading GSM8K (configuration='main') train split…")
    ds = load_dataset("gsm8k", "main")
    pool = ds["train"]
    print(f"Train pool size: {len(pool)}")

    # Convert & optionally exclude
    filtered = []
    for ex in pool:
        q = str(ex["question"]).strip()
        if q in test_qs:
            continue
        a = to_boxed_answer(ex["answer"])
        filtered.append({"question": q, "answer": a})
    print(f"After exclusion: {len(filtered)} (removed {len(pool) - len(filtered)})")

    # Deduplicate by question text
    uniq = {}
    for ex in filtered:
        if ex["question"] not in uniq:
            uniq[ex["question"]] = ex
    deduped = list(uniq.values())
    print(f"After dedupe: {len(deduped)} (removed {len(filtered) - len(deduped)} dups)")

    # Shuffle & split
    random.seed(SEED)
    random.shuffle(deduped)
    n_total = len(deduped)
    n_val = int(round(n_total * VAL_RATIO))
    n_train = n_total - n_val
    train = deduped[:n_train]
    val   = deduped[n_train:]
    print(f"Train: {len(train)} | Val: {len(val)} | Seed: {SEED} | Val ratio: {VAL_RATIO}")

    # Save JSONL
    out_train = os.path.join(OUT_DIR, "train.jsonl")
    out_val   = os.path.join(OUT_DIR, "val.jsonl")
    with open(out_train, "w", encoding="utf-8") as ft:
        for ex in train:
            ft.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out_val, "w", encoding="utf-8") as fv:
        for ex in val:
            fv.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Meta
    meta = {
        "pool_size": len(pool),
        "after_exclusion": len(filtered),
        "after_dedupe": len(deduped),
        "train": len(train),
        "val": len(val),
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "target_format": r"\boxed{number}",
        "test_csv_used": os.path.exists(TEST_CSV),
        "test_csv_path": TEST_CSV,
        "out_dir": OUT_DIR,
    }
    with open(os.path.join(OUT_DIR, "splits_meta.json"), "w", encoding="utf-8") as fm:
        json.dump(meta, fm, indent=2, ensure_ascii=False)

    print("Saved:")
    print(f"- {out_train}")
    print(f"- {out_val}")
    print(f"- {os.path.join(OUT_DIR, 'splits_meta.json')}")

if __name__ == "__main__":
    main()
