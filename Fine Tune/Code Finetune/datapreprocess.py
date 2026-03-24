# preprocess_code_csv_to_jsonl.py
# Build fixed-size train/val splits with a 60/40 mix from evol/others.
# Dedupe within each dataset and across combined splits.
# Output JSONL: {"instruction": "...", "response": "..."}

import os, json, random
import pandas as pd

# ========= CONFIG =========
CSV_EVOL   = os.path.join("Code FineTune", "datasets", "dataset_evol_only.csv")
CSV_OTHERS = os.path.join("Code FineTune", "datasets", "dataset_others_only.csv")

OUT_DIR = os.path.join("Code FineTune", "dataset_splits_code")

TRAIN_N = 7000
VAL_N   = 700

EVOL_RATIO = 0.60  # 60% evol, 40% others

SEED = 42

MIN_QUERY_CHARS  = 1
MIN_ANSWER_CHARS = 1
# =========================

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_clean(csv_path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{name} CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Query" not in df.columns or "Answer" not in df.columns:
        raise ValueError(f"{name} must contain columns Query/Answer. Found: {list(df.columns)}")

    df = df[["Query", "Answer"]].copy()
    df["Query"] = df["Query"].astype(str).str.strip()
    df["Answer"] = df["Answer"].astype(str).str.strip()

    # Drop empty/too-short
    df = df[(df["Query"].str.len() >= MIN_QUERY_CHARS) & (df["Answer"].str.len() >= MIN_ANSWER_CHARS)].copy()

    # Dedupe within dataset
    before = len(df)
    df = df.drop_duplicates(subset=["Query", "Answer"]).reset_index(drop=True)
    after = len(df)
    print(f"[prep:{name}] rows={before:,} -> dedup={after:,} (removed {before-after:,})")
    return df

def take_n(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) < n:
        raise ValueError(f"Not enough rows to sample: need {n:,}, have {len(df):,}")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def remove_used(pool: pd.DataFrame, used: pd.DataFrame) -> pd.DataFrame:
    # Remove exact Query+Answer pairs from pool
    merged = pool.merge(used, on=["Query", "Answer"], how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)

def dedupe_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["Query", "Answer"]).reset_index(drop=True)

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    evol = load_clean(CSV_EVOL, "evol")
    others = load_clean(CSV_OTHERS, "others")

    # Targets (exact ints)
    train_evol_n   = int(round(TRAIN_N * EVOL_RATIO))
    train_others_n = TRAIN_N - train_evol_n

    val_evol_n     = int(round(VAL_N * EVOL_RATIO))
    val_others_n   = VAL_N - val_evol_n

    print(f"[prep] target train: evol={train_evol_n:,} others={train_others_n:,} total={TRAIN_N:,}")
    print(f"[prep] target val  : evol={val_evol_n:,} others={val_others_n:,} total={VAL_N:,}")

    # ---- Build VAL first (so it is held out) ----
    val_evol   = take_n(evol,   val_evol_n,   SEED + 10)
    val_others = take_n(others, val_others_n, SEED + 11)

    evol_rem   = remove_used(evol,   val_evol)
    others_rem = remove_used(others, val_others)

    # ---- Build TRAIN from remaining ----
    train_evol   = take_n(evol_rem,   train_evol_n,   SEED + 20)
    train_others = take_n(others_rem, train_others_n, SEED + 21)

    train_df = pd.concat([train_evol, train_others], ignore_index=True)
    val_df   = pd.concat([val_evol,   val_others], ignore_index=True)

    # ---- Cross-dedupe within each split ----
    train_before = len(train_df)
    val_before   = len(val_df)

    train_df = dedupe_df(train_df)
    val_df   = dedupe_df(val_df)

    # ---- Ensure NO overlap between train and val ----
    # Remove any pairs from train that appear in val (should be rare, but safe)
    train_df = remove_used(train_df, val_df)

    print(f"[prep] train before={train_before:,} after_dedupe/anti_overlap={len(train_df):,}")
    print(f"[prep] val   before={val_before:,} after_dedupe={len(val_df):,}")

    # ---- Backfill if dedupe reduced sizes ----
    # We'll attempt to top up train/val to the exact requested sizes while keeping ratio.
    def backfill(split_name: str, split_df: pd.DataFrame, want_total: int,
                 want_evol: int, want_others: int,
                 evol_pool: pd.DataFrame, others_pool: pd.DataFrame,
                 seed_base: int):
        # Current counts by origin (we’ll tag origin during backfill selection)
        # For simplicity, we compute by membership against original sets (slow) is avoided.
        # We'll backfill by taking additional from pools and then dedupe again.

        need = want_total - len(split_df)
        if need <= 0:
            return split_df, evol_pool, others_pool

        # Determine deficits relative to target ratio using current size (approx).
        # Prefer filling evol first if evol target likely short, else others.
        # We'll compute based on how many from each we *intended* minus how many are currently present.
        # Since we didn't store origin column, we estimate by rebuilding with origin quickly:
        # (Cheap enough for 7.7k)
        split_e = split_df.merge(evol[["Query","Answer"]], on=["Query","Answer"], how="inner")
        split_o = split_df.merge(others[["Query","Answer"]], on=["Query","Answer"], how="inner")

        cur_e = len(split_e)
        cur_o = len(split_o)

        def_e = max(0, want_evol - cur_e)
        def_o = max(0, want_others - cur_o)

        # If both deficits are 0 but size is short (due to weird overlaps), fill by ratio.
        if def_e + def_o == 0:
            take_e = int(round(need * EVOL_RATIO))
            take_o = need - take_e
        else:
            # Fill primarily where deficit exists
            take_e = min(def_e, need)
            take_o = min(def_o, need - take_e)
            # If still remaining, fill by ratio from whatever is available
            rem = need - (take_e + take_o)
            if rem > 0:
                add_e = int(round(rem * EVOL_RATIO))
                add_o = rem - add_e
                take_e += add_e
                take_o += add_o

        if take_e > 0:
            extra_e = take_n(evol_pool, take_e, seed_base + 1)
            evol_pool = remove_used(evol_pool, extra_e)
            split_df = pd.concat([split_df, extra_e], ignore_index=True)

        if take_o > 0:
            extra_o = take_n(others_pool, take_o, seed_base + 2)
            others_pool = remove_used(others_pool, extra_o)
            split_df = pd.concat([split_df, extra_o], ignore_index=True)

        # Dedupe again
        split_df = dedupe_df(split_df)
        return split_df, evol_pool, others_pool

    # Pools for backfill should exclude anything already used in either split
    used_all = pd.concat([train_df, val_df], ignore_index=True)
    evol_pool   = remove_used(evol, used_all)
    others_pool = remove_used(others, used_all)

    # Backfill val first, then train (so val remains clean holdout)
    val_df, evol_pool, others_pool = backfill(
        "val", val_df, VAL_N, val_evol_n, val_others_n, evol_pool, others_pool, SEED + 100
    )
    # Ensure no overlap again
    train_df = remove_used(train_df, val_df)

    train_df, evol_pool, others_pool = backfill(
        "train", train_df, TRAIN_N, train_evol_n, train_others_n, evol_pool, others_pool, SEED + 200
    )
    train_df = remove_used(train_df, val_df)

    # Final checks
    if len(train_df) != TRAIN_N or len(val_df) != VAL_N:
        print(f"[prep][WARN] final sizes train={len(train_df):,}/{TRAIN_N:,} val={len(val_df):,}/{VAL_N:,}")
        print("             If this happens, your source data may be too small after dedupe.")
    else:
        print(f"[prep] final sizes OK train={len(train_df):,} val={len(val_df):,}")

    # Write JSONL
    train_rows = [{"instruction": q, "response": a} for q, a in zip(train_df["Query"], train_df["Answer"])]
    val_rows   = [{"instruction": q, "response": a} for q, a in zip(val_df["Query"], val_df["Answer"])]

    train_path = os.path.join(OUT_DIR, "train.jsonl")
    val_path   = os.path.join(OUT_DIR, "val.jsonl")
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    print("[prep] wrote:", train_path)
    print("[prep] wrote:", val_path)
    if train_rows:
        print("[prep] sample train row:", json.dumps(train_rows[0], ensure_ascii=False)[:600])

if __name__ == "__main__":
    main()
