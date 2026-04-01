# run_batch_finetune_code.py
# Batch launcher for multiple pruned Qwen2.5-Coder-7B-Instruct model folders.
# Saves each model's adapter directly into:

import os
import sys
import traceback
import gc
import json
import csv
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lora_finetune_code as ft

# ===================== YOUR PATHS =====================
BASE_MODELS_DIR = os.path.join("Pruned Model", "Qwen2.5-Coder-7B-Instruct")

# Save adapters here, with subfolder names:
ADAPTERS_ROOT = os.path.join("adapters")

TRAIN_FILE = os.path.join("Code FineTune", "dataset_splits_code", "train.jsonl")
VAL_FILE   = os.path.join("Code FineTune", "dataset_splits_code", "val.jsonl")
# =====================================================

STATUS_FILE    = os.path.join(ADAPTERS_ROOT, "finetune_status.txt")
SCOREBOARD_CSV = os.path.join(ADAPTERS_ROOT, "finetune_results.csv")

# Speed-first safe defaults for 7B on RTX A6000 (48GB):
# - seq len 1024
# - 2 epochs
# - higher micro-batch to use more GPU
# - checkpointing OFF (faster; you have headroom)
# - eval/save not too frequent
OVERRIDES = {
    "MAX_LENGTH": 1024,
    "EPOCHS": 2,

    "BATCH_SIZE": 2,
    "GRAD_ACCUM": 8,

    "GRADIENT_CKPT": False,

    "EVAL_STEPS": 400,
    "SAVE_STEPS": 400,
    "SAVE_TOTAL_LIMIT": 2,
    "LOGGING_STEPS": 20,

    "LR": 2e-4,
    "WARMUP_RATIO": 0.03,
    "WEIGHT_DECAY": 0.01,

    # Keep QuickEM OFF for speed (generate() is expensive)
    "ENABLE_QUICK_EM": False,
    # If you ever enable it:
    # "INTERIM_EM_SAMPLE": 8,
    # "INTERIM_MAX_NEW": 128,
}

SKIP_DONE = True  # skip if run_meta.json exists in output dir

def looks_like_model_dir(path: str) -> bool:
    try:
        names = set(os.listdir(path))
    except Exception:
        return False
    indicators = {
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "model.safetensors",
        "pytorch_model.bin",
        "generation_config.json",
    }
    return any(n in names for n in indicators)

def model_subdirs(root: str):
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and looks_like_model_dir(p):
            yield name, p

def print_status(message: str):
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "a", encoding="utf-8-sig") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)

def append_score(row: dict):
    cols = [
        "timestamp", "model_name", "model_dir", "output_dir",
        "best_eval_loss", "last_eval_loss", "last_val_em_exact",
        "epochs", "batch_size", "grad_accum", "max_length", "lr",
    ]
    os.makedirs(os.path.dirname(SCOREBOARD_CSV), exist_ok=True)
    write_header = not os.path.exists(SCOREBOARD_CSV)
    with open(SCOREBOARD_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in cols})

def apply_overrides():
    for k, v in OVERRIDES.items():
        if hasattr(ft, k) and v is not None:
            setattr(ft, k, v)

    # Force BF16 LoRA (your requirement)
    if hasattr(ft, "USE_BF16"):
        ft.USE_BF16 = True
    if hasattr(ft, "USE_FP16"):
        ft.USE_FP16 = False

def is_already_done(out_dir: str) -> bool:
    if not SKIP_DONE:
        return False
    return os.path.exists(os.path.join(out_dir, "run_meta.json"))

def read_run_meta(out_dir: str):
    p = os.path.join(out_dir, "run_meta.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def main():
    if not os.path.isdir(BASE_MODELS_DIR):
        raise FileNotFoundError(f"BASE_MODELS_DIR not found: {BASE_MODELS_DIR}")
    os.makedirs(ADAPTERS_ROOT, exist_ok=True)

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"TRAIN_FILE not found: {TRAIN_FILE}")
    if not os.path.exists(VAL_FILE):
        raise FileNotFoundError(f"VAL_FILE not found: {VAL_FILE}")

    apply_overrides()

    # set dataset paths in trainer module
    ft.TRAIN_FILE = TRAIN_FILE
    ft.VAL_FILE = VAL_FILE

    # Optional: CUDA info
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available(), "| CUDA:", torch.version.cuda)
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            free, total = torch.cuda.mem_get_info()
            print(f"VRAM (free/total): {free/1e9:.1f} GB / {total/1e9:.1f} GB")
    except Exception as e:
        print("CUDA check skipped:", e)

    subdirs = list(model_subdirs(BASE_MODELS_DIR))
    if not subdirs:
        print_status(f"No model-like subfolders found under: {BASE_MODELS_DIR}")
        return

    print_status(f"Found {len(subdirs)} model(s) under {BASE_MODELS_DIR}")

    trained, failed, skipped = 0, 0, 0

    for idx, (name, model_dir) in enumerate(subdirs, start=1):
        out_dir = os.path.join(ADAPTERS_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)

        if is_already_done(out_dir):
            print_status(f"[{idx}/{len(subdirs)}] Skip (already done): {name} → {out_dir}")
            skipped += 1
            continue

        # per-run vars
        ft.MODEL_DIR = model_dir
        ft.OUTPUT_DIR = out_dir

        print_status(f"[{idx}/{len(subdirs)}] Start: {name}")
        print(f"MODEL_DIR : {ft.MODEL_DIR}")
        print(f"OUTPUT_DIR: {ft.OUTPUT_DIR}")
        print(f"TRAIN_FILE: {ft.TRAIN_FILE}")
        print(f"VAL_FILE  : {ft.VAL_FILE}")

        try:
            ft.run()

            meta = read_run_meta(out_dir)
            append_score({
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "model_name": name,
                "model_dir": model_dir,
                "output_dir": out_dir,
                "best_eval_loss": meta.get("best_eval_loss"),
                "last_eval_loss": meta.get("last_eval_loss"),
                "last_val_em_exact": meta.get("last_val_em_exact"),
                "epochs": getattr(ft, "EPOCHS", None),
                "batch_size": getattr(ft, "BATCH_SIZE", None),
                "grad_accum": getattr(ft, "GRAD_ACCUM", None),
                "max_length": getattr(ft, "MAX_LENGTH", None),
                "lr": getattr(ft, "LR", None),
            })

            print_status(f"[{idx}/{len(subdirs)}] Done: {name} ✓")
            trained += 1

        except Exception as e:
            failed += 1
            print_status(f"[{idx}/{len(subdirs)}] Failed: {name} ✗  ({e})")
            traceback.print_exc()

        # VRAM hygiene
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    print("\n=== Summary ===")
    print(f"Trained : {trained}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"Adapters saved under: {ADAPTERS_ROOT}")
    print(f"Status file: {STATUS_FILE}")
    print(f"Scoreboard: {SCOREBOARD_CSV}")

if __name__ == "__main__":
    main()
