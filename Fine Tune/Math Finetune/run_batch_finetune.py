# run_batch_finetune.py
# Batch launcher for multiple models contained in subfolders.
# - Keeps your structure: imports lora_finetune and calls ft.run() per model
# - Scans BASE_MODELS_DIR for subfolders that look like HF model dirs
# - Writes each adapter to ADAPTERS_ROOT/lora_out_<model_subfolder_name>
# - Fixes scoreboard EM read (val_em comes from run_meta.json["val_em_numeric"])
# - Uses faster defaults safe for A6000: BATCH_SIZE=4, GRAD_ACCUM=4, EVAL_STEPS=50
# - NEW: Auto-resume/skip: if a model output already has run_meta.json or final_scores.txt, it is skipped

import os
import sys
import traceback
import gc
import json
import csv
from datetime import datetime

# Ensure we can import lora_finetune from the same directory as this script
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lora_finetune as ft

# === YOUR PATHS ===
# Parent folder that contains one subfolder per model (each subfolder = HF model dir)
BASE_MODELS_DIR = os.path.join("Pruned Model", "Qwen2.5-Math-7B-Instruct")
# Where to save LoRA adapters and run outputs per model
ADAPTERS_ROOT   = os.path.join("Adapters", "Qwen2.5-Math-7B-Instruct")

# Progress status file (UTF-8 with BOM for Windows/Notepad emoji support)
STATUS_FILE = os.path.join(ADAPTERS_ROOT, "finetune_status.txt")
SCOREBOARD_CSV = os.path.join(ADAPTERS_ROOT, "finetune_results.csv")

# Minimal, safe overrides for speed on RTX A6000 (48GB).
# (We only touch knobs that affect throughput/feedback cadence. Everything else remains as in lora_finetune.py.)
OVERRIDES = {
    "MAX_LENGTH": 1024,     # keep 1k tokens context for speed
    "BATCH_SIZE": 2,        # safe on A6000 with BF16
    "GRAD_ACCUM": 4,        # keep effective batch reasonable
    "EVAL_STEPS": 200,       # faster validation signal
    "LOGGING_STEPS": 10,
    # Toggle this to True only if you ever hit VRAM limits at longer seq-lens
    "GRADIENT_CKPT": False,
    # Optionally adjust LoRA size:
    # "LORA_R": 32, "LORA_ALPHA": 64, "LORA_DROPOUT": 0.10,
    # Eval generation caps for EM (leave as trainer defaults unless you know you want different):
    # "INTERIM_EM_SAMPLE": 60, "INTERIM_EM_EVERY": 2, "INTERIM_MAX_NEW": 384, "FINAL_MAX_NEW": 384,
}

# Skip models that already have a completed run output (auto-resume)
SKIP_DONE = True  # set to False to force re-run even if outputs exist

# ----------------------------- helpers ----------------------------- #

def looks_like_model_dir(path: str) -> bool:
    """
    Heuristics to detect a Hugging Face model folder.
    """
    try:
        names = set(os.listdir(path))
    except Exception:
        return False
    # Any of these usually indicates a model dir:
    indicators = {"config.json", "tokenizer.json", "tokenizer.model", "pytorch_model.bin", "model.safetensors"}
    return any(n in names for n in indicators)

def model_subdirs(root: str):
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and looks_like_model_dir(p):
            yield name, p

def print_status(message: str):
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    # Plain text status log; append a timestamped line
    with open(STATUS_FILE, "a", encoding="utf-8-sig") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)

def append_score(row: dict):
    # CSV columns
    cols = [
        "timestamp", "model_name", "model_dir", "output_dir",
        "val_em", "val_em_strict", "val_found_rate", "val_parse_fail_rate",
        "epochs", "batch_size", "grad_accum", "max_length",
    ]
    os.makedirs(os.path.dirname(SCOREBOARD_CSV), exist_ok=True)
    write_header = not os.path.exists(SCOREBOARD_CSV)
    with open(SCOREBOARD_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in cols})

def apply_overrides():
    """
    Apply throughput/feedback overrides to the ft (trainer) module variables,
    without changing your trainer’s structure.
    """
    if "MAX_LENGTH" in OVERRIDES and OVERRIDES["MAX_LENGTH"] is not None:
        ft.MAX_LENGTH = int(OVERRIDES["MAX_LENGTH"])
    if "BATCH_SIZE" in OVERRIDES and OVERRIDES["BATCH_SIZE"] is not None:
        ft.BATCH_SIZE = int(OVERRIDES["BATCH_SIZE"])
    if "GRAD_ACCUM" in OVERRIDES and OVERRIDES["GRAD_ACCUM"] is not None:
        ft.GRAD_ACCUM = int(OVERRIDES["GRAD_ACCUM"])
    if "EVAL_STEPS" in OVERRIDES and OVERRIDES["EVAL_STEPS"] is not None:
        ft.EVAL_STEPS = int(OVERRIDES["EVAL_STEPS"])
    if "LOGGING_STEPS" in OVERRIDES and OVERRIDES["LOGGING_STEPS"] is not None:
        ft.LOGGING_STEPS = int(OVERRIDES["LOGGING_STEPS"])
    if "GRADIENT_CKPT" in OVERRIDES and OVERRIDES["GRADIENT_CKPT"] is not None:
        ft.GRADIENT_CKPT = bool(OVERRIDES["GRADIENT_CKPT"])
    # Optional LoRA overrides
    if "LORA_R" in OVERRIDES and OVERRIDES["LORA_R"] is not None:
        ft.LORA_R = int(OVERRIDES["LORA_R"])
    if "LORA_ALPHA" in OVERRIDES and OVERRIDES["LORA_ALPHA"] is not None:
        ft.LORA_ALPHA = int(OVERRIDES["LORA_ALPHA"])
    if "LORA_DROPOUT" in OVERRIDES and OVERRIDES["LORA_DROPOUT"] is not None:
        ft.LORA_DROPOUT = float(OVERRIDES["LORA_DROPOUT"])
    # Optional EM eval overrides
    for k in ("INTERIM_EM_SAMPLE","INTERIM_EM_EVERY","INTERIM_MAX_NEW","FINAL_MAX_NEW"):
        if k in OVERRIDES and OVERRIDES[k] is not None:
            setattr(ft, k, int(OVERRIDES[k]))

def read_em_from_run_meta(out_dir: str):
    """
    Read EM from run_meta.json using your trainer's keys (val_em_numeric, etc.).
    """
    meta_path = os.path.join(out_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        return None, None, None, None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Primary metric for scoreboard
        em = meta.get("val_em_numeric")
        em_strict = meta.get("val_em_strict")
        found = meta.get("val_found_rate")
        parse_fail = meta.get("val_parse_fail_rate")
        return em, em_strict, found, parse_fail
    except Exception:
        return None, None, None, None

def is_already_done(out_dir: str) -> bool:
    """
    Decide whether to skip a model because it looks completed.
    We consider the run 'done' if either run_meta.json or final_scores.txt exists.
    """
    if not SKIP_DONE:
        return False
    if os.path.exists(os.path.join(out_dir, "run_meta.json")):
        return True
    if os.path.exists(os.path.join(out_dir, "final_scores.txt")):
        return True
    return False

# ----------------------------- main loop ----------------------------- #

def main():
    # Ensure roots exist
    if not os.path.isdir(BASE_MODELS_DIR):
        raise FileNotFoundError(f"BASE_MODELS_DIR not found: {BASE_MODELS_DIR}")
    os.makedirs(ADAPTERS_ROOT, exist_ok=True)

    # Apply throughput/feedback overrides
    apply_overrides()

    # Optional: quick CUDA info
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available(), "| CUDA:", torch.version.cuda)
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("CUDA check skipped:", e)

    # Enumerate models
    subdirs = list(model_subdirs(BASE_MODELS_DIR))
    if not subdirs:
        print_status(f"No model-like subfolders found under: {BASE_MODELS_DIR}")
        return

    print_status(f"Found {len(subdirs)} model(s) under {BASE_MODELS_DIR}")

    trained, failed, skipped = 0, 0, 0

    for idx, (name, model_dir) in enumerate(subdirs, start=1):
        # Set per-run ft variables
        ft.MODEL_DIR = model_dir
        ft.OUTPUT_DIR = os.path.join(ADAPTERS_ROOT, f"lora_out_{name}")
        os.makedirs(ft.OUTPUT_DIR, exist_ok=True)

        # Optional: You can override datasets per run by editing lora_finetune.py,
        # or keep the same TRAIN_FILE / VAL_FILE set in lora_finetune.py.

        # Auto-skip already completed runs
        if is_already_done(ft.OUTPUT_DIR):
            print_status(f"[{idx}/{len(subdirs)}] Skip (already done): {name} → {ft.OUTPUT_DIR}")
            skipped += 1
            continue

        print_status(f"[{idx}/{len(subdirs)}] Start: {name}")
        print(f"MODEL_DIR : {ft.MODEL_DIR}")
        print(f"OUTPUT_DIR: {ft.OUTPUT_DIR}")
        print(f"TRAIN_FILE: {ft.TRAIN_FILE}")
        print(f"VAL_FILE  : {ft.VAL_FILE}")

        try:
            # Train (auto-resume if checkpoint exists) — handled inside ft.run()
            ft.run()

            # Read EM metrics from this run
            val_em, val_em_strict, found_rate, parse_fail_rate = read_em_from_run_meta(ft.OUTPUT_DIR)

            # Append to scoreboard
            append_score({
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "model_name": name,
                "model_dir": ft.MODEL_DIR,
                "output_dir": ft.OUTPUT_DIR,
                "val_em": val_em,
                "val_em_strict": val_em_strict,
                "val_found_rate": found_rate,
                "val_parse_fail_rate": parse_fail_rate,
                "epochs": ft.EPOCHS,
                "batch_size": ft.BATCH_SIZE,
                "grad_accum": ft.GRAD_ACCUM,
                "max_length": ft.MAX_LENGTH,
            })

            print_status(f"[{idx}/{len(subdirs)}] Done: {name} ✓  (val_em={val_em})")
            trained += 1

        except Exception as e:
            failed += 1
            print_status(f"[{idx}/{len(subdirs)}] Failed: {name} ✗  ({e})")
            traceback.print_exc()

        # VRAM hygiene between runs
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # Summary
    print("\n=== Summary ===")
    print(f"Trained : {trained}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"Adapters saved under: {ADAPTERS_ROOT}")
    print(f"Status file: {STATUS_FILE}")
    print(f"Scoreboard: {SCOREBOARD_CSV}")

if __name__ == "__main__":
    main()
