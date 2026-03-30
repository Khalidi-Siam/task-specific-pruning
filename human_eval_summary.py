import csv
import json
import re
from pathlib import Path

RESULTS_FILE = "samples.jsonl_results.jsonl"
HUMANEVAL_ROOT = Path(__file__).resolve().parent / "Human_eval_result"

# Each entry: (column name in CSV, folder name under Human_eval_result, subfolder pattern)
# Patterns match the actual folder names written by human_eval_script.py:
#   humaneval_output_pruned_({level})
#   humaneval_output_finetuned_({level})
#   humaneval_output_random33_({level})
#   etc.
VARIANT_CONFIG = [
    ("pruned pass@1(%)",    "Pruned Model",         "humaneval_output_pruned_({level})"),
    ("finetuned pass@1(%)", "Fine Tuned Model",     "humaneval_output_finetuned_({level})"),
    ("random33 pass@1(%)", "Random33 Pruned Model", "humaneval_output_random33_({level})"),
    ("random42 pass@1(%)", "Random42 Pruned Model", "humaneval_output_random42_({level})"),
    ("reversed pass@1(%)", "Reverse Pruned Model",  "humaneval_output_reversed_({level})"),
]


# ─────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────

def compute_pass_at_1(results_file: Path) -> float:
    """
    Read a samples.jsonl_results.jsonl file and compute Pass@1 (%).
    Each line must have a "passed" boolean field.
    """
    total = 0
    passed = 0
    with results_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            if record.get("passed", False):
                passed += 1
    if total == 0:
        return 0.0
    return (passed / total) * 100


def format_score(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def safe_sort_key(value: str):
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def extract_level_from_subfolder(subfolder_name: str, pattern: str, model: str) -> str | None:
    """
    Given a concrete subfolder name and the pattern template,
    back-calculate what {level} was.

    The new pattern format is a literal prefix with a {level} placeholder, e.g.:
        humaneval_output_finetuned_({level})
        humaneval_output_random33_({level})

    {model} is no longer embedded in the folder name, but the parameter is
    kept for backward compatibility (the replace is a no-op when absent).
    """
    # 1. Fill in {model} (no-op for the new patterns that don't contain it)
    model_filled = pattern.replace("{model}", model)
    # 2. Split on the {level} placeholder so we can re.escape each literal part
    parts = model_filled.split("{level}")
    if len(parts) != 2:
        return None
    prefix, suffix = parts
    regex = re.escape(prefix) + r"([^/\\]+)" + re.escape(suffix)
    match = re.fullmatch(regex, subfolder_name)
    if match:
        return match.group(1)
    return None


# ─────────────────────────────────────────────────────────────────
# Per-variant scan
# ─────────────────────────────────────────────────────────────────

def scan_variant(eval_folder: str, subfolder_pattern: str, model_name: str) -> dict[str, float]:
    """
    Walk Human_eval_result/<eval_folder>/<model_name>/
    looking for subfolders that match the pattern and contain
    samples.jsonl_results.jsonl.

    Returns {level_str: pass@1_%} for every found level.
    """
    model_dir = HUMANEVAL_ROOT / eval_folder / model_name
    if not model_dir.exists():
        print(f"[skip] Folder not found, skipping variant '{eval_folder}': {model_dir}")
        return {}

    scores: dict[str, float] = {}
    for sub in sorted(model_dir.iterdir()):
        if not sub.is_dir():
            continue
        results_path = sub / RESULTS_FILE
        if not results_path.exists():
            continue
        level = extract_level_from_subfolder(sub.name, subfolder_pattern, model_name)
        if level is None:
            print(f"[warn] Could not extract level from folder name: {sub.name!r}")
            continue
        scores[level] = compute_pass_at_1(results_path)

    if not scores:
        print(f"[skip] No result files found for variant '{eval_folder}' in {model_dir}")
    return scores


# ─────────────────────────────────────────────────────────────────
# Original model
# ─────────────────────────────────────────────────────────────────

def get_original_score(model_name: str) -> float:
    # original_model_human_eval_script.py saves directly into:
    #   Human_eval_result/Original Model/{model_name}/samples.jsonl  (+ _results.jsonl after eval)
    original_dir = HUMANEVAL_ROOT / "Original Model" / model_name
    results_file = original_dir / RESULTS_FILE
    if not results_file.exists():
        raise FileNotFoundError(
            f"Original model result not found: {results_file}\n"
            "Run original_model_human_eval_script.py for the original model first."
        )
    return compute_pass_at_1(results_file)


# ─────────────────────────────────────────────────────────────────
# Summary builder
# ─────────────────────────────────────────────────────────────────

def build_summary_rows(model_name: str):
    """
    Collect scores for all variants and return (rows, active_variants).
    active_variants contains only configs where at least one result was found.
    """
    original_score = get_original_score(model_name)

    variant_scores: dict[str, dict[str, float]] = {}
    active_variants = []   # (column_name, eval_folder, subfolder_pattern)
    all_levels: set[str] = set()

    for column_name, eval_folder, subfolder_pattern in VARIANT_CONFIG:
        scores = scan_variant(eval_folder, subfolder_pattern, model_name)
        if not scores:
            continue
        variant_scores[column_name] = scores
        active_variants.append((column_name, eval_folder, subfolder_pattern))
        all_levels.update(scores.keys())

    levels = sorted(all_levels, key=safe_sort_key)

    # Original row — repeat its score for every active variant column
    original_row = {"model": "original"}
    for column_name, _, _ in active_variants:
        original_row[column_name] = format_score(original_score)

    rows = [original_row]
    for level in levels:
        row = {"model": level}
        for column_name, _, _ in active_variants:
            score = variant_scores[column_name].get(level)
            row[column_name] = format_score(score) if score is not None else "not found"
        rows.append(row)

    return rows, active_variants


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────

def generate_humaneval_summary(model_name: str, output_file: str | None = None) -> Path | None:
    """
    Generate a CSV summarising HumanEval Pass@1 (%) for all pruning
    variants of the given model.

    Parameters
    ----------
    model_name  : str
        The model folder name, e.g. "Qwen2.5-Coder-7B-Instruct".
        Must match the folder names inside Human_eval_result/.
    output_file : str | None
        Path (relative or absolute) for the output CSV.
        Defaults to "humaneval_summary_<model_name>.csv" in the project root.
    """
    if output_file is None:
        output_file = f"humaneval_summary_{model_name}.csv"

    output_path = (Path(__file__).resolve().parent / output_file).resolve()

    rows, active_variants = build_summary_rows(model_name)

    if not active_variants:
        print("[warn] No variant data found. CSV will not be written.")
        return None

    fieldnames = ["model"] + [col for col, _, _ in active_variants]

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"HumanEval summary saved to: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────
# Main — update model_name to the model you want to summarise
# ─────────────────────────────────────────────────────────────────

def main():
    """
    Generate a HumanEval Pass@1 summary CSV.

    Adjust `model_name` to match the folder name under Human_eval_result/
    (e.g. "Qwen2.5-Coder-7B-Instruct" or "Qwen2.5-Coder-1.5B-Instruct").

    The script automatically skips any variant whose folder is missing,
    so you can run it at any stage of the pipeline.
    """
    model_name = "Qwen2.5-Coder-1.5B-Instruct"

    generate_humaneval_summary(model_name=model_name)


if __name__ == "__main__":
    main()
