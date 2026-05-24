import csv
import json
import re
from pathlib import Path


TOKEN_THRESHOLD = 1024
VARIANT_CONFIG = [
    ("pruned", "pruning model outputs", "pruned trap(%)"),
    ("finetuned", "finetuned model outputs", "finetuned trap(%)"),
    ("random33", "random33 model outputs", "random33 trap(%)"),
    ("random42", "random42 model outputs", "random42 trap(%)"),
    ("reversed", "reversed model outputs", "reversed trap(%)"),
]
TASK_TYPES_WITHOUT_FINETUNED = {"qna", "conversational"}


def load_trap_percentage(metrics_json_path):
    with metrics_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    per_prompt_metrics = data.get("per_prompt_metrics", [])
    total_prompts_processed = len(per_prompt_metrics)

    if total_prompts_processed <= 0:
        return 0.0

    trap_count = 0

    for metric in per_prompt_metrics:
        tokens_generated = metric.get("tokens_generated")
        if tokens_generated is None:
            continue
        if tokens_generated >= TOKEN_THRESHOLD:
            trap_count += 1

    return (trap_count / total_prompts_processed) * 100


def extract_variant_key(file_name, variant, task_type):
    pattern = rf"{task_type}_generation_metrics_{variant}_\(([^)]+)\)\.json$"
    match = re.search(pattern, file_name)
    if not match:
        return None
    return match.group(1)


def safe_sort_key(value):
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def format_percent(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def get_active_variants(task_type):
    task_type = task_type.lower()
    if task_type in TASK_TYPES_WITHOUT_FINETUNED:
        return [cfg for cfg in VARIANT_CONFIG if cfg[0] != "finetuned"]
    return VARIANT_CONFIG


def build_trap_rows(base_dir, model_folder, task_type):
    active_variants = get_active_variants(task_type)
    original_dir = base_dir / "original model outputs" / model_folder

    original_file = original_dir / f"{task_type}_generation_metrics_original.json"
    if not original_file.exists():
        raise FileNotFoundError(f"Original metrics JSON not found: {original_file}")

    original_trap_percent = load_trap_percentage(original_file)

    variant_trap_maps = {}
    all_keys = set()

    for variant, folder_name, _column_name in active_variants:
        variant_dir = base_dir / folder_name / model_folder
        if not variant_dir.exists():
            variant_trap_maps[variant] = {}
            continue
        variant_files = sorted(variant_dir.glob(f"{task_type}_generation_metrics_{variant}_(*).json"))

        trap_map = {}
        for file_path in variant_files:
            key = extract_variant_key(file_path.name, variant, task_type)
            if key is None:
                continue
            trap_map[key] = load_trap_percentage(file_path)

        variant_trap_maps[variant] = trap_map
        all_keys.update(trap_map.keys())

    keys = sorted(all_keys, key=safe_sort_key)

    original_row = {"model": "original"}
    for _variant, _folder_name, column_name in active_variants:
        original_row[column_name] = format_percent(original_trap_percent)

    rows = [original_row]

    for key in keys:
        row = {"model": key}
        for variant, _folder_name, column_name in active_variants:
            row[column_name] = format_percent(variant_trap_maps[variant].get(key, 0.0))
        rows.append(row)

    return rows


def generate_trap_summary(model_folder, output_file, task_type):
    base_dir = Path(__file__).resolve().parent
    output_path = (base_dir / output_file).resolve()
    active_variants = get_active_variants(task_type)

    rows = build_trap_rows(base_dir, model_folder, task_type)

    fieldnames = ["model"] + [column_name for _variant, _folder_name, column_name in active_variants]

    with output_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Trap summary saved to: {output_path}")


def main():
    '''
    Main function to generate the trap summary CSV.
    Adjust the `model_folder` and `task_type` variables as needed to specify the model and task for which you want to generate the summary. The output will be saved as a CSV file in the same directory as this script.
    Distractor(qna, conversational) task have no finetuned variant, so the function will automatically exclude it from the summary if 'task_type' is set to either 'qna' or 'conversational'.
    '''
    model_folder = "Qwen2.5-Math-1.5B-Instruct"
    task_type = "math"
    output_file = f"trap_summary_{task_type}.csv"

    generate_trap_summary(
        model_folder=model_folder,
        output_file=output_file,
        task_type=task_type,
    )


if __name__ == "__main__":
    main()
