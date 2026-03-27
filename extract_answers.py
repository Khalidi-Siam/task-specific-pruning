import csv
import json
import re
from pathlib import Path


VARIANT_CONFIG = [
    ("pruned", "pruning model outputs", "pruned EM(%)"),
    ("finetuned", "finetuned model outputs", "finetuned EM(%)"),
    ("random33", "random33 model outputs", "random33 EM(%)"),
    ("random42", "random42 model outputs", "random42 EM(%)"),
    ("reversed", "reversed model outputs", "reversed EM(%)"),
]


def extract_answer(response):
    # First try to find complete \boxed{...} pattern.
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1)

    # If complete pattern not found, try to find incomplete \boxed{ pattern.
    # This handles cases like $\\boxed{1500 where closing } is missing.
    match = re.search(r"\\boxed\{([^}$\s]+)", response)
    if match:
        return match.group(1)

    return "not found in this format"


def parse_prompt_no(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, int):
        return raw_value

    value = str(raw_value).strip()
    if not value:
        return None
    if value.upper() == "STARTING":
        return None

    try:
        return int(value)
    except ValueError:
        return None


def load_ground_truth(ground_truth_csv_path):
    ground_truth = {}

    with ground_truth_csv_path.open("r", encoding="utf-8", newline="") as gt_file:
        reader = csv.DictReader(gt_file)
        for row in reader:
            prompt_no = parse_prompt_no(row.get("prompt_No") or row.get("prompt_no"))
            if prompt_no is None:
                continue
            ground_truth[prompt_no] = row.get("answer", "")

    if not ground_truth:
        raise ValueError("No ground-truth rows were loaded from the CSV file.")

    return ground_truth


def load_predictions(jsonl_path):
    predictions = {}

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            data = json.loads(stripped)
            prompt_no = parse_prompt_no(data.get("prompt_no") or data.get("prompt_No"))
            if prompt_no is None:
                continue

            response = data.get("response", "")
            predictions[prompt_no] = extract_answer(response)

    return predictions


def compute_em_percentage(predictions, ground_truth):
    total = len(ground_truth)
    correct = 0

    for prompt_no, gt_answer in ground_truth.items():
        pred_answer = predictions.get(prompt_no)
        if pred_answer is not None:
            # Try numeric comparison first
            try:
                gt_num = float(gt_answer)
                pred_num = float(pred_answer)
                if gt_num == pred_num:
                    correct += 1
            except (ValueError, TypeError):
                # Fall back to string comparison
                if pred_answer == gt_answer:
                    correct += 1

    return (correct / total) * 100


def extract_variant_key(file_name, variant):
    pattern = rf"math_generated_outputs_{variant}_\(([^)]+)\)\.jsonl$"
    match = re.search(pattern, file_name)
    if not match:
        return None
    return match.group(1)


def safe_sort_key(value):
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def format_em(value):
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return formatted


def build_em_rows(base_dir, model_folder, ground_truth):
    original_dir = base_dir / "original model outputs" / model_folder

    original_file = original_dir / "math_generated_outputs_original.jsonl"
    if not original_file.exists():
        raise FileNotFoundError(f"Original JSONL not found: {original_file}")

    original_predictions = load_predictions(original_file)
    original_em = compute_em_percentage(original_predictions, ground_truth)

    variant_em_maps = {}
    all_keys = set()

    for variant, folder_name, _column_name in VARIANT_CONFIG:
        variant_dir = base_dir / folder_name / model_folder
        variant_files = sorted(variant_dir.glob(f"math_generated_outputs_{variant}_(*).jsonl"))

        em_map = {}
        for file_path in variant_files:
            key = extract_variant_key(file_path.name, variant)
            if key is None:
                continue
            em_map[key] = compute_em_percentage(load_predictions(file_path), ground_truth)

        variant_em_maps[variant] = em_map
        all_keys.update(em_map.keys())

    keys = sorted(all_keys, key=safe_sort_key)

    original_row = {"model": "original"}
    for _variant, _folder_name, column_name in VARIANT_CONFIG:
        original_row[column_name] = format_em(original_em)

    rows = [original_row]

    for key in keys:
        row = {"model": key}
        for variant, _folder_name, column_name in VARIANT_CONFIG:
            row[column_name] = format_em(variant_em_maps[variant].get(key, 0.0))
        rows.append(row)

    return rows


def generate_em_summary(model_folder, ground_truth_file, output_file):
    base_dir = Path(__file__).resolve().parent
    ground_truth_path = (base_dir / ground_truth_file).resolve()
    output_path = (base_dir / output_file).resolve()

    ground_truth = load_ground_truth(ground_truth_path)
    rows = build_em_rows(base_dir=base_dir, model_folder=model_folder, ground_truth=ground_truth)

    fieldnames = ["model"] + [column_name for _variant, _folder_name, column_name in VARIANT_CONFIG]

    with output_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"EM summary saved to: {output_path}")


def main():
    '''
    Main function to generate the EM summary CSV.
    Adjust the `model_folder`, `ground_truth_file`, and `output_file` variables as needed to specify the model, ground truth CSV file, and output CSV file for which you want to generate the summary. The output will be saved as a CSV file in the same directory as this script.
    '''
    model_folder = "Qwen2.5-Math-1.5B-Instruct"
    ground_truth_file = "ground truth.csv"
    output_file = "em_summary.csv"

    generate_em_summary(
        model_folder=model_folder,
        ground_truth_file=ground_truth_file,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
