import csv
import json
import re
from pathlib import Path

import torch
from bert_score import score


VOLUME_PATH = "/root/datasets"
BASE_DIR = Path(VOLUME_PATH) / "result"

EXPERIMENTS = {
    "pruned": {
        "folder": "pruning model outputs",
        "tag": "pruned",
    },
    "random33": {
        "folder": "random33 model outputs",
        "tag": "random33",
    },
    "random42": {
        "folder": "random42 model outputs",
        "tag": "random42",
    },
    "reversed": {
        "folder": "reversed model outputs",
        "tag": "reversed",
    },
}


def normalize_response_text(value):
    """Return a safe, non-empty string for BERTScore tokenization."""
    if value is None:
        return "<EMPTY_RESPONSE>"
    if not isinstance(value, str):
        value = str(value)

    cleaned = value.strip()
    if not cleaned:
        return "<EMPTY_RESPONSE>"
    return cleaned


def load_records_by_prompt_no(file_path: Path):
    records = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt_no = item.get("prompt_no")

            # Skip metadata rows such as STARTING/ENDING.
            if isinstance(prompt_no, str) and not prompt_no.isdigit():
                continue

            records[str(prompt_no)] = normalize_response_text(item.get("response", ""))
    return records


def compute_f1(ref_records, cand_records, device):
    ref_keys = set(ref_records.keys())
    cand_keys = set(cand_records.keys())

    if ref_keys != cand_keys:
        missing_in_cand = sorted(ref_keys - cand_keys)
        missing_in_ref = sorted(cand_keys - ref_keys)
        raise ValueError(
            "prompt_no mismatch between reference and candidate files. "
            f"Missing in candidate: {missing_in_cand[:5]} | "
            f"Missing in reference: {missing_in_ref[:5]}"
        )

    ordered_prompt_nos = sorted(ref_keys, key=lambda x: int(x))
    refs = [ref_records[k] for k in ordered_prompt_nos]
    cands = [cand_records[k] for k in ordered_prompt_nos]

    _, _, f1 = score(
        cands,
        refs,
        lang="en",
        rescale_with_baseline=True,
        device=device,
    )
    return f1.mean().item()


def find_level_files(folder: Path, task_type: str, tag: str):
    pattern = re.compile(
        rf"^{re.escape(task_type)}_generated_outputs_{re.escape(tag)}_\((\d+(?:\.\d+)?)\)\.jsonl$"
    )

    level_to_file = {}
    for file_path in folder.glob(f"{task_type}_generated_outputs_{tag}_(*).jsonl"):
        match = pattern.match(file_path.name)
        if not match:
            continue
        level = match.group(1)
        level_to_file[level] = file_path
    return level_to_file


def write_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level(%)", "pruned", "random33", "random42", "reversed"])
        for row in rows:
            writer.writerow(row)


def generate_bertscore_csv(model_name: str, task_type: str, output_csv: Path | None = None):
    original_file = (
        BASE_DIR
        / "original model outputs"
        / model_name
        / f"{task_type}_generated_outputs_original.jsonl"
    )

    if not original_file.exists():
        raise FileNotFoundError(f"Reference file not found: {original_file}")

    if output_csv is None:
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        output_csv = BASE_DIR / f"bert_score_summary_{task_type}_{safe_model}.csv"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    ref_records = load_records_by_prompt_no(original_file)

    level_scores = {}
    for column, cfg in EXPERIMENTS.items():
        run_folder = BASE_DIR / cfg["folder"] / model_name
        level_files = find_level_files(run_folder, task_type, cfg["tag"])

        if not level_files:
            raise FileNotFoundError(
                f"No files found for {column} in {run_folder} for task {task_type}"
            )

        level_scores[column] = {}
        for level, candidate_file in level_files.items():
            cand_records = load_records_by_prompt_no(candidate_file)
            avg_f1 = compute_f1(ref_records, cand_records, device)
            level_scores[column][level] = avg_f1
            print(f"{column:<8} level {level:>7}: F1={avg_f1:.4f}")

    all_levels = sorted(
        {level for col_scores in level_scores.values() for level in col_scores.keys()},
        key=lambda x: float(x),
    )

    rows = [["original", 1, 1, 1, 1]]
    for level in all_levels:
        rows.append(
            [
                level,
                f"{level_scores['pruned'].get(level, float('nan')):.4f}",
                f"{level_scores['random33'].get(level, float('nan')):.4f}",
                f"{level_scores['random42'].get(level, float('nan')):.4f}",
                f"{level_scores['reversed'].get(level, float('nan')):.4f}",
            ]
        )

    write_csv(rows, output_csv)
    print(f"Saved CSV: {output_csv}")
    return output_csv


    # Update only these two values when you want a different run.
model_name = "Qwen2.5-Coder-7B-Instruct"
task_type = "qna"
generate_bertscore_csv(model_name=model_name, task_type=task_type)