import csv
import json
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util


EXPERIMENTS = {
    "pruned": {"folder": "pruned model outputs", "tag": "pruned"},
    "random33": {"folder": "random33 model outputs", "tag": "random33"},
    "random42": {"folder": "random42 model outputs", "tag": "random42"},
    "reversed": {"folder": "reversed model outputs", "tag": "reversed"},
}


def normalize_response_text(value):
    """Return a safe, non-empty string for embedding."""
    if value is None:
        return "<EMPTY_RESPONSE>"
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned if cleaned else "<EMPTY_RESPONSE>"


def load_records_by_prompt_no(file_path: Path):
    records = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt_no = item.get("prompt_no")
            if isinstance(prompt_no, str) and not prompt_no.isdigit():
                continue
            records[str(prompt_no)] = normalize_response_text(item.get("response", ""))
    return records


def compute_sbert_similarity(ref_records, cand_records, model, device):
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

    # Encode all responses at once
    ref_embeddings = model.encode(refs, convert_to_tensor=True, device=device)
    cand_embeddings = model.encode(cands, convert_to_tensor=True, device=device)

    # Compute cosine similarity for each pair
    cosine_scores = util.cos_sim(cand_embeddings, ref_embeddings).diagonal()
    return cosine_scores.mean().item()


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


def write_csv(rows, output_csv: Path, active_experiments: dict):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level(%)"] + list(active_experiments.keys()))
        for row in rows:
            writer.writerow(row)


def generate_sbert_csv(model_name: str, task_type: str, output_csv: Path | None = None):
    original_file = (
         "original model outputs"
        / model_name
        / f"{task_type}_generated_outputs_original.jsonl"
    )

    if not original_file.exists():
        raise FileNotFoundError(f"Reference file not found: {original_file}")

    if output_csv is None:
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        output_csv = "semantic_similarity_result" / f"sbert_summary_{task_type}_{safe_model}.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    # Load SBERT model
    sbert_model = SentenceTransformer("all-mpnet-base-v2", device=device)

    ref_records = load_records_by_prompt_no(original_file)

    # Only keep experiments whose folder actually exists
    active_experiments = {}
    for column, cfg in EXPERIMENTS.items():
        run_folder = cfg["folder"] / model_name
        if not run_folder.exists():
            print(f"[skip] Folder not found, skipping variant '{column}': {run_folder}")
            continue
        active_experiments[column] = cfg

    level_scores = {}
    for column, cfg in active_experiments.items():
        run_folder = cfg["folder"] / model_name
        level_files = find_level_files(run_folder, task_type, cfg["tag"])

        if not level_files:
            print(f"[skip] No output files found for variant '{column}' in {run_folder}")
            continue

        level_scores[column] = {}
        for level, candidate_file in level_files.items():
            cand_records = load_records_by_prompt_no(candidate_file)
            avg_sim = compute_sbert_similarity(ref_records, cand_records, sbert_model, device)
            level_scores[column][level] = avg_sim
            print(f"{column:<8} level {level:>7}: CosineSim={avg_sim:.4f}")

    # Only include columns that actually have data
    active_experiments = {col: cfg for col, cfg in active_experiments.items() if col in level_scores}

    if not active_experiments:
        print("[warn] No variant data found. CSV will not be written.")
        return None

    all_levels = sorted(
        {level for col_scores in level_scores.values() for level in col_scores.keys()},
        key=lambda x: float(x),
    )

    rows = [["original"] + [1] * len(active_experiments)]
    for level in all_levels:
        rows.append(
            [
                level,
                *[
                    f"{level_scores[col][level]:.4f}" if level in level_scores[col] else "not found"
                    for col in active_experiments
                ],
            ]
        )

    write_csv(rows, output_csv, active_experiments)
    print(f"Saved CSV: {output_csv}")
    return output_csv


# Update these two values when you want a different run
model_name = "Qwen2.5-Coder-7B-Instruct"
task_type = "conversational"
generate_sbert_csv(model_name=model_name, task_type=task_type)