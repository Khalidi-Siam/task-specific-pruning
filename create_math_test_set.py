import pandas as pd
from datasets import load_dataset
import os
import re

def _extract_ground_truth(answer_text):
    """Extract the final GSM8K answer from text like '...\n#### 72'."""
    text = str(answer_text)

    match = re.search(r"####\s*([^\n\r]+)", text)
    if match:
        extracted = match.group(1).strip()
    else:
        extracted = text.strip().splitlines()[-1].strip() if text.strip() else ""

    # Normalize common formatting noise in GSM8K final answers.
    extracted = extracted.replace(",", "")
    if extracted.startswith("$"):
        extracted = extracted[1:].strip()

    return extracted


def _build_prompts_df(test_data):
    return pd.DataFrame({"prompt": [row["question"] for row in test_data]})


def _build_ground_truth_df(test_data):
    return pd.DataFrame(
        {
            "prompt_No": list(range(1, len(test_data) + 1)),
            "answer": [_extract_ground_truth(row.get("answer", "")) for row in test_data],
        }
    )


def create_math_test_set(output_path=None, ground_truth_output_path=None):
    if output_path is None:
        output_path = os.path.join("datasets", "math_prompts.csv")
    if ground_truth_output_path is None:
        ground_truth_output_path = os.path.join("datasets", "ground truth.csv")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(ground_truth_output_path) or ".", exist_ok=True)

    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]

    prompts_df = _build_prompts_df(test_data)
    ground_truth_df = _build_ground_truth_df(test_data)

    prompts_df.to_csv(output_path, index=False)
    ground_truth_df.to_csv(ground_truth_output_path, index=False)

    print(f"Math prompts dataset created: {output_path} ({len(prompts_df)} rows)")
    print(
        f"Math ground-truth dataset created: {ground_truth_output_path} ({len(ground_truth_df)} rows)"
    )


# if __name__ == "__main__":
#     create_math_test_set()