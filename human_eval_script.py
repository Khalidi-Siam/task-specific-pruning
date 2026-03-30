import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import gc

from human_eval.evaluation import evaluate_functional_correctness


# =====================================================
# SANITIZER 1: REMOVE MARKDOWN FENCES
# =====================================================
def remove_markdown_fences(text: str) -> str:
    text = text.replace("```python", "")
    text = text.replace("```", "")
    return text.strip()


# =====================================================
# SANITIZER 2: REMOVE EXPLANATORY GARBAGE
# =====================================================
def remove_explanatory_text(text: str) -> str:
    lines = text.split('\n')

    # Find first line that looks like code
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if (
            stripped.startswith('def ')
            or stripped.startswith('from ')
            or stripped.startswith('import ')
        ):
            start_idx = i
            break

    if start_idx is None:
        return text.strip()

    end_idx = len(lines)
    explanation_markers = [
        'This function', 'This solution', 'This implementation',
        'Note that', 'Here is', "Here's", 'The function',
        'In this', 'For example', 'Example usage',
        '# Test', 'print(', 'assert ', 'if __name__'
    ]

    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if any(line.startswith(marker) for marker in explanation_markers):
            end_idx = i
            break

    return '\n'.join(lines[start_idx:end_idx]).strip()


# =====================================================
# OPTIONAL: FIX TYPING IMPORTS (SAFE, DETERMINISTIC)
# =====================================================
def maybe_add_typing_import(code: str) -> str:
    tokens = ["List[", "Tuple[", "Dict[", "Optional[", "Set["]
    if any(tok in code for tok in tokens):
        if "from typing import" not in code:
            return "from typing import *\n\n" + code
    return code


# =====================================================
# MAIN FUNCTION
# =====================================================
def human_eval_deterministic(
    model_path: str,
    seed: int = 42,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    output_dir: str = "humaneval_official_eval",
):
    # =====================================================
    # DETERMINISM
    # =====================================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================================================
    # MODEL
    # =====================================================
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================
    # DATASET (OFFICIAL PROMPTS)
    # =====================================================
    dataset = load_dataset("openai_humaneval", split="test")

    os.makedirs(output_dir, exist_ok=True)
    sample_file = os.path.join(output_dir, "samples.jsonl")

    records = []

    # =====================================================
    # GENERATION (GREEDY, PASS@1)
    # =====================================================
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        prompts = batch["prompt"]

        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are generating solutions for the HumanEval benchmark. Output ONLY valid Python code. Output ONLY the function definition requested. Do NOT include explanations, comments outside the function, examples, tests, prints, or prose. Do NOT include markdown, backticks, or formatting. Do NOT add helper text before or after the code. The output must consist of one or more top-level def functions only. No top-level statements other than function definitions. No if __name__ == \"__main__\" blocks. No print statements. No test cases. No explanations. Respond with Python code ONLY."},
                    {"role": "user", "content": p},
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        for j in range(len(prompts)):
            prompt_len = inputs.input_ids[j].shape[-1]
            gen = outputs[j][prompt_len:]
            text = tokenizer.decode(gen, skip_special_tokens=True)

            # =================================================
            # SANITIZATION PIPELINE (ONLY YOUR TWO)
            # =================================================
            code = remove_markdown_fences(text)
            code = remove_explanatory_text(code)
            code = maybe_add_typing_import(code)

            records.append({
                "task_id": batch["task_id"][j],
                "completion": code
            })

    # =====================================================
    # SANITY CHECK
    # =====================================================
    assert len(records) == 164, "HumanEval requires exactly 164 samples"
    assert len({r["task_id"] for r in records}) == 164, "Duplicate task_ids found"

    # =====================================================
    # WRITE JSONL (OFFICIAL FORMAT)
    # =====================================================
    with open(sample_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote samples to {sample_file}")

    # =====================================================
    # OFFICIAL HUMAN_EVAL EVALUATION
    # =====================================================
    results = evaluate_functional_correctness(
        sample_file=sample_file,
        k=[1],          # deterministic Pass@1
        n_workers=4,
        timeout=3.0
    )

    print("DONE:", results)
    return results


# =====================================================
# HELPERS — map model type + value to folder names
# =====================================================

# Maps each model type to (model_name_suffix, result_folder_prefix)
_TYPE_MAP = {
    "Fine Tuned Model":     ("finetuned_({value})ft",  "humaneval_output_finetuned_({value})"),
    "Pruned Model":         ("pruned_({value})p",      "humaneval_output_pruned_({value})"),
    "Random33 Pruned Model":("random33_({value})p",    "humaneval_output_random33_({value})"),
    "Random42 Pruned Model":("random42_({value})p",    "humaneval_output_random42_({value})"),
    "Reverse Pruned Model": ("reversed_({value})p",    "humaneval_output_reversed_({value})"),
}

def get_model_name(model_type: str, output_dir: str, value) -> str:
    """Return the model folder name for the given type and pruning value."""
    entry = _TYPE_MAP.get(model_type)
    if entry is None:
        raise ValueError(f"Unknown model type: {model_type!r}")
    suffix_template, _ = entry
    return f"{output_dir}_{suffix_template.replace('{value}', str(value))}"

def get_result_subfolder(model_type: str, value) -> str:
    """Return the HumanEval output subfolder name for the given type and value."""
    entry = _TYPE_MAP.get(model_type, (None, f"humaneval_output_({{value}})"))
    _, folder_template = entry
    return folder_template.replace("{value}", str(value))


# =====================================================
# USAGE
# =====================================================

values = [5.405, 10.135, 20.27, 25, 30.405, 35.135]
output_dir = "Qwen2.5-Coder-7B-Instruct"
type = "Fine Tuned Model"  # choose from "Fine Tuned Model" | "Pruned Model" | "Random33 Pruned Model" | "Random42 Pruned Model" | "Reverse Pruned Model"

for value in values:
    print(f"\n{'='*60}")
    print(f"Starting evaluation for {value}% pruning")
    print(f"{'='*60}")

    model_name      = get_model_name(type, output_dir, value)
    result_subfolder = get_result_subfolder(type, value)
    model_path      = os.path.join(type, output_dir, model_name)

    print(f"Model path: {model_path}")

    human_eval_deterministic(
        model_path=model_path,
        seed=42,
        batch_size=16,
        max_new_tokens=512,
        output_dir=os.path.join("Human_eval_result", type, output_dir, result_subfolder)
    )
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()