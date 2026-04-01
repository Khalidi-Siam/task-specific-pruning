# Task-Specific Neuron Pruning for LLMs

A research pipeline for **task-specific structural pruning** of large language models (LLMs). The pipeline identifies task-relevant neurons via selectivity scoring based on neuron activation, physically prunes neurons from each MLP layers, fine-tunes the pruned models, and evaluates the results across multiple tasks and pruning strategies.

You will find all the supplementary material in this anonymous link: https://anonymous.4open.science/r/task-specific-pruning-EC02/
---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Step 1 — Build Datasets](#step-1--build-datasets)
   - [Step 2 — Capture Neuron Activations & Compute Selectivity Scores](#step-2--capture-neuron-activations--compute-selectivity-scores)
   - [Step 3 — Prune the Model](#step-3--prune-the-model)
   - [Step 4 — Fine-Tune](#step-4--fine-tune)
   - [Step 5 — Generate Outputs](#step-5--generate-outputs)
   - [Step 6 — Evaluate / Benchmark](#step-6--evaluate--benchmark)
5. [Output Directory Structure](#output-directory-structure)
6. [Important Notes](#important-notes)

---

## Research Overview

This research investigates whether pruning neurons that are **not relevant to a target task** (e.g., math or code) can reduce model size while preserving task performance in task specific large language models.

Three pruning strategies are supported and compared:

| Strategy | Description |
|---|---|
| **Selective Pruning** | Removes neurons with the *lowest* selectivity score for the target task |
| **Reverse Pruning** | Removes neurons with the *highest* selectivity score (ablation baseline) |
| **Random Pruning** | Removes neurons chosen at random (controlled baseline, seeded for reproducibility) |

Evaluated tasks: **Math**, **Code**, **QnA**, **Conversational**

---

## Environment Setup

### Requirements

- Python 3.10+
- CUDA 12.1 (GPU required; CPU inference is not supported for large models)
- ~40–80 GB free disk space per model (for storing pruned variants)

### Installation

```bash
# 1. (Recommended) Create and activate a virtual environment
python -m venv env
# Windows
env\Scripts\activate
# Linux / macOS
source env/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt
```

> **Note:** `requirements.txt` points to `https://download.pytorch.org/whl/cu121` for CUDA 12.1-compatible PyTorch wheels. If you are using a different CUDA version, replace this index URL accordingly before installing.

### Key Dependencies

| Package | Purpose |
|---|---|
| `torch`, `transformers`, `accelerate` | Model loading, generation, training |
| `peft` | LoRA-based fine-tuning |
| `datasets`, `evaluate` | Dataset loading and metrics |
| `bert-score` | BERTScore for QnA evaluation |
| `sentence-transformers` | SBERT cosine similarity for conversational evaluation |
| `human-eval` | HumanEval benchmark for code evaluation |
| `scikit-learn`, `numpy`, `pandas` | Selectivity score computation and analysis |

---

## Project Structure

```
task-specific-pruning/
│
├── build_datasets_script.py          # Step 1: Download and label datasets
├── run_selective_pruning_experiment.py  # Step 2: Capture activations & compute scores
│
├── selective_pruning.py              # Step 3a: Selective pruning
├── random_pruning.py                 # Step 3b: Random pruning (baseline)
├── reverse_pruning.py                # Step 3c: Reverse pruning (ablation)
│
├── Fine Tune/
│   ├── Math Finetune/                # Step 4: Fine-tuning pipeline for math models
│   └── Code Finetune/                # Step 4: Fine-tuning pipeline for code models
│
├── original_model_output_script.py   # Step 5a: Generate outputs for the original model
├── prune_model_output_script.py      # Step 5b: Generate outputs for selectively pruned models
├── random_prune_model_output_script.py  # Step 5c: Generate outputs for randomly pruned models
├── reverse_prune_model_output_script.py # Step 5d: Generate outputs for reverse-pruned models
├── finetune_model_output_script.py   # Step 5e: Generate outputs for fine-tuned models
│
├── extract_answers.py                # Step 6a: Math benchmark — Exact Match (EM) summary
├── original_model_human_eval_script.py  # Step 6b: HumanEval inference for the original model
├── human_eval_script.py              # Step 6b: HumanEval inference for pruned/fine-tuned models
├── human_eval_summary.py             # Step 6b: Aggregate all HumanEval results into a summary CSV
├── bert_score_eval.py                # Step 6c: QnA benchmark — BERTScore F1 summary
├── sbert_score_eval.py               # Step 6d: Conversational benchmark — SBERT cosine similarity
├── trap_count.py                     # Step 6e: Trap count summary (all task types)
│
├── datasets/                         # Auto-created: downloaded & labeled datasets
├── code_prompts.csv                  # Code task prompts (used for trap count)
├── extracted_sample_1k.csv           # Pre-extracted code dataset sample
│
├── load_model.py                     # Helper: loads a model and runs inference
├── load_and_lebel_datasets.py        # Helper: downloads and labels datasets
├── dataset_helper.py                 # Helper: builds math/code dataset splits
├── neuron_activation_per_layer.py    # Helper: captures per-layer neuron activations
├── compute_selectivity_score.py      # Helper: computes per-neuron selectivity scores
├── output_jsonl.py                   # Helper: single-sample JSONL output writer
├── output_jsonl_batch.py             # Helper: batched JSONL output writer
├── output_text.py                    # Helper: plain-text output writer
├── dialogue_preprocess.py            # Helper: conversational dataset preprocessor
│
└── requirements.txt
```

---

## Pipeline Walkthrough

### Step 1 — Build Datasets

**Script:** `build_datasets_script.py`

Downloads and labels all datasets needed for both math and code model experiments.

```bash
python build_datasets_script.py
```

**Datasets used:**

| Label | Category | Source |
|---|---|---|
| 1 | Conversational | `suriya7/everyday-Conversational-cleaned` |
| 2 | QnA | `rajpurkar/squad` |
| 3 | Code | `extracted_sample_1k.csv` (local) |
| 4 | Math | `gsm8k` |

**Output:**
- Processed and labeled datasets saved to `datasets/` in the project root.

---

### Step 2 — Capture Neuron Activations & Compute Selectivity Scores

**Script:** `run_selective_pruning_experiment.py`

Runs the model on the labeled dataset, captures per-neuron activations across all MLP layers, and computes a selectivity score for each neuron with respect to the target task category.

**Before running**, update the following variables inside the script:

```python
model_id        = "Qwen/Qwen2.5-Math-7B-Instruct"  # HuggingFace model ID
batch_size      = 32   # 32 for 7B models, 64 for 1.5B models
output_dir      = "model activation/Qwen2.5-Math-7B-Instruct"
target_categories    = [4]     # 4 = math, 3 = code
distractor_categories = [1, 2]  # conversational + QnA
```

```bash
python run_selective_pruning_experiment.py
```

**Output (stored in `model activation/<model-name>/`):**
```
model activation/
└── Qwen2.5-Math-7B-Instruct/
    ├── config.txt                      # Run configuration log
    ├── neuron_activations/             # Raw per-layer activation data
    │   ├── layer_0_activations.npy
    │   ├── layer_1_activations.npy
    │   └── ...
    └── selectivity_scores/             # Per-neuron selectivity scores
        ├── layer_0_selectivity.csv
        ├── layer_1_selectivity.csv
        └── ...
```

---

### Step 3 — Prune the Model

Three scripts are available. All physically modify and save the model weights.

#### 3a. Selective Pruning

**Script:** `selective_pruning.py`

Prunes neurons with the **lowest** selectivity scores — i.e., neurons least relevant to the target task.

**Before running**, update these variables:

```python
output_dir   = "Qwen2.5-Math-7B-Instruct"  # must match the name used in Step 2
model_id     = f"Qwen/{output_dir}"
pruning_levels = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135]  # 7B model
# pruning_levels = [5.71, 10, 15.71, 20, 25.71, 30, 35.71]  # 1.5B model
```

```bash
python selective_pruning.py
```

**Output (stored in `Pruned Model/<model-name>/`):**
```
Pruned Model/
└── Qwen2.5-Math-7B-Instruct/
    ├── Qwen2.5-Math-7B-Instruct-pruned-5.405p/
    ├── Qwen2.5-Math-7B-Instruct-pruned-10.135p/
    └── ...   (one folder per pruning level)
```

---

#### 3b. Random Pruning

**Script:** `random_pruning.py`

Prunes neurons **randomly** at each layer. Use a fixed seed for reproducibility.

**Before running**, update these variables:

```python
output_dir   = "Qwen2.5-Math-7B-Instruct"
model_id     = f"Qwen/{output_dir}"
seed         = 42   # default; change for multiple random baselines (e.g., 33, 42)
pruning_levels = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135]  # 7B
```

```bash
python random_pruning.py
```

**Output (stored in `Random pruned model<seed>/<model-name>/`):**
```
Random pruned model42/
└── Qwen2.5-Math-7B-Instruct/
    ├── Qwen2.5-Math-7B-Instruct-pruned-5.405p/
    ├── Qwen2.5-Math-7B-Instruct-pruned-10.135p/
    └── ...
```

---

#### 3c. Reverse Pruning

**Script:** `reverse_pruning.py`

Prunes neurons with the **highest** selectivity scores — the opposite of selective pruning. Requires the selectivity scores from Step 2.

**Before running**, update these variables:

```python
output_dir   = "Qwen2.5-Math-7B-Instruct"
model_id     = f"Qwen/{output_dir}"
pruning_levels = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135]  # 7B
```

```bash
python reverse_pruning.py
```

**Output (stored in `Reverse Pruned Model/<model-name>/`):**
```
Reverse Pruned Model/
└── Qwen2.5-Math-7B-Instruct/
    ├── Qwen2.5-Math-7B-Instruct-pruned-5.405p/
    ├── Qwen2.5-Math-7B-Instruct-pruned-10.135p/
    └── ...
```

---

### Step 4 — Fine-Tune

Fine-tuning is only applicable to **selectively pruned** models and is supported for **math** and **code** tasks separately.

Fine-tuning pipelines are located under the `Fine Tune/` directory:

```
Fine Tune/
├── Math Finetune/    # Fine-tune pipeline for math models
└── Code Finetune/    # Fine-tune pipeline for code models
```

Run the appropriate pipeline for your model type. Refer to the scripts inside each subdirectory for configuration details (model path, pruning level, training hyperparameters).

**Output (stored in `Fine Tuned Model/<model-name>/`):**
```
Fine Tuned Model/
└── Qwen2.5-Math-7B-Instruct/
    ├── Qwen2.5-Math-7B-Instruct-5.405ft/
    ├── Qwen2.5-Math-7B-Instruct-10.135ft/
    └── ...
```

---

### Step 5 — Generate Outputs

There is a dedicated output generation script for each model variant. All scripts share the same configurable variables and call `load_model` internally.

**Common configuration variables (set at top of each script):**

```python
output_dir   = "Qwen2.5-Math-1.5B-Instruct"  # model name (used as folder name)
task_types   = ["math"]   # choose from: "math", "code", "conversational", "qna"
batch_size   = 16         # adjust based on GPU memory; set to 1 to measure VRAM usage
half_precision = True
chat_template  = True
values       = [5.71, 10, 15.71, 20, 25.71, 30, 35.71]  # pruning levels (1.5B)
# values     = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135]  # pruning levels (7B)
```

> **VRAM measurement**: Set `batch_size = 1` to capture per-response VRAM usage. Batch mode does not currently measure VRAM.

#### 5a. Original (Unpruned) Model

**Script:** `original_model_output_script.py`

Additional variable:
```python
model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # HuggingFace model ID
```

```bash
python original_model_output_script.py
```

**Output:** `original model outputs/<model-name>/`
```
original model outputs/
└── Qwen2.5-Math-1.5B-Instruct/
    ├── math_generated_outputs_original.jsonl
    └── math_generation_metrics_original.json
```

---

#### 5b. Selectively Pruned Model

**Script:** `prune_model_output_script.py`

```bash
python prune_model_output_script.py
```

**Output:** `pruned model outputs/<model-name>/`
```
pruned model outputs/
└── Qwen2.5-Math-1.5B-Instruct/
    ├── math_generated_outputs_pruned_(5.71).jsonl
    ├── math_generation_metrics_pruned_(5.71).json
    ├── math_generated_outputs_pruned_(10).jsonl
    └── ...   (one pair of files per pruning level)
```

---

#### 5c. Randomly Pruned Model

**Script:** `random_prune_model_output_script.py`

Additional variable:
```python
seed = 42  # must match the seed used during random pruning
```

```bash
python random_prune_model_output_script.py
```

**Output:** `random<seed> model outputs/<model-name>/`
```
random42 model outputs/
└── Qwen2.5-Math-1.5B-Instruct/
    ├── math_generated_outputs_random42_(5.71).jsonl
    ├── math_generation_metrics_random42_(5.71).json
    └── ...
```

---

#### 5d. Reverse-Pruned Model

**Script:** `reverse_prune_model_output_script.py`

```bash
python reverse_prune_model_output_script.py
```

**Output:** `reversed model outputs/<model-name>/`
```
reversed model outputs/
└── Qwen2.5-Math-1.5B-Instruct/
    ├── math_generated_outputs_reversed_(5.71).jsonl
    ├── math_generation_metrics_reversed_(5.71).json
    └── ...
```

---

#### 5e. Fine-Tuned Model

**Script:** `finetune_model_output_script.py`

```bash
python finetune_model_output_script.py
```

**Output:** `finetuned model outputs/<model-name>/`
```
finetuned model outputs/
└── Qwen2.5-Math-1.5B-Instruct/
    ├── math_generated_outputs_finetuned_(5.71).jsonl
    ├── math_generation_metrics_finetuned_(5.71).json
    └── ...
```

---

### Step 6 — Evaluate / Benchmark

#### 6a. Math — Exact Match (EM) Score

**Script:** `extract_answers.py`

Extracts `\boxed{...}` answers from all model output JSONL files and computes Exact Match (EM) percentage against the ground truth.

**Before running**, update these variables:

```python
model_folder       = "Qwen2.5-Math-1.5B-Instruct"
ground_truth_file  = Path("datasets") / "ground truth.csv"
output_file        = f"em_summary_{model_folder}.csv"
```

```bash
python extract_answers.py
```

**Output:** `em_summary_<model-name>.csv` (saved in project root)

The CSV summarises EM (%) for the original model and all pruning variants across all pruning levels.

---

#### 6b. Code — HumanEval Pass@1

HumanEval inference is split into two scripts: one for the **original model** and one for all **pruned / fine-tuned variants**. A third script then aggregates all results into a single summary CSV.

##### 6b-i. Run HumanEval — Original Model

**Script:** `original_model_human_eval_script.py`

**Before running**, update these variables:

```python
output_dir = "Qwen2.5-Coder-7B-Instruct"   # model folder name
model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"  # HuggingFace model ID
```

```bash
python original_model_human_eval_script.py
```

**Output:** `Human_eval_result/Original Model/<model-name>/`
```
Human_eval_result/
└── Original Model/
    └── Qwen2.5-Coder-7B-Instruct/
        ├── samples.jsonl
        └── samples.jsonl_results.jsonl
```

---

##### 6b-ii. Run HumanEval — Pruned / Fine-Tuned Variants

**Script:** `human_eval_script.py`

Runs the official HumanEval benchmark (164 problems) with greedy decoding (Pass@1) for each pruning level of a given variant type.

**Before running**, update these variables:

```python
output_dir = "Qwen2.5-Coder-7B-Instruct"
type       = "Fine Tuned Model"   # choose from:
                                  #   "Pruned Model"
                                  #   "Fine Tuned Model"
                                  #   "Random33 Pruned Model"
                                  #   "Random42 Pruned Model"
                                  #   "Reverse Pruned Model"
values     = [5.405, 10.135, 20.27, 25, 30.405, 35.135]
```

```bash
python human_eval_script.py
```

**Output:** `Human_eval_result/<type>/<model-name>/<result-subfolder>/`
```
Human_eval_result/
├── Original Model/
│   └── Qwen2.5-Coder-7B-Instruct/
│       ├── samples.jsonl
│       └── samples.jsonl_results.jsonl
├── Pruned Model/
│   └── Qwen2.5-Coder-7B-Instruct/
│       ├── humaneval_output_pruned_(5.405)/
│       │   ├── samples.jsonl
│       │   └── samples.jsonl_results.jsonl
│       └── ...
├── Fine Tuned Model/
│   └── Qwen2.5-Coder-7B-Instruct/
│       ├── humaneval_output_finetuned_(5.405)/
│       └── ...
├── Random33 Pruned Model/
│   └── Qwen2.5-Coder-7B-Instruct/
│       ├── humaneval_output_random33_(5.405)/
│       └── ...
├── Random42 Pruned Model/
│   └── Qwen2.5-Coder-7B-Instruct/
│       ├── humaneval_output_random42_(5.405)/
│       └── ...
└── Reverse Pruned Model/
    └── Qwen2.5-Coder-7B-Instruct/
        ├── humaneval_output_reversed_(5.405)/
        └── ...
```

---

##### 6b-iii. Aggregate HumanEval Results — Summary CSV

**Script:** `human_eval_summary.py`

Reads all `samples.jsonl_results.jsonl` files under `Human_eval_result/` and computes Pass@1 (%) for every pruning level and variant. Missing variant folders are silently skipped; missing individual pruning level results are shown as `not found` in the CSV.

**Before running**, update this variable:

```python
model_name = "Qwen2.5-Coder-7B-Instruct"  # must match folder name inside Human_eval_result/
```

```bash
python human_eval_summary.py
```

**Output:** `humaneval_summary_<model-name>.csv` (saved in project root)

The CSV summarises Pass@1 (%) for the original model and all pruning variants across all pruning levels.

---

#### 6c. QnA — BERTScore F1

**Script:** `bert_score_eval.py`

Computes BERTScore F1 between each pruned model's outputs and the original model's outputs on the QnA task.

**Before running**, update these variables:

```python
model_name = "Qwen2.5-Coder-7B-Instruct"
task_type  = "qna"
```

```bash
python bert_score_eval.py
```

**Output:** `semantic_similarity_result/bert_score_summary_<task>_<model>.csv`

The CSV contains F1 scores for each pruning strategy (selective, random×2, reverse) across all pruning levels.

---

#### 6d. Conversational — SBERT Cosine Similarity

**Script:** `sbert_score_eval.py`

Computes SBERT cosine similarity between each pruned model's conversational outputs and the original model's outputs using `all-mpnet-base-v2`.

**Before running**, update these variables:

```python
model_name = "Qwen2.5-Coder-7B-Instruct"
task_type  = "conversational"
```

```bash
python sbert_score_eval.py
```

**Output:** `semantic_similarity_result/sbert_summary_<task>_<model>.csv`

---

#### 6e. Trap Count (All Task Types)

**Script:** `trap_count.py`

Counts "trapped" responses — responses where the model generated ≥ 1024 tokens, indicating it got stuck in a loop or failed to terminate. Works for all task types.

**Before running**, update these variables:

```python
model_folder = "Qwen2.5-Math-1.5B-Instruct"
task_type    = "math"   # math, code, qna, conversational
output_file  = f"trap_summary_{task_type}.csv"
```

```bash
python trap_count.py
```

**Output:** `trap_summary_<task>.csv` (saved in project root)

The CSV summarises trap percentage (%) for all pruning variants across all pruning levels.

> **Note on code task:** Since HumanEval prompts are used for code benchmarking, a separate `code_prompts.csv` is provided for trap count calculation on the code task, as the HumanEval set is not directly used for trap counting.

> **Note on distractor tasks:** QnA and conversational tasks do not have a fine-tuned variant. The trap count script automatically excludes the `finetuned` column for these tasks.

---

## Output Directory Structure

After running the full pipeline, your directory will look like this:

```
task-specific-pruning/
│
├── datasets/                              # Step 1 output
│   └── ground truth.csv
│
├── model activation/                      # Step 2 output
│   └── <model-name>/
│       ├── config.txt
│       ├── neuron_activations/
│       └── selectivity_scores/
│
├── Pruned Model/                          # Step 3a output
│   └── <model-name>/
│       ├── <model-name>-pruned-Xp/
│       └── ...
│
├── Random pruned model42/                 # Step 3b output (seed 42)
│   └── <model-name>/
│       └── ...
│
├── Reverse Pruned Model/                  # Step 3c output
│   └── <model-name>/
│       └── ...
│
├── Fine Tuned Model/                      # Step 4 output
│   └── <model-name>/
│       └── ...
│
├── original model outputs/                # Step 5a output
│   └── <model-name>/
│       ├── <task>_generated_outputs_original.jsonl
│       └── <task>_generation_metrics_original.json
│
├── pruning model outputs/                 # Step 5b output
│   └── <model-name>/
│       └── ...
│
├── random42 model outputs/                # Step 5c output
│   └── <model-name>/
│       └── ...
│
├── reversed model outputs/                # Step 5d output
│   └── <model-name>/
│       └── ...
│
├── finetuned model outputs/               # Step 5e output
│   └── <model-name>/
│       └── ...
│
├── Human_eval_result/                     # Step 6b output
│   ├── Original Model/<model-name>/
│   │   ├── samples.jsonl
│   │   └── samples.jsonl_results.jsonl
│   ├── Pruned Model/<model-name>/
│   │   ├── humaneval_output_pruned_(<level>)/
│   │   │   ├── samples.jsonl
│   │   │   └── samples.jsonl_results.jsonl
│   │   └── ...
│   ├── Fine Tuned Model/<model-name>/
│   │   ├── humaneval_output_finetuned_(<level>)/
│   │   └── ...
│   ├── Random33 Pruned Model/<model-name>/
│   │   ├── humaneval_output_random33_(<level>)/
│   │   └── ...
│   ├── Random42 Pruned Model/<model-name>/
│   │   ├── humaneval_output_random42_(<level>)/
│   │   └── ...
│   └── Reverse Pruned Model/<model-name>/
│       ├── humaneval_output_reversed_(<level>)/
│       └── ...
│
├── semantic_similarity_result/            # Step 6c & 6d output
│   ├── bert_score_summary_<task>_<model>.csv
│   └── sbert_summary_<task>_<model>.csv
│
├── em_summary_<model-name>.csv            # Step 6a output
├── humaneval_summary_<model-name>.csv     # Step 6b output
└── trap_summary_<task>.csv                # Step 6e output
```

---

## Important Notes

- **Execution order matters:** Steps must generally be run in sequence (1 → 2 → 3 → 5 → 6), but you are not required to run all pruning types or all output scripts serially. You may run experiments in any order as long as the prerequisite outputs exist.
  - Example: You cannot generate pruned model outputs (Step 5) before pruning (Step 3).
  - Example: You cannot generate fine-tuned model outputs before fine-tuning or merging LoRA adapters with pruned models (Step 4). You can find lora merge code inside finetune pipeline (`lora_merge.py`). 
- **Partial pipeline support:** All evaluation and summary scripts (`extract_answers.py`, `bert_score_eval.py`, `sbert_score_eval.py`, `trap_count.py`, `human_eval_summary.py`) gracefully handle an incomplete pipeline:
  - If an **entire variant folder is absent** (e.g. you never ran reverse pruning), that variant's column is excluded from the CSV entirely.
  - If a **specific pruning level file is missing** within a present folder, that cell is written as `not found` in the CSV instead of 0.
- **HumanEval workflow:** Run `original_model_human_eval_script.py` once for the baseline, then `human_eval_script.py` for each variant type, then `human_eval_summary.py` to aggregate everything into a single CSV.
- **Pruning levels:** Pre-defined levels are provided for Qwen 1.5B and 7B models as comments inside each pruning script. If you use a different model, update the `pruning_levels` list.
- **Seed for random pruning:** The default seed is `42`. If you run random pruning with multiple seeds (e.g., `33` and `42`), each creates a separate output folder (`Random pruned model33/`, `Random pruned model42/`) and corresponding output folder (`random33 model outputs/`, `random42 model outputs/`).