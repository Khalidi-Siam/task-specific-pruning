from neuron_activation_per_layer import neuron_activation_per_layer
from compute_selectivity_score import compute_selectivity_scores
import gc
import torch
import os

category_labels = {
    "conversational_ultrachat": 0,
    "conversational_daily_dialog": 1,
    "QnA": 2,
    "code": 3,
    "math": 4,
    "poetry": 5,
    "humor": 6
}

dataset_name = {
    1 : "suriya7/everyday-Conversational-cleaned",
    2 : "rajpurkar/squad",
    3 : "extracted_sample_1k",
    4 : "gsm8k" 
}


def _infer_target_type(target_categories):
    if 4 in target_categories:
        return "math"
    if 3 in target_categories:
        return "code"
    return "custom"


def run_selective_pruning_experiment(
    model_id,
    batch_size,
    output_dir,
    target_categories,
    distractor_categories,
    half_precision=True,
    target_type=None,
):
    if target_type is None:
        target_type = _infer_target_type(target_categories)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Make the config file
    config_file_path = os.path.join(output_dir, "config.txt")
    with open(config_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("\t\t****Configuration****\n\n")
        log_file.write(f"model_name: {model_id}\n")
        log_file.write(f"batch_size: {batch_size}\n")
        log_file.write(f"half_precision: {half_precision}\n")
        log_file.write(f"target_type: {target_type}\n")

        log_file.write("\n\t\t****Dataset categories****\n")
        log_file.write("\n***Target categories***\n")
        for key, value in category_labels.items():
            if value in target_categories:
                log_file.write(f"{value}: {key}\n")

        log_file.write("\n\n***Distractor categories***\n")
        for key, value in category_labels.items():
            if value in distractor_categories:
                log_file.write(f"{value}: {key}\n")

        log_file.write("\n\n***Dataset names***\n")
        for cat in target_categories + distractor_categories:
            log_file.write(f"{cat}: {dataset_name[cat]}\n")

    print("****Step 3: Capture neuron activation****")
    neuron_activation_per_layer(
        model_id,
        batch_size,
        half_precision=half_precision,
        output_dir=output_dir,
        target_type=target_type,
    )

    print("****Step 4: Compute Selectivity Scores****")
    compute_selectivity_scores(output_dir=output_dir)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example run; update these values as needed.
    # task label 0, 5, 6 are not used in the research
    run_selective_pruning_experiment(
        model_id="Qwen/Qwen2.5-Math-7B-Instruct", # choose any base model
        batch_size=32, # adjust based on your GPU memory. i.e 32 for 7b and 64 for 1.5b models
        output_dir="model activation/Qwen2.5-Math-7B-Instruct",
        target_categories=[4], # if math model target category is 4, if code model target category is 3
        distractor_categories=[1, 2],
        half_precision=True,
    )



