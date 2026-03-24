import pandas as pd
import numpy as np
import os


def compute_selectivity_scores(output_dir=None):
    if output_dir:
        activation_dir = os.path.join(output_dir, "activation_logs")
        metadata_path = os.path.join(output_dir, "activation_metadata", "metadata.csv")
        selectivity_output_dir = os.path.join(output_dir, "selectivity_scores")
    else:
        activation_dir = "activation_logs"
        metadata_path = "activation_metadata/metadata.csv"
        selectivity_output_dir = "selectivity_scores"
    
    os.makedirs(selectivity_output_dir, exist_ok=True)

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    target_distractor_labels = metadata_df["target_distractor"].values

    for file in os.listdir(activation_dir):
        if not file.endswith(".npy"):
            continue

        layer = int(file.split("_")[1].split(".")[0])
        activations = np.load(os.path.join(activation_dir, file))  # shape: (num_prompts, intermediate_size)

        # Use target_distractor column: 0 = distractor, 1 = target
        activations_target = activations[target_distractor_labels == 1]
        activations_distractor = activations[target_distractor_labels == 0]

        # Compute directional selectivity per neuron
        mean_target = activations_target.mean(axis=0)
        mean_distractor = activations_distractor.mean(axis=0)
        std_all = activations.std(axis=0) + 1e-6  # prevent divide-by-zero

        # Positive score = target-active, Negative score = distractor-active
        selectivity = (mean_target - mean_distractor) / std_all

        # Save directional selectivity scores
        output_file = os.path.join(selectivity_output_dir, f"layer_{layer}_selectivity.csv")
        pd.DataFrame({
            "layer": layer,
            "neuron_id": np.arange(len(selectivity)),
            "selectivity_score": selectivity
        }).to_csv(output_file, index=False)

        print(f"✅ Saved directional selectivity scores for Layer {layer}")
