import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc



def prune_mlp_layer(mlp, pruned_indices):
    """
    Physically prune neurons from a Hugging Face LlamaMLP block,
    while keeping the optimized MLP class (with fused SiLU kernels).
    """
    keep_indices = [i for i in range(mlp.gate_proj.out_features) if i not in pruned_indices]
    keep_indices = torch.tensor(keep_indices, dtype=torch.long)

    # Clone old layers
    old_gate = mlp.gate_proj
    old_up = mlp.up_proj
    old_down = mlp.down_proj

    # ---- gate_proj (input → hidden) ----
    new_gate = torch.nn.Linear(old_gate.in_features, len(keep_indices), bias=old_gate.bias is not None)
    new_gate.weight.data = old_gate.weight.data[keep_indices, :].clone()
    if old_gate.bias is not None:
        new_gate.bias.data = old_gate.bias.data[keep_indices].clone()

    # ---- up_proj (input → hidden) ----
    new_up = torch.nn.Linear(old_up.in_features, len(keep_indices), bias=old_up.bias is not None)
    new_up.weight.data = old_up.weight.data[keep_indices, :].clone()
    if old_up.bias is not None:
        new_up.bias.data = old_up.bias.data[keep_indices].clone()

    # ---- down_proj (hidden → output) ----
    new_down = torch.nn.Linear(len(keep_indices), old_down.out_features, bias=old_down.bias is not None)
    new_down.weight.data = old_down.weight.data[:, keep_indices].clone()
    if old_down.bias is not None:
        new_down.bias.data = old_down.bias.data.clone()

    # ✅ Replace weights back into the same MLP class
    mlp.gate_proj = new_gate.to(old_gate.weight.dtype).to(old_gate.weight.device)
    mlp.up_proj = new_up.to(old_up.weight.dtype).to(old_up.weight.device)
    mlp.down_proj = new_down.to(old_down.weight.dtype).to(old_down.weight.device)

    return mlp



def physical_pruning(model, percentile=10.0):
    """
    Physically prune neurons uniformly across all layers by percentile.
    This version prunes RANDOM neurons instead of selectivity-based.
    """
    num_layers = model.config.num_hidden_layers

    total_pruned = 0
    total_neurons = 0

    for layer_idx in range(num_layers):
        block = model.model.layers[layer_idx]
        mlp = block.mlp

        # Get total number of neurons in this layer
        total_layer_neurons = mlp.gate_proj.out_features

        # Calculate number of neurons to prune based on percentile
        num_to_prune = int(np.ceil(total_layer_neurons * percentile / 100.0))

        # Randomly select neurons to prune
        all_indices = np.arange(total_layer_neurons)
        np.random.shuffle(all_indices)
        pruned_indices = all_indices[:num_to_prune]

        mlp = prune_mlp_layer(mlp, pruned_indices)

        total_pruned += len(pruned_indices)
        total_neurons += total_layer_neurons
        print(f"✅ Layer {layer_idx}: pruned {len(pruned_indices)}/{total_layer_neurons} neurons (random)")

    print(f"\n🎯 TOTAL: pruned {total_pruned}/{total_neurons} neurons "
          f"({100*total_pruned/total_neurons:.2f}%)")

    # Update config (all layers now share smaller intermediate_size)
    new_size = model.model.layers[0].mlp.gate_proj.out_features
    model.config.intermediate_size = new_size

    return model


def save_pruned_model(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Convert to BF16 before saving
    model = model.to(torch.bfloat16)

    # ✅ Save with safetensors, sharded into ≤10GB chunks
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="10GB", torch_dtype=torch.bfloat16)
    tokenizer.save_pretrained(save_dir)

    print(f"💾 Saved pruned model to {save_dir} in BF16 with safetensors")


def helper(model_name, output_dir, percentile, seed=42):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    num_layers = model.config.num_hidden_layers

    print(f"\n🔪 Pruning with percentile {percentile} (RANDOM neurons)...")

    # Set random seed for reproducibility (optional)
    np.random.seed(seed)

    pruned_model = physical_pruning(model, percentile=percentile)
    # Updated save_dir to use Google Drive path
    # save_pruned_model(pruned_model, tokenizer, save_dir=f"C:\\T2430447\\Random pruning\\Pruned_Models\\{output_dir}\\{output_dir}-pruned-{percentile}p")
    save_pruned_model(pruned_model, tokenizer, save_dir=os.path.join("Random pruned model", output_dir, f"{output_dir}-pruned-{percentile}p"))


if __name__ == "__main__":
    # models = ["Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen2.5-Math-7B-Instruct", "deepseek-ai/deepseek-math-7b-instruct", "mistralai/Mathstral-7B-v0.1"]
    output_dir = "Qwen2.5-Math-7B-Instruct"
    # os.makedirs(f"C:\\T2430447\\Random pruning\\Pruned_Models\\{output_dir}", exist_ok=True)
    seed = 42  # Set a fixed seed for reproducibility
    os.makedirs(os.path.join(f"Random pruned model{seed}", output_dir), exist_ok=True)
    # pruning_levels = [5.71, 10, 15.71, 20, 25.71, 30, 35.71]
    pruning_levels =[5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135]  #qwen 7b

    for level in pruning_levels:
        helper(f"Qwen/{output_dir}", output_dir, percentile=level, seed=seed)

