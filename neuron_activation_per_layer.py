import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import numpy as np


def neuron_activation_per_layer(model_id, batch_size, half_precision=False, output_dir=None, target_type=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if half_precision:
        print("Loading using half precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, #use bf16
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
        ).to(device)

    model.eval()

    # Load training dataset
    print("Loading training prompts...")
    dataset_path = os.path.join("datasets", f"train_prompts_{target_type}_target.csv")
    df = pd.read_csv(dataset_path).dropna()
    prompts = df["prompt"].astype(str).tolist()
    labels = df["target_distractor"].astype(int).tolist()

    # Configuration
    num_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size  # Size of MLP intermediate layer (e.g., 5632 for TinyLlama)

    # Create directories for saving activations and metadata
    if output_dir:
        activation_dir = os.path.join(output_dir, "activation_logs")
        metadata_dir = os.path.join(output_dir, "activation_metadata")
    else:
        activation_dir = "activation_logs"
        metadata_dir = "activation_metadata"
    
    os.makedirs(activation_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Pre-tokenize prompts and check for truncation
    print("Tokenizing prompts...")
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # Pre-allocate storage for MLP intermediate activations
    all_activations = torch.zeros((len(prompts), num_layers, intermediate_size), dtype=torch.float32, device="cpu")

    # Define a hook to capture MLP intermediate activations (after silu(gate_proj(x)) * up_proj(x))
    def get_mlp_hook(layer_idx):
        def hook(module, input, output):
            # Extract intermediate activations from the MLP's forward pass
            # We need to compute silu(gate_proj(x)) * up_proj(x) manually
            x = input[0]  # Input to the MLP (hidden states)
            gate = module.gate_proj(x)  # gate_proj output
            up = module.up_proj(x)  # up_proj output
            intermediate = module.act_fn(gate) * up  # SwiGLU: silu(gate_proj(x)) * up_proj(x)
            abs_output = intermediate.abs()  # Take absolute value
            mask = attention_mask[i:i + batch_size].unsqueeze(-1).float()
            mean_abs_activations = (abs_output * mask).sum(dim=1) / mask.sum(dim=1)
            all_activations[i:i + batch_size, layer_idx] = mean_abs_activations.cpu()
        return hook

    # Register hooks on the MLP module for each transformer block
    hooks = []
    for layer_idx in range(num_layers):
        try:
            # Access the MLP component (assumes Llama-like architecture)
            mlp = model.model.layers[layer_idx].mlp
            if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "act_fn")):
                raise AttributeError("MLP does not have expected gate_proj, up_proj, or act_fn attributes.")
            hook = mlp.register_forward_hook(get_mlp_hook(layer_idx))
            hooks.append(hook)
        except AttributeError as e:
            print(f"Error accessing MLP for layer {layer_idx}: {e}")
            return

    # Process prompts in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing"):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]
            model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

    # Remove hooks after processing
    for hook in hooks:
        hook.remove()

    # Save metadata (including actual token lengths after truncation)
    print("Saving metadata...")
    metadata_df = pd.DataFrame({
        "prompt_id": np.arange(len(prompts)),
        "label": df["label"].tolist(),
        "target_distractor": labels,
        "prompt_text": prompts,
    })
    metadata_path = os.path.join(metadata_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    # Save activations
    print("Saving activations...")
    for layer_idx in range(num_layers):
        activation_path = os.path.join(activation_dir, f"layer_{layer_idx}.npy")
        np.save(
            activation_path,
            all_activations[:, layer_idx].numpy()
        )
        print(f"✅ Saved layer {layer_idx} activations")

    print("All data saved successfully!")