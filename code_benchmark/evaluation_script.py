print("Script started!")
import os
import sys
import gc
import torch

print(f"Torch OK. CUDA available: {torch.cuda.is_available()}")


print("Importing human_eval_anal2...")
from human_eval import human_eval_deterministic

print("All imports complete!\n")

values = [5.405, 10.135, 20.27, 25, 30.405, 35.135]

output_dir = "Qwen2.5-Coder-7B-Instruct"
for value in values:
    print(f"\n{'='*60}")
    print(f"Starting evaluation for {value}% pruning")
    print(f"{'='*60}")
    
    base_path = os.path.join("Fine Tuned Model", output_dir)
    model_id = os.path.join(base_path, f"{output_dir}-{value}ft")
    print(f"Model path: {model_id}")
    

    human_eval_deterministic(
        model_path=model_id,
        seed=42,
        batch_size=16,
        max_new_tokens=512,
        output_dir=os.path.join("Target_code", "test", f"{output_dir}-{value}ft")
    )
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared\n")