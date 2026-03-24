import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

pruned_model_path = os.path.join("Pruned Model", "Qwen2.5-Math-1.5B-Instruct-pruned-35.71p")
model = AutoModelForCausalLM.from_pretrained(pruned_model_path, torch_dtype=torch.bfloat16)
# original_model_path = os.path.join("Pruned Model", "deepseek-math-7b-instruct-pruned-40.7p")
# model.eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

total_params_pruned = sum(p.numel() for p in model.parameters())



original_model = "Qwen2.5-Math-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(original_model, torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")

model = model.to("cuda" if torch.cuda.is_available() else "cpu")
total_params_original = sum(p.numel() for p in model.parameters())




print(f"Total parameters in the original model: {total_params_original}")
print(f"Total parameters in the pruned model: {total_params_pruned}")

change = ((total_params_original - total_params_pruned) / total_params_original) * 100
print(f"Percentage reduction in parameters due to pruning: {change:.2f}%")