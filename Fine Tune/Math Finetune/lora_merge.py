from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import gc


def lora_merge(value, model_name):
    base_model_name_or_path = os.path.join("Pruned Model", model_name, f"{model_name}-pruned-{value}p")
    lora_model_path = os.path.join("adapters", model_name, f"lora_out_{model_name}-pruned-{value}p", "best_em")
    """
    best_em: best adapter based on exact match(EM)
    best: best adapter based on validation loss
    choose anyone based on your preference. currently we are using best_em for evaluation and merging.
    """

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model in bfloat16 on the determined device
    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch.bfloat16)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    # Wrap the base model with LoRA weights (inherits bfloat16 dtype)
    lora_model = PeftModel.from_pretrained(model, lora_model_path, torch_dtype=torch.bfloat16)
    lora_model = lora_model.to(device)
    # Merge the LoRA weights permanently
    lora_model = lora_model.merge_and_unload()

    # os.makedirs(os.path.join("Fine Tuned Model"), exist_ok=True)
    merged_model_path = os.path.join("Fine Tuned Model", model_name, f"{model_name}-{value}ft")
    lora_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model saved to {merged_model_path}")

# Clean up to free memory
    del model
    del lora_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # values = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135] #qwen 7b
    # values = [6.25, 10.71, 16.07, 21.43, 25, 31.25] # Mathstral
    # values = [5.71, 10, 15.71, 20, 25.71, 30, 35.71] # Qwen 1.5B
    # values = [5.81, 10.46, 15.11, 20.93, 25.58, 30.23, 34.88] # deepseek 7B
    values = [30.405, 35.135]

    model_name = "Qwen2.5-Math-7B-Instruct"
    for value in values:
        lora_merge(value, model_name)