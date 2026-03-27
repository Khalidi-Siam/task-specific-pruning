from load_model import load_model
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

type = ["math"]
# type = ["code"]
 #change when necessary
half_precision = True

VOLUME_PATH = "/root/datasets"
gc.collect()
torch.cuda.empty_cache()
chat_template = True 
# values = [5.71, 10, 15.71, 20, 25.71, 30, 35.71] # Qwen 1.5B
# values = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135] #qwen 7b
# values = [6.25, 10.71, 16.07, 21.43, 25, 31.25] # Mathstral
# values = [5.81, 10.46, 15.11, 20.93, 25.58, 30.23, 34.88] # deepseek 7B
values = [35.135]

output_dir = "Qwen2.5-Math-7B-Instruct"
for value in values:
    base_path = os.path.join(VOLUME_PATH, "Fine Tuned Model", output_dir)
    model_id = os.path.join(base_path, f"{output_dir}-{value}ft")
    
    for prompt_type in type:
        print(f"Generating pruned outputs for {prompt_type} prompts...")
        load_model(model_id, prompt_type, output_dir=output_dir, half_precision=half_precision, value=value, type="finetuned", chat_template=chat_template, in_batch=True)
        # add a line so that after each part execute the gpu memory freed up for next part. so that we don't OOM error
        gc.collect()
        torch.cuda.empty_cache()