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


gc.collect()
torch.cuda.empty_cache()
chat_template = True
values = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135]
# values = [20.27]
# values = [35.135]
# values = [5.71, 10, 15.71, 20, 25.71, 30, 35.71]
# values = [6.25, 10.71, 16.07, 21.43, 25, 31.25] #matstral

output_dir = "Qwen2.5-Math-7B-Instruct"
for value in values:
    base_path = os.path.join("Random pruned model", output_dir)
    model_id = os.path.join(base_path, f"{output_dir}-pruned-{value}p")
    for prompt_type in type:
        print(f"Generating pruned outputs for {prompt_type} prompts...")
        load_model(model_id, prompt_type, output_dir=output_dir, half_precision=half_precision, value=value, type="random", chat_template=chat_template)
        gc.collect()
        torch.cuda.empty_cache()