from load_model import load_model
import gc
import torch


category_labels = {
"conversational_ultrachat": 0,
"conversational_daily_dialog": 1,
"QnA": 2,
"code": 3,
"math": 4,
"poetry": 5,
"humor": 6
}

chat_template = True
type = ["math"]
# type = ["code"]
 #change when necessary
half_precision = True
output_dir = "Qwen2.5-Math-1.5B-Instruct"
model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
for prompt_type in type:
    print(f"Generating original outputs for {prompt_type} prompts...")
    load_model(model_id, prompt_type, output_dir=output_dir, half_precision=half_precision, type="original", chat_template=chat_template, in_batch=True)
    gc.collect()
    torch.cuda.empty_cache()