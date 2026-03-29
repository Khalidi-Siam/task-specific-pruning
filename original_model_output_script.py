from load_model import load_model
import gc
import torch


chat_template = True
half_precision = True
batch_size = 16

category_labels = {
"conversational": 1,
"qna": 2,
"code": 3,
"math": 4,
}
task_types = ["math"] # math, code, conversational, qna

output_dir = "Qwen2.5-Math-1.5B-Instruct"
model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"

for task_type in task_types:
    print(f"Generating original outputs for {task_type} prompts...")
    load_model(model_id, task_type, output_dir=output_dir, half_precision=half_precision, type="original", chat_template=chat_template, batch_size=batch_size)
    gc.collect()
    torch.cuda.empty_cache()