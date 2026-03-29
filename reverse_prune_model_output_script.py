from load_model import load_model
import gc
import torch
import os


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

# values = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135] # qwen 7b values(both code and math)
values = [5.71, 10, 15.71, 20, 25.71, 30, 35.71] # qweb 1.5b values (both code and math)

output_dir = "Qwen2.5-Math-1.5B-Instruct"
for value in values:
    base_path = os.path.join("Reverse Pruned Model", output_dir)
    model_id = os.path.join(base_path, f"{output_dir}-pruned-{value}p")
    for task_type in task_types:
        print(f"Generating outputs for {task_type} prompts...")
        load_model(model_id, task_type=task_type, output_dir=output_dir, half_precision=half_precision, value=value, model_type="reversed", chat_template=chat_template, batch_size=batch_size)

        gc.collect()
        torch.cuda.empty_cache()