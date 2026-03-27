from load_model import load_model
import gc
import torch
import os
import json


category_labels = {
    "conversational_ultrachat": 0,
    "conversational_daily_dialog": 1,
    "QnA": 2,
    "code": 3,
    "math": 4,
    "poetry": 5,
    "humor": 6
}

type = ["conversational"]
# type = ["code"]
 #change when necessary
half_precision = True
chat_template = True

VOLUME_PATH = "/root/datasets"

# ✅ checkpoint file (separate for reversed to avoid conflict)
CHECKPOINT_FILE = os.path.join(VOLUME_PATH, f"checkpoint_reversed_pruning_{type[0]}.json")

# values

values = [5.71, 10, 15.71, 20, 25.71, 30, 35.71]

output_dir = "Qwen2.5-Coder-1.5B-Instruct"


# ----------------------------
# ✅ Load checkpoint
# ----------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f))
    return set()


# ----------------------------
# ✅ Save checkpoint
# ----------------------------
def save_checkpoint(completed):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(completed), f)


completed_values = load_checkpoint()
print(f"📌 Already completed: {completed_values}")

gc.collect()
torch.cuda.empty_cache()


# ----------------------------
# 🚀 Main loop
# ----------------------------
for value in values:

    if value in completed_values:
        print(f"⏭️ Skipping value {value} (already completed)")
        continue

    print(f"\n🚀 Processing value: {value}")

    try:
        base_path = os.path.join(VOLUME_PATH, "Reverse Pruned Model", output_dir)
        model_id = os.path.join(base_path, f"{output_dir}-pruned-{value}p")

        for prompt_type in type:
            print(f"Generating pruned outputs for {prompt_type} prompts...")

            load_model(
                model_id,
                prompt_type,
                output_dir=output_dir,
                half_precision=half_precision,
                value=value,
                type="reversed",
                chat_template=chat_template,
                in_batch=True
            )

            gc.collect()
            torch.cuda.empty_cache()

        # ✅ mark complete only if success
        completed_values.add(value)
        save_checkpoint(completed_values)

        print(f"✅ Completed value: {value}")

    except Exception as e:
        print(f"❌ Error at value {value}: {e}")
        print("⛔ Will retry this value in next run")
        break