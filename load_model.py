import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from benchmark import main_func
from output_jsonl import output_jsonl
from output_jsonl_batch import output_jsonl_batch


def load_model(model_name, prompt_type, output_dir=None, half_precision=False, value=None, model_type="original", batch_size=16, chat_template=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if(half_precision):
        print("Using half precision for model loading.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  
            low_cpu_mem_usage=True,  # Reduce memory usage during loading
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        ).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on: {device}")
    model.eval()  # Set the model to evaluation mode

    # -------------------------
    # Output generation
    # -------------------------
    with torch.no_grad():
        if batch_size > 1:
            output_jsonl_batch(model, tokenizer, prompt_type, output_dir=output_dir, model_type=model_type, value=value, chat_template=chat_template)
        else:
            output_jsonl(model, tokenizer, prompt_type, output_dir=output_dir, model_type=model_type, value=value, chat_template=chat_template)
