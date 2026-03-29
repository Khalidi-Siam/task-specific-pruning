import csv
import torch
import os
import gc
import json
import time
from datetime import datetime


def output_jsonl(model, tokenizer, prompt_type, max_length=1024, output_dir=None, model_type="original", value=None, chat_template=False):
    """
    Generate answers for prompts one by one using the provided model and tokenizer.
    Tracks per-prompt VRAM usage, tokens, and speed with accurate peak measurement.
    Saves responses to JSONL and metrics (per-prompt + aggregated) to JSON.
    """

    prompt_csv_path = os.path.join("datasets", f"{prompt_type}_prompts.csv")

    # --- Setup output paths ---
    if output_dir:
        output_dir = os.path.join(f"{model_type} model outputs", output_dir)

        os.makedirs(output_dir, exist_ok=True)
        if value:
            output_jsonl_path = os.path.join(output_dir, f"{prompt_type}_generated_outputs_{model_type}_({value}).jsonl")
            metrics_file_path = os.path.join(output_dir, f"{prompt_type}_generation_metrics_{model_type}_({value}).json")
        else:
            output_jsonl_path = os.path.join(output_dir, f"{prompt_type}_generated_outputs_{model_type}.jsonl")
            metrics_file_path = os.path.join(output_dir, f"{prompt_type}_generation_metrics_{model_type}.json")
    else:
        if value:
            output_jsonl_path = f"{prompt_type}_generated_outputs_{model_type}_({value}).jsonl"
            metrics_file_path = f"{prompt_type}_generation_metrics_{model_type}_({value}).json"
        else:
            output_jsonl_path = f"{prompt_type}_generated_outputs_{model_type}.jsonl"
            metrics_file_path = f"{prompt_type}_generation_metrics_{model_type}.json"

    print(f"Starting output generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = next(model.parameters()).device
    print(f"Using device: {device}")
    print(f"Reading prompts from: {prompt_csv_path}")
    print(f"Writing outputs to: {output_jsonl_path}")
    print(f"Metrics will be saved to: {metrics_file_path}")

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("[INFO] Tokenizer had no pad token. Using eos_token as pad_token.")

    tokenizer.padding_side = "left"

    if not os.path.exists(prompt_csv_path):
        raise FileNotFoundError(f"CSV file not found: {prompt_csv_path}")

    # --- Init counters ---
    total_prompts = 0
    total_tokens_generated = 0
    total_generation_time = 0.0
    peak_vram_usage = 0.0
    total_vram_usage = 0.0
    prompt_metrics = []

    # --- Init JSONL file ---
    with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
        start_json = {
            "prompt_no": "STARTING",
            "prompt": f"Generation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "response": "Generation started"
        }
        outfile.write(json.dumps(start_json, ensure_ascii=False) + "\n")

    # --- Process prompts one by one ---
    with open(prompt_csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row_idx, row in enumerate(reader, start=1):
            if "prompt" not in row or not row["prompt"].strip():
                print(f"Skipping empty/malformed row {row_idx}")
                continue

            prompt = row["prompt"].strip()
            print(f"Processing prompt {row_idx}...")

            try:
                # Prepare input
                if chat_template:
                    # print("****Using chat template****")
                    if("Qwen" in output_dir and "math" in prompt_type):
                        messages = [
                                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                                    {"role": "user", "content": prompt}
                                ]
                    elif("Qwen" in output_dir and "code" in prompt_type):
                        messages = [
                            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    elif(("deepseek" in output_dir or "Mathstral" in output_dir) and "math" in prompt_type):
                        messages = [
                            {
                                "role": "user",
                                "content": prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                            }
                        ]   
                    else:
                        messages = [[{"role": "user", "content": prompt}]]

                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(device)
                else:
                    # print("****Not using chat template****")
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)

                # Reset VRAM peak stats
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    vram_before_alloc = torch.cuda.memory_allocated() / 1024**3
                    vram_before_reserved = torch.cuda.memory_reserved() / 1024**3
                else:
                    vram_before_alloc = vram_before_reserved = 0.0

                # Generate
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                torch.cuda.synchronize()
                end_time = time.time()
                generation_time = end_time - start_time
                total_generation_time += generation_time

                # After generation VRAM
                if torch.cuda.is_available():
                    vram_after_alloc = torch.cuda.memory_allocated() / 1024**3
                    vram_after_reserved = torch.cuda.memory_reserved() / 1024**3
                    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
                    peak_vram_usage = max(peak_vram_usage, peak_alloc, peak_reserved)
                else:
                    vram_after_alloc = vram_after_reserved = peak_alloc = peak_reserved = 0.0

                # Token count
                gen_len = outputs[0].shape[-1] - inputs["input_ids"].shape[-1]
                total_tokens_generated += gen_len
                tokens_per_second = gen_len / generation_time if generation_time > 0 else 0.0

                # VRAM usage (use peak for accuracy)
                total_vram_usage += peak_alloc
                total_prompts += 1

                # Decode
                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Save response
                json_output = {"prompt_no": row_idx, "prompt": prompt, "response": answer}
                with open(output_jsonl_path, "a", encoding="utf-8") as outfile:
                    outfile.write(json.dumps(json_output, ensure_ascii=False) + "\n")

                # Save per-prompt metrics
                metrics = {
                    "prompt_no": row_idx,
                    "tokens_generated": gen_len,
                    "generation_time_seconds": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "vram_allocated_gb_before": vram_before_alloc,
                    "vram_allocated_gb_after": vram_after_alloc,
                    "vram_reserved_gb_before": vram_before_reserved,
                    "vram_reserved_gb_after": vram_after_reserved,
                    "vram_peak_allocated_gb": peak_alloc,
                    "vram_peak_reserved_gb": peak_reserved,
                }
                prompt_metrics.append(metrics)

                print(f"✅ Prompt {row_idx} completed ({gen_len} tokens in {generation_time:.2f}s | Peak VRAM {peak_alloc:.2f} GB)")

                # Cleanup
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"❌ Error on prompt {row_idx}: {e}")
                error_json = {"prompt_no": row_idx, "prompt": prompt, "response": f"ERROR: {str(e)}"}
                with open(output_jsonl_path, "a", encoding="utf-8") as outfile:
                    outfile.write(json.dumps(error_json, ensure_ascii=False) + "\n")

            # if total_prompts >= 50:
            #     break

    # --- Final aggregated metrics ---
    avg_tokens_per_second = total_tokens_generated / total_generation_time if total_generation_time > 0 else 0.0
    avg_vram_usage = total_vram_usage / total_prompts if total_prompts > 0 else 0.0

    final_metrics = {
        "total_prompts_processed": total_prompts,
        "total_tokens_generated": total_tokens_generated,
        "total_generation_time_seconds": total_generation_time,
        "overall_tokens_per_second": avg_tokens_per_second,
        "peak_vram_usage_gb": peak_vram_usage,
        "average_vram_usage_per_prompt_gb": avg_vram_usage,
        "per_prompt_metrics": prompt_metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metrics_file_path, "w", encoding="utf-8") as metrics_file:
        json.dump(final_metrics, metrics_file, indent=2, ensure_ascii=False)

    print(f"\n🎉 Completed {total_prompts} prompts.")
    print(f"Metrics saved to {metrics_file_path}")
    print(f"Peak VRAM usage across all prompts: {peak_vram_usage:.2f} GB")
    print(f"Average peak VRAM per prompt: {avg_vram_usage:.2f} GB")
    print(f"Overall generation speed: {avg_tokens_per_second:.2f} tokens/sec")
