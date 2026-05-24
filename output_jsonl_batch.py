import csv
import torch
import os
import gc
import json
import time
from datetime import datetime

VOLUME_PATH = "/root/datasets"  # persistent volume
def output_jsonl_batch(model, tokenizer, prompt_type, max_length=1024, output_dir=None, type="original", value=None, batch_size=40, chat_template=False):
    """
    Generate answers for prompts in batches using the provided model and tokenizer.
    Ensures each prompt generates up to max_length tokens.
    Tracks per-prompt tokens and speed.
    Saves responses to JSONL and metrics (per-prompt + aggregated) to JSON.
    """

    prompt_csv_path = os.path.join(VOLUME_PATH, f"{prompt_type}_prompts.csv")

    # --- Setup output paths ---
    if output_dir:
        if type == "original":
            output_dir = os.path.join(VOLUME_PATH, "original model outputs", output_dir)
        elif type == "pruned":
            output_dir = os.path.join(VOLUME_PATH, "pruning model outputs", output_dir)
        elif type == "masked":
            output_dir = os.path.join(VOLUME_PATH, "masking model outputs", output_dir)
        elif type == "finetuned":
            output_dir = os.path.join(VOLUME_PATH, "finetuned model outputs", output_dir)
        elif type == "reversed":
            output_dir = os.path.join(VOLUME_PATH, "reversed model outputs", output_dir)
        else:
            output_dir = os.path.join(VOLUME_PATH, f"{type} model outputs", output_dir)

        os.makedirs(output_dir, exist_ok=True)
        if value:
            output_jsonl_path = os.path.join(output_dir, f"{prompt_type}_generated_outputs_{type}_({value}).jsonl")
            metrics_file_path = os.path.join(output_dir, f"{prompt_type}_generation_metrics_{type}_({value}).json")
        else:
            output_jsonl_path = os.path.join(output_dir, f"{prompt_type}_generated_outputs_{type}.jsonl")
            metrics_file_path = os.path.join(output_dir, f"{prompt_type}_generation_metrics_{type}.json")
    else:
        if value:
            output_jsonl_path = f"{prompt_type}_generated_outputs_{type}_({value}).jsonl"
            metrics_file_path = f"{prompt_type}_generation_metrics_{type}_({value}).json"
        else:
            output_jsonl_path = f"{prompt_type}_generated_outputs_{type}.jsonl"
            metrics_file_path = f"{prompt_type}_generation_metrics_{type}.json"

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
    prompt_metrics = []

    # --- Read existing progress ---
    processed_prompts = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    data = json.loads(line)
                    if "prompt_no" in data and isinstance(data["prompt_no"], int):
                        # Count as processed if it's not an error response
                        if not data.get("response", "").startswith("ERROR:"):
                            processed_prompts.add(data["prompt_no"])
                except Exception:
                    pass
        print(f"Resuming from {len(processed_prompts)} completed prompts.")

    # --- Init JSONL file if not exists ---
    if not os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            start_json = {
                "prompt_no": "STARTING",
                "prompt": f"Generation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "response": "Generation started"
            }
            outfile.write(json.dumps(start_json, ensure_ascii=False) + "\n")

    # --- Read all prompts ---
    with open(prompt_csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        all_rows = [(row_idx, row["prompt"].strip())
                    for row_idx, row in enumerate(reader, start=1)
                    if "prompt" in row and row["prompt"].strip()]

    # --- Process in batches ---
    for batch_start in range(0, len(all_rows), batch_size):
        batch_rows = all_rows[batch_start:batch_start + batch_size]
        pending_rows = [r for r in batch_rows if r[0] not in processed_prompts]

        if not pending_rows:
            print(f"Skipping completed batch {batch_start // batch_size + 1}")
            continue

        row_indices = [r[0] for r in pending_rows]
        prompts = [r[1] for r in pending_rows]

        print(f"Processing batch {batch_start // batch_size + 1} ({len(prompts)} rows pending out of {len(batch_rows)})...")

        try:
            # Prepare chat messages
            if chat_template:
                if("Qwen" in output_dir and "math" in prompt_type):
                    messages = [
                        [
                            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                            {"role": "user", "content": prompt}
                        ]
                        for prompt in prompts
                    ]
                elif("Qwen" in output_dir and "code" in prompt_type):
                        messages = [
                            [
                                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ]
                            for prompt in prompts
                        ]
                elif(("deepseek" in output_dir or "Mathstral" in output_dir) and "math" in prompt_type):
                    messages = [
                        [
                            {
                                "role": "user",
                                "content": prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                            }
                        ]
                    for prompt in prompts
                    ]
                else:
                    print("***Using normal ***")
                    messages = [[{"role": "user", "content": p}] for p in prompts]
                    
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                )
            else:
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
            # Move tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # --- Batch generation ---
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            batch_time = end_time - start_time
            total_generation_time += batch_time

            # --- Decode each prompt and compute actual tokens_generated ---
            for i, row_id in enumerate(row_indices):
                input_len = inputs["input_ids"].shape[1]
                # Only take the part that is after input_ids
                generated_ids = outputs[i][input_len:]
                # Actual tokens generated ignoring padding used for batch alignment
                real_generated_ids = generated_ids[generated_ids != tokenizer.pad_token_id]
                answer = tokenizer.decode(real_generated_ids, skip_special_tokens=True).strip()
                gen_len = len(real_generated_ids)

                total_tokens_generated += gen_len
                total_prompts += 1
                tokens_per_second = gen_len / batch_time if batch_time > 0 else 0.0

                # Save response
                json_output = {"prompt_no": row_id, "prompt": prompts[i], "response": answer}
                with open(output_jsonl_path, "a", encoding="utf-8") as outfile:
                    outfile.write(json.dumps(json_output, ensure_ascii=False) + "\n")

                # Save per-prompt metrics
                metrics = {
                    "prompt_no": row_id,
                    "tokens_generated": gen_len,
                    "generation_time_seconds": batch_time,
                    "tokens_per_second": tokens_per_second,
                }
                prompt_metrics.append(metrics)

            print(f"✅ Batch completed ({len(prompts)} prompts)")

            # --- Intermediate metrics update ---
            avg_tokens_per_second = total_tokens_generated / total_generation_time if total_generation_time > 0 else 0.0
            
            interim_metrics = {
                "total_prompts_processed": total_prompts,
                "total_tokens_generated": total_tokens_generated,
                "total_generation_time_seconds": total_generation_time,
                "overall_tokens_per_second": avg_tokens_per_second,
                "per_prompt_metrics": prompt_metrics,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Merge with existing metrics if resuming
            if os.path.exists(metrics_file_path):
                try:
                    with open(metrics_file_path, "r", encoding="utf-8") as f:
                        old_metrics = json.load(f)
                    
                    merged_metrics = {
                        "total_prompts_processed": interim_metrics["total_prompts_processed"] + old_metrics.get("total_prompts_processed", 0),
                        "total_tokens_generated": interim_metrics["total_tokens_generated"] + old_metrics.get("total_tokens_generated", 0),
                        "total_generation_time_seconds": interim_metrics["total_generation_time_seconds"] + old_metrics.get("total_generation_time_seconds", 0),
                    }
                    merged_metrics["overall_tokens_per_second"] = (
                        merged_metrics["total_tokens_generated"] / merged_metrics["total_generation_time_seconds"] 
                        if merged_metrics["total_generation_time_seconds"] > 0 else 0.0
                    )
                    merged_metrics["per_prompt_metrics"] = old_metrics.get("per_prompt_metrics", []) + interim_metrics["per_prompt_metrics"]
                    merged_metrics["timestamp"] = interim_metrics["timestamp"]
                    
                    with open(metrics_file_path, "w", encoding="utf-8") as metrics_file:
                        json.dump(merged_metrics, metrics_file, indent=2, ensure_ascii=False)
                except Exception:
                    with open(metrics_file_path, "w", encoding="utf-8") as metrics_file:
                        json.dump(interim_metrics, metrics_file, indent=2, ensure_ascii=False)
            else:
                with open(metrics_file_path, "w", encoding="utf-8") as metrics_file:
                    json.dump(interim_metrics, metrics_file, indent=2, ensure_ascii=False)

            # Clear temporary prompt metrics to prevent duplicating them in the file on the next batch loop
            prompt_metrics = []

            # Cleanup
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"❌ Error in batch starting at prompt {row_indices[0]}: {e}")
            for row_id, prompt in zip(row_indices, prompts):
                error_json = {"prompt_no": row_id, "prompt": prompt, "response": f"ERROR: {str(e)}"}
                with open(output_jsonl_path, "a", encoding="utf-8") as outfile:
                    outfile.write(json.dumps(error_json, ensure_ascii=False) + "\n")

        # if total_prompts >= 50:
        #     break

    # --- Final aggregated metrics ---
    if total_prompts > 0:
        print(f"\n🎉 Completed {total_prompts} new prompts in this run.")
        print(f"Metrics continuously updated to {metrics_file_path}")
    else:
        print(f"\n🎉 All prompts were already completed! Skipped generation.")
