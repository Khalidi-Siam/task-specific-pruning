import json
import csv
import re
import os

# Path to the JSONL file
# jsonl_file_path = os.path.join(os.path.dirname(__file__), '..', 'pruning model outputs', 'Qwen2.5-Math-7B-Instruct', 'nctb_generated_outputs_pruned_(25.58).jsonl')
# jsonl_file_path = os.path.join(os.path.dirname(__file__), '..', 'original model outputs', 'deepseek-math-7b-instruct', 'math_generated_outputs_original.jsonl')
jsonl_file_path = os.path.join(os.path.dirname(__file__), '..', 'finetuned model outputs', 'deepseek-math-7b-instruct', 'math_generated_outputs_finetuned_(25.58).jsonl')
# jsonl_file_path = os.path.join(os.path.dirname(__file__), '..', 'masking model outputs', 'deepseek-math-7b-instruct', 'math_generated_outputs_masked_(-0.8).jsonl')


# Output CSV file path
csv_file_path = os.path.join(os.path.dirname(__file__), 'extracted_answers.csv')

# Function to extract answer from response
def extract_answer(response):
    # First try to find complete \boxed{...} pattern
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1)
    
    # If complete pattern not found, try to find incomplete \boxed{ pattern
    # This handles cases like $\\boxed{1500 where closing } is missing
    match = re.search(r'\\boxed\{([^}$\s]+)', response)
    if match:
        return match.group(1)
    
    return "not found in this format"

# Read JSONL and write to CSV
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['prompt_no', 'answer'])  # Header
    
    for line in jsonl_file:
        data = json.loads(line.strip())
        prompt_no = data.get('prompt_no')
        if prompt_no == "STARTING":
            continue  # Skip metadata
        response = data.get('response', '')
        answer = extract_answer(response)
        writer.writerow([prompt_no, answer])

print(f"Extraction complete. CSV saved to: {csv_file_path}")
