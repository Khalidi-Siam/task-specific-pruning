import json
import csv
import re
import os


def extract_answer(response):
    """Extract answer from response text"""
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


def process_combined_data(value, model_name, output_dir, result_type):
    """
    Process both JSONL (for answers) and JSON (for trap detection) files
    and create a combined CSV output
    """
    # File paths
    base_dir = os.path.dirname(__file__)
    # model_name = 'Qwen2.5-Math-1.5B-Instruct'
    
    jsonl_file_path = os.path.join(
        base_dir, 
        output_dir, 
        model_name, 
        f'math_generated_outputs_{result_type}_({value}).jsonl'
    )
    
    json_file_path = os.path.join(
        base_dir,
        output_dir,
        model_name,
        f'math_generation_metrics_{result_type}_({value}).json'
    )
    
    # Output CSV file path
    csv_file_path = f'combined_output_({value}).csv'
    
    # Token threshold for trap detection
    TOKEN_THRESHOLD = 1024
    
    # Step 1: Load trap information from JSON file
    trap_dict = {}
    trap_count = 0
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        per_prompt_metrics = data.get('per_prompt_metrics', [])
        
        for metric in per_prompt_metrics:
            if 'prompt_no' in metric and 'tokens_generated' in metric:
                prompt_no = metric['prompt_no']
                tokens_generated = metric['tokens_generated']
                # Mark as trap if tokens_generated >= threshold
                if tokens_generated >= TOKEN_THRESHOLD:
                    trap_dict[prompt_no] = 'trap'
                    trap_count += 1
                else:
                    trap_dict[prompt_no] = ''
    except FileNotFoundError:
        print(f"Warning: JSON file not found: {json_file_path}")
        print("Continuing without trap detection...")
    
    # Step 2: Read JSONL file and extract answers, then combine with trap info
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, \
             open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            
            writer = csv.writer(csv_file)
            writer.writerow(['prompt_no', 'Answer', 'comment'])  # Header
            
            for line in jsonl_file:
                data = json.loads(line.strip())
                prompt_no = data.get('prompt_no')
                
                if prompt_no == "STARTING":
                    continue  # Skip metadata
                
                response = data.get('response', '')
                answer = extract_answer(response)
                
                # Get trap comment from trap_dict
                comment = trap_dict.get(prompt_no, '')
                
                writer.writerow([prompt_no, answer, comment])
        
        print(f"Value: {value}")
        print(f"Total trapped responses (tokens_generated >= {TOKEN_THRESHOLD}): {trap_count}")
        print(f"Combined CSV file generated: {csv_file_path}")
        print("-" * 60)
        
        # Append to trap log file
        with open('trap_log.txt', 'a', encoding='utf-8') as log_file:
            log_file.write(f"{json_file_path}: {trap_count}\n")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error processing value {value}: {e}")


def main():
    """Main function to process all values"""
    # Define the values to process
    model_name = 'Qwen2.5-Math-1.5B-Instruct'
    output_dir = 'finetuned model outputs(best)'
    result_type = 'finetuned'
    values = [5.71, 10, 15.71, 20, 25.71, 30, 35.71] # qwen1.5B
    # values = [6.25, 10.71, 16.07, 21.43, 25, 31.25] # Mathstral 7B
    # values = [5.405, 10.135, 15.54, 20.27, 25, 30.405, 35.135]  # Qwen2.5-Math-7B
    # values = [5.81, 10.46, 15.11, 20.93, 25.58, 30.23, 34.88] # deepseek 7B
    # values = [10.46, 15.11, 20.93, 25.58, 30.23]
    
    # Clear the trap log file at the start
    with open('trap_log.txt', 'w', encoding='utf-8') as log_file:
        log_file.write("Trap Detection Log\n")
        log_file.write("=" * 60 + "\n")
    
    # Process each value
    for value in values:
        process_combined_data(value, model_name, output_dir, result_type)


if __name__ == "__main__":
    main()
