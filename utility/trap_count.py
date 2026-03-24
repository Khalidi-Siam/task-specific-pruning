import json
import csv

# Input JSON file path

def trap_count(values):
    for value in values:
        file_path = f"math_generation_metrics_pruned_({value}).json"

        # Token threshold for trap detection
        TOKEN_THRESHOLD = 1024

        trap_count = 0

        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Access per_prompt_metrics
        per_prompt_metrics = data.get('per_prompt_metrics', [])

        # Process each metric
        for metric in per_prompt_metrics:
            if 'prompt_no' in metric and 'tokens_generated' in metric:
                prompt_no = metric['prompt_no']
                tokens_generated = metric['tokens_generated']
                # Check if tokens_generated >= threshold (exact match for traps)
                if tokens_generated >= TOKEN_THRESHOLD:
                    trap_count += 1

        # Generate CSV log file
        csv_file = file_path.replace('.json', '.csv')

        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['prompt_no', 'comment'])
            for metric in per_prompt_metrics:
                if 'prompt_no' in metric and 'tokens_generated' in metric:
                    prompt_no = metric['prompt_no']
                    tokens_generated = metric['tokens_generated']
                    comment = 'trap' if tokens_generated >= TOKEN_THRESHOLD else ''
                    writer.writerow([prompt_no, comment])

        # Print summary
        print(f"Total trapped responses (tokens_generated >= {TOKEN_THRESHOLD}): {trap_count}")
        print(f"Log file generated: {csv_file}")


if __name__ == "__main__":
    values = [5.81, 10.46, 15.11, 20.93, 25.58, 30.23, 34.88]
    trap_count(values)