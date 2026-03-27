import json
import os
from datetime import datetime

def convert_jsonl_to_txt(jsonl_file_path):
    """
    Convert JSONL file to TXT file with prompt and response format.
    
    Args:
        jsonl_file_path (str): Path to the input JSONL file
    
    Returns:
        str: Path to the created TXT file, or None if failed
    """
    # Generate output file name by replacing .jsonl with .txt
    if jsonl_file_path.endswith('.jsonl'):
        txt_file_path = jsonl_file_path.replace('.jsonl', '.txt')
    else:
        txt_file_path = jsonl_file_path + '.txt'
    
    try:
        prompt_count = 0
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                
                for line_num, line in enumerate(jsonl_file, 1):
                    try:
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        # Parse JSON line
                        data = json.loads(line.strip())
                        
                        prompt_no = data.get('prompt_no', '')
                        prompt = data.get('prompt', '')
                        response = data.get('response', '')
                        
                        # Handle different types of entries
                        if prompt_no == "STARTING":
                            txt_file.write(f"=== GENERATION METADATA ===\n")
                            txt_file.write(f"Start Time: {prompt}\n")
                            # txt_file.write(f"Status: {response}\n")
                            txt_file.write("=" * 50 + "\n\n")
                            
                        elif prompt_no == "SUMMARY":
                            txt_file.write("\n" + "=" * 50 + "\n")
                            txt_file.write(f"=== GENERATION SUMMARY ===\n")
                            txt_file.write(f"End Time: {prompt}\n")
                            txt_file.write(f"Summary: {response}\n")
                            txt_file.write("=" * 50 + "\n")
                            
                        else:
                            # Regular prompt-response pairs
                            prompt_count += 1
                            txt_file.write(f"PROMPT {prompt_no}:\n")
                            txt_file.write(f"{prompt}\n\n")
                            txt_file.write(f"RESPONSE:\n")
                            txt_file.write(f"{response}\n")
                            txt_file.write("-" * 80 + "\n\n")
                            
                    except json.JSONDecodeError as e:
                        print(f"    Warning: Could not parse line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"    Warning: Error processing line {line_num}: {e}")
                        continue
        
        # Verify the file was created successfully
        if os.path.exists(txt_file_path):
            file_size = os.path.getsize(txt_file_path)
            return txt_file_path
        else:
            print(f"    Error: Failed to create {txt_file_path}")
            return None
        
    except FileNotFoundError:
        print(f"    Error: File {jsonl_file_path} not found.")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

def main():
    """
    Main function to convert all JSONL files in model outputs directory to TXT format.
    """
    
    # Base directory containing model outputs
    base_directory = os.path.join("original model outputs")
    # base_directory = os.path.join("finetuned model outputs")
    # base_directory = os.path.join("pruning model outputs")
    
    # Check if base directory exists
    if not os.path.exists(base_directory):
        print(f"Base directory not found: {base_directory}")
        return
    
    print(f"Processing all JSONL files in: {base_directory}")
    print("=" * 60)
    
    # Process all JSONL files in all subdirectories
    total_files_processed = 0
    
    for model_dir in os.listdir(base_directory):
        model_path = os.path.join(base_directory, model_dir)
        
        # Skip if not a directory
        if not os.path.isdir(model_path):
            continue
            
        print(f"\nProcessing model directory: {model_dir}")
        print("-" * 40)
        
        # Find all JSONL files in this model directory
        jsonl_files = [f for f in os.listdir(model_path) if f.endswith('.jsonl')]
        
        if not jsonl_files:
            print(f"  No JSONL files found in {model_dir}")
            continue
        
        # Process each JSONL file
        for jsonl_file in jsonl_files:
            jsonl_file_path = os.path.join(model_path, jsonl_file)
            print(f"  Converting: {jsonl_file}")
            
            result = convert_jsonl_to_txt(jsonl_file_path)
            if result:
                total_files_processed += 1
                print(f"    ✓ Created: {os.path.basename(result)}")
            else:
                print(f"    ✗ Failed to convert: {jsonl_file}")
    
    print("\n" + "=" * 60)
    print(f"Conversion completed! Total files processed: {total_files_processed}")
    
    return total_files_processed

def batch_convert_directory(directory_path):
    """
    Convert all JSONL files in a directory to TXT format.
    
    Args:
        directory_path (str): Path to directory containing JSONL files
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    jsonl_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    
    if not jsonl_files:
        print(f"No JSONL files found in {directory_path}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files. Converting...")
    
    for jsonl_file in jsonl_files:
        print(f"Converting: {jsonl_file}")
        convert_jsonl_to_txt(jsonl_file)
    
    print("Batch conversion completed!")

if __name__ == "__main__":
    # Run the main function to convert all JSONL files to TXT format
    print("JSONL to TXT Converter")
    print("=" * 60)
    print("This script will convert all JSONL files in the model outputs directory")
    print("to TXT format while preserving the original JSONL files.")
    print("=" * 60)
    
    try:
        files_processed = main()
        if files_processed > 0:
            print(f"\n🎉 Success! Converted {files_processed} JSONL files to TXT format.")
        else:
            print("\n⚠️  No files were processed. Please check the directory structure.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        
    input("\nPress Enter to exit...")  # Keep window open to see results