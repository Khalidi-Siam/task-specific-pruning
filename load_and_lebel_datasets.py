import pandas as pd
from datasets import load_dataset, Dataset
from itertools import islice
import os
from dialogue_preprocess import preprocess_dialogue

def get_streamed_dataset(dataset_name, split, count, **kwargs):
    stream = load_dataset(dataset_name, split=split, streaming=True, **kwargs)
    samples = list(islice(stream, count))
    return Dataset.from_list(samples)


# Combine into labeled DataFrame with numerical category labels
def to_df(dataset, category_num, text_key="text", type=None):
    prompts = []
    print(text_key)
    
    if type == "math":
        # Math: Use only the question
        # var = "\nPlease reason step by step, and put your final answer within \\boxed{}"
        prompts = [f"{row["question"]}" for row in dataset]
    if type == "code":
        # Code: Use the Query column from CSV
        prompts = [row["Query"] for row in dataset]
    elif type == "qna":
        # QnA: Combine context and question with a newline
        prompts = [f"{row['context']}\n{row['question']}" for row in dataset]
    elif type == "dialogue":
        # Use the dialogue preprocessing module
        for row in dataset:
            dialogue_list = row["conversations"]  # assuming "conversations" field is list of dicts with role/text
            processed_prompt = preprocess_dialogue(dialogue_list)
            if processed_prompt is not None:
                prompts.append(processed_prompt)
    else:
        # Default: Use the provided text_key or fallback to "content"
        key = text_key if text_key in dataset.column_names else "content"
        prompts = [row[key] for row in dataset]
    
    return pd.DataFrame({
        "prompt": prompts,
        "label": [category_num] * len(dataset)
    })


def load_and_label_datasets(category_labels):
    chat_ds = get_streamed_dataset("suriya7/everyday-Conversational-cleaned", "train", 1000)
    qna_ds = get_streamed_dataset("rajpurkar/squad", "train", 1000)

    math_ds = get_streamed_dataset("gsm8k", "train", 1000, name="main")
    code_csv_path = os.path.join(".", "extracted_sample_1k.csv")
    code_csv = pd.read_csv(code_csv_path)
    code_ds = Dataset.from_pandas(code_csv)


    df = pd.concat([
        to_df(chat_ds, category_labels["conversational_daily_dialog"], "conversations", "dialogue"),
        to_df(qna_ds, category_labels["QnA"], "question", "qna"),
        to_df(math_ds, category_labels["math"], "question", "math"),
        to_df(code_ds, category_labels["code"], "Query", "code"),
    ])

    # Ensure datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    file_path = os.path.join("datasets", "category_labeled_prompts.csv")

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"✅ Dataset with numerical category labels saved as {file_path} ({len(df)} prompts)")

    # Print label mapping for reference
    print("\nLabel Mapping:")
    for category, num in category_labels.items():
        print(f"{num}: {category}")

    # Print category counts
    print("\nCategory Counts:")
    print(df["label"].value_counts().sort_index().rename(index=category_labels))