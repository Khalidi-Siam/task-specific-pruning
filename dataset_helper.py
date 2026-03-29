import os
import pandas as pd
from sklearn.model_selection import train_test_split

from create_math_test_set import create_math_test_set


CATEGORY_LABELS = {
    "conversational_daily_dialog": 1,
    "QnA": 2,
    "code": 3,
    "math": 4,
    "poetry": 5,
    "humor": 6,
}

TEST_FILES_BY_LABEL = {
    1: "conversational_prompts.csv",
    2: "qna_prompts.csv",
    3: "code_prompts.csv",
}


def _print_summary(df, prefix):
    print(f"\n{prefix} rows: {len(df)}")
    if len(df) == 0:
        return
    print("Label counts:")
    print(df["label"].value_counts().sort_index())
    if "target_distractor" in df.columns:
        print("Target/Distractor counts:")
        print(df["target_distractor"].value_counts().sort_index())


def _build_train_for_target(df, target_label, distractor_labels, output_path):
    selected_labels = set([target_label] + distractor_labels)
    train_df = df[df["label"].isin(selected_labels)].copy()

    train_df["target_distractor"] = train_df["label"].apply(
        lambda x: 1 if int(x) == int(target_label) else 0
    )
    train_df = train_df.reset_index(drop=True)

    train_df.to_csv(output_path, index=False)
    print(f"Saved train set: {output_path} ({len(train_df)} rows)")
    _print_summary(train_df, f"Train target={target_label}")


def _build_non_math_test_sets(test_df):
    # Build conversational and QnA test sets from split test partition
    for label, file_name in TEST_FILES_BY_LABEL.items():
        if label == 3:  # Skip code, it will be loaded from root directory
            continue
        task_df = test_df[test_df["label"] == label][["prompt"]].copy().reset_index(drop=True)
        output_path = os.path.join("datasets", file_name)
        task_df.to_csv(output_path, index=False)
        print(f"Saved test set: {output_path} ({len(task_df)} rows)")


def _build_code_test_set_from_root():
    # Load code_prompts.csv from root directory and save to datasets directory
    root_code_path = "code_prompts.csv"
    if not os.path.exists(root_code_path):
        print(f"Warning: {root_code_path} not found in root directory, skipping code prompts")
        return
    
    code_df = pd.read_csv(root_code_path)[["prompt"]].copy().reset_index(drop=True)
    output_path = os.path.join("datasets", "code_prompts.csv")
    code_df.to_csv(output_path, index=False)
    print(f"Saved test set: {output_path} ({len(code_df)} rows)")


def dataset_helper(
    csv_path="datasets/category_labeled_prompts.csv",
    random_state=42,
    test_size=0.2,
    target_math=4,
    target_code=3,
    distractor_labels=None,
):
    if distractor_labels is None:
        distractor_labels = [1, 2]

    os.makedirs("datasets", exist_ok=True)

    print(f"Loading category-labeled dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    required_columns = {"prompt", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Input CSV must contain at least 'prompt' and 'label' columns")

    df["prompt"] = df["prompt"].astype(str).fillna("")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    _print_summary(train_df, "Base train split")
    _print_summary(test_df, "Base test split")

    # Build two target-specific train files that share the same distractors.
    _build_train_for_target(
        train_df,
        target_label=target_math,
        distractor_labels=distractor_labels,
        output_path="datasets/train_prompts_math_target.csv",
    )
    _build_train_for_target(
        train_df,
        target_label=target_code,
        distractor_labels=distractor_labels,
        output_path="datasets/train_prompts_code_target.csv",
    )

    # Build non-math task test sets from the split test partition.
    _build_non_math_test_sets(test_df)

    # Build code test set from root directory code_prompts.csv
    _build_code_test_set_from_root()

    # Build math task test set from GSM8K indices (not from split test partition).
    create_math_test_set(output_path="datasets/math_prompts.csv")

    print("\nAll datasets created successfully:")
    print("- datasets/train_prompts_math_target.csv")
    print("- datasets/train_prompts_code_target.csv")
    print("- datasets/conversational_prompts.csv")
    print("- datasets/qna_prompts.csv")
    print("- datasets/code_prompts.csv")
    print("- datasets/math_prompts.csv")
    print("- datasets/ground truth.csv")

