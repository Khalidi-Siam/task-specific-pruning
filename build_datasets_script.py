from load_and_lebel_datasets import load_and_label_datasets
from dataset_helper import dataset_helper

category_labels = {
    "conversational_daily_dialog": 1,
    "QnA": 2,
    "code": 3,
    "math": 4,
}

dataset_name = {
    1 : "suriya7/everyday-Conversational-cleaned",
    2 : "rajpurkar/squad",
    3 : "extracted_sample_1k",
    4 : "gsm8k" 
}

load_and_label_datasets(category_labels)
dataset_helper(
    target_math=4,
    target_code=3,
    distractor_labels=[1, 2]
)





