import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import HF_TOKEN
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login


iclr_2024 = pd.read_csv('../data/iclr_2024/submissions/submissions.csv')
iclr_2025 = pd.read_csv('../data/iclr_2025/submissions/submissions.csv')
iclr_2024_id = iclr_2024['submission_id'].tolist()
iclr_2025_id = iclr_2025['submission_id'].tolist()
                                                                                               


def get_dataset(
        dataset: Dataset,
        split: str = "train",
        year: str = "2025",
) -> Dataset:
          
    data_list = []
    for item in tqdm(dataset[split], desc="Filtering data"):
        if item['mode'] != 'best':
            continue
                    
        if item['id'] in iclr_2025_id:
            if year == "2025":
                title = iclr_2025[iclr_2025['submission_id'] == item['id']]['title'].values[0]
                abstract = iclr_2025[iclr_2025['submission_id'] == item['id']]['abstract'].values[0]
            else:
                continue
        elif item['id'] in iclr_2024_id:
            if year == "2024":
                title = iclr_2024[iclr_2024['submission_id'] == item['id']]['title'].values[0]
                abstract = iclr_2024[iclr_2024['submission_id'] == item['id']]['abstract'].values[0]
            else:
                continue
        else:
            continue
                    
        item_rating = ast.literal_eval(item['rating'])
        y_true = float(sum(item_rating) / len(item_rating)) if item['rating'] else 0.0
                         
        if item['decision'] == 'Accept':
            decision = 1
        elif item['decision'] == 'Reject':
            decision = 0
        else:
            decision = -1
              
        data_dict = {
            'title': title,
            'abstract': abstract,
            'inputs': item['inputs'],
            'outputs': item['outputs'],
            'y_true': y_true,
            'decision': decision
        }
        data_list.append(data_dict)
    new_dataset = Dataset.from_list(data_list)
    return new_dataset


def load_data(venue: str):

    save_path = f"../data/{venue}/iclr"

    login(token=HF_TOKEN)

    dataset = load_dataset("WestlakeNLP/DeepReview-13K")

    train_dataset = get_dataset(dataset, "train", venue)
    test_dataset = get_dataset(dataset, "test", venue)
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
          
    dataset.save_to_disk(save_path)


if __name__ == '__main__':
    load_data("iclr_2025")

