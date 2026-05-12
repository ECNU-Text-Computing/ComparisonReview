import os
from typing import Tuple, Optional
from collections import defaultdict
import pandas as pd
from datasets import Dataset, DatasetDict
from paper import get_papers, Paper
from reviewer.review import PeerReviewSystem
import json
import random
import re

          
GENERATE = True
                                                                   
MODE = "mixed"
                  
BATCH_SIZE = 64

def generate_unique_paper_pairs(
        papers: list[Paper],
        comparison_path: str,
        mode: str = "ordered",
        similarity_path: Optional[str] = None,
) -> pd.DataFrame:

    print(f"Sampling mode: {mode}")

    n = len(papers)
    print(f"Total paper nums: {n}")

            
    sampled_paper_pairs = []

    if mode == "ordered" or mode == "mixed":
                
        used_paper_ids = set()
        for i in range(n):
            for j in range(i + 1, n):
                          
                if papers[i].paper_id in used_paper_ids or papers[j].paper_id in used_paper_ids:
                    continue
                                  
                if abs(papers[i].y_true - papers[j].y_true) < 1.5:
                    continue
                used_paper_ids.add(papers[i].paper_id)
                used_paper_ids.add(papers[j].paper_id)
                sampled_paper_pairs.append((papers[i], papers[j]))

    if mode == "similarity" or mode == "mixed":
                      
        similarity_df = pd.read_csv(similarity_path)
        paper_dict = {paper.paper_id: paper for paper in papers}
        used_paper_ids = set()
        for _, row in similarity_df.iterrows():
            paper_id_1 = row['paper_id_1']
            paper_id_2 = row['paper_id_2']
                         
            if paper_id_1 not in paper_dict or paper_id_2 not in paper_dict:
                continue
            if paper_id_1 in used_paper_ids or paper_id_2 in used_paper_ids:
                continue
            paper_1 = paper_dict[paper_id_1]
            paper_2 = paper_dict[paper_id_2]
                              
            if abs(paper_1.y_true - paper_2.y_true) < 1.5:
                continue
            used_paper_ids.add(paper_id_1)
            used_paper_ids.add(paper_id_2)
            sampled_paper_pairs.append((paper_1, paper_2))

    if mode not in ["ordered", "similarity", "mixed"]:
        raise ValueError(f"Unsupported mode: {mode}.")

          
    rows = [
        {
            "paper_id_1": p1.paper_id,
            "title_1": p1.title,
            "abstract_1": p1.abstract,
            "y_true_1": p1.y_true,
            "decision_1": p1.decision,
            "paper_id_2": p2.paper_id,
            "title_2": p2.title,
            "abstract_2": p2.abstract,
            "y_true_2": p2.y_true,
            "decision_2": p2.decision,
        }
        for p1, p2 in sampled_paper_pairs
    ]
    df = pd.DataFrame(rows)
    df.to_csv(comparison_path, index=False)
    return sampled_paper_pairs


def swap_prompt(text: str) -> str:
    pattern = re.compile(
        r"### Paper 1:\n(.*?\n```(?:.|\n)*?```)\n\n### Paper 2:\n(.*?\n```(?:.|\n)*?```)",
        re.DOTALL
    )
    swapped = pattern.sub(r"### Paper 1:\n\2\n\n### Paper 2:\n\1", text)
    return swapped

def swap_completion(text: str) -> str:
    def swap_text_pairs(t, pattern1, pattern2, temp_prefix="TEMP_"):
        temp_marker = temp_prefix + pattern1
        t = t.replace(pattern1, temp_marker)
        t = t.replace(pattern2, pattern1)
        t = t.replace(temp_marker, pattern2)
        return t

    text = swap_text_pairs(text, 'paper_1', 'paper_2')
    text = swap_text_pairs(text, 'Paper 1', 'Paper 2')
    return text

def swap_ground_truth(label: str) -> str:
    if label == "0":
        return "1"
    elif label == "1":
        return "0"
    else:
        return label

def filter_data(
        data_list: list[dict],
        seed: int = 42
) -> list[dict]:
    random.seed(seed)
    prompts = [ex["prompt"] for ex in data_list]
    ground_truth = [ex["ground_truth"] for ex in data_list]
    completions = [ex["completion"] for ex in data_list]
    new_data_list = []
    for p, c, gt in zip(prompts, completions, ground_truth):
        try:
            if random.random() > 0.5:
                p = swap_prompt(p)
                c = swap_completion(c)
                gt = swap_ground_truth(gt)
            if not c == "":
                c = json.loads(c)
                required_keys = {"paper_1_review", "paper_2_review", "chosen_paper"}
                custom_order = ["paper_1_review", "paper_2_review", "chosen_paper"]
                if not required_keys.issubset(c.keys()):
                    continue
                c = json.dumps({k: c[k] for k in custom_order if k in c})
            new_data_list.append({
                "prompt": p,
                "completion": c,
                "ground_truth": gt
            })
        except json.decoder.JSONDecodeError:
            continue
    return new_data_list


def run(
        venue: str,
        split: str,
        review_system: PeerReviewSystem,
) -> Tuple[Dataset, Dataset]:
           
    dataset_path = f"./data/{venue}/iclr"
    comparison_path = f"./data/{venue}/comparisons_{split}.csv"
    similarity_path = f"./data/{venue}/embedding/similarity_train.csv" if split == "train"        else f"./data/{venue}/embedding/similarity.csv"

    if GENERATE:
        papers = get_papers(dataset_path, split=split)
                         
        if not os.path.exists(comparison_path):
            if MODE == "ordered":
                train_df = generate_unique_paper_pairs(
                    papers=papers,
                    comparison_path=comparison_path,
                    mode="ordered"
                )
            elif MODE == "similarity":
                train_df = generate_unique_paper_pairs(
                    papers=papers,
                    comparison_path=comparison_path,
                    mode="similarity",
                    similarity_path=similarity_path
                )
            elif MODE == "mixed":
                train_df = generate_unique_paper_pairs(
                    papers=papers,
                    comparison_path=comparison_path,
                    mode="mixed",
                    similarity_path=similarity_path
                )

            print(f"Total pair nums: {len(train_df)}")
        review_system.generate_comparisons(
            comparison_path=comparison_path,
            batch_size=BATCH_SIZE
        )

               
    df = pd.read_csv(comparison_path)
    df = df.dropna()
    sft_data_list = []
    grpo_data_list = []
    for index, row in df.iterrows():
        paper1, paper2 = review_system.read_paper_pair(row)
        if int(paper1.y_true > paper2.y_true) != int(row["preference"]):
                         
            prompt = review_system.judger.create_prompt(paper1, paper2)
            ground_truth = str(int(paper1.y_true > paper2.y_true))
            data_dict = {
                "prompt": prompt,
                "completion": "",
                "ground_truth": ground_truth
            }
            grpo_data_list.append(data_dict)
        else:
                         
            prompt = review_system.judger.create_prompt(paper1, paper2)
            completion = row["answer"]
            ground_truth = str(int(paper1.y_true > paper2.y_true))
            data_dict = {
                "prompt": prompt,
                "completion": completion,
                "ground_truth": ground_truth
            }
            sft_data_list.append(data_dict)
            grpo_data_list.append(data_dict)

             
    sft_data_list = filter_data(sft_data_list)
    grpo_data_list = filter_data(grpo_data_list)
    sft_dataset = Dataset.from_list(sft_data_list)
    grpo_dataset = Dataset.from_list(grpo_data_list)
    return sft_dataset, grpo_dataset

def construct(venue: str):
           
    sft_path = f"./data/{venue}/comparisons_sft"
    grpo_path = f"./data/{venue}/comparisons_grpo"

    review_system = PeerReviewSystem(
        comparison_model="./model/Qwen3-235B-A22B-Instruct-2507-AWQ",
                                                  
        generate=True
    )
    train_sft_dataset, train_grpo_dataset = run(year, "train", review_system)
    test_sft_dataset, test_grpo_dataset = run(year, "test", review_system)
    sft_dataset_dict = DatasetDict({
        "train": train_sft_dataset,
        "test": test_sft_dataset
    })
    grpo_dataset_dict = DatasetDict({
        "train": train_grpo_dataset,
        "test": test_grpo_dataset
    })
    sft_dataset_dict.save_to_disk(sft_path)
    grpo_dataset_dict.save_to_disk(grpo_path)


if __name__ == "__main__":
    construct("iclr_2025")
