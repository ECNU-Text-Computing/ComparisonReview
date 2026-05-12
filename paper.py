from dataclasses import dataclass
import re
import pandas as pd
from datasets import load_from_disk


@dataclass
class Paper:
    paper_id: int
    title: str
    abstract: str
    y_true: float
    decision: int


def save_papers(
        papers: list[Paper],
        paper_path: str
) -> None:
    rows = [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "y_true": paper.y_true,
            "decision": paper.decision,
        }
        for paper in papers
    ]
    df = pd.DataFrame(rows)
    df.to_csv(paper_path, index=False)

def get_papers(
        dataset_path: str,
        split: str = "test"
) -> list[Paper]:
    dataset = load_from_disk(dataset_path)
    dataset = dataset[split]
    papers = []
    for sample in dataset:
        title = sample["title"]
        abstract = sample["abstract"]
        y_true = sample["y_true"]
        decision = sample["decision"]
               
        paper_id = len(papers)
                     
        papers.append(
            Paper(paper_id=paper_id, title=title, abstract=abstract, y_true=y_true, decision=decision)
        )
    return papers

def get_other_papers(
        other_data_path: str,
        number: int = 600,
        random_seed: int = 42
) -> list[Paper]:
    df = pd.read_csv(other_data_path)
                      
    if len(df) > number:
        df = df.sample(n=number, random_state=random_seed).reset_index(drop=True)
    papers = []
    for index, row in df.iterrows():
        title = row['title']
        abstract = row['abstract']
        y_true = row['avg_rating']
        if 'Accept' in row['decision']:
            decision = 1
        elif 'Reject' in row['decision']:
            decision = 0
        else:
            decision = -1
                              
        paper_id = index
                     
        papers.append(
            Paper(paper_id=paper_id, title=title, abstract=abstract, y_true=y_true, decision=decision)
        )
    return papers
