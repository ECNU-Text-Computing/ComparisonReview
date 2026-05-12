from dataclasses import dataclass
import pandas as pd
from openai import OpenAI
import json
import time
import os


API_KEY = "sk-anonymous"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "z-ai/glm-4.5-air:free"


@dataclass
class PaperAnswers:
    focal_papers: list[str]
    other_papers: list[str]


def extract_json(answer):
                          
    try:
        if isinstance(answer, str):
            data = json.loads(answer)
            paper1, paper2 = data.get('paper_1_review', ''), data.get('paper_2_review', '')

            def replace_paper_id(s):
                return s.replace('aper 1', 'aper').replace('aper 2', 'aper')

            paper1 = replace_paper_id(paper1)
            paper2 = replace_paper_id(paper2)
            return paper1, paper2
        return '', ''
    except json.JSONDecodeError:
        return '', ''


def main():
    output_file = '../data/iclr_2025/merged_answers/summaries-standard_Qwen2.5-7B-Instruct-grpo.csv'

                    
    processed_ids = set()
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        processed_ids = set(existing_df['paper_id'].tolist())
        print(f"Processed {len(processed_ids)} papers...")

          
    df = pd.read_csv('../data/iclr_2025/comparisons-standard_Qwen2.5-7B-Instruct-grpo.csv')
    all_paper_ids = sorted(pd.concat([df['paper_id_1'], df['paper_id_2']]).unique().tolist())

                       
    paper_answers = {}
    for paper_id in all_paper_ids:
        if paper_id in processed_ids:
            continue

        paper_answers[paper_id] = PaperAnswers(focal_papers=[], other_papers=[])

                           
        mask1 = (df['paper_id_1'] == paper_id)
        for answer in df[mask1].sort_values(['paper_id_1', 'paper_id_2'])['answer'].tolist():
            focal_paper, other_paper = extract_json(answer)
            paper_answers[paper_id].focal_papers.append(focal_paper)
            paper_answers[paper_id].other_papers.append(other_paper)

                           
        mask2 = (df['paper_id_2'] == paper_id)
        for answer in df[mask2].sort_values(['paper_id_2', 'paper_id_1'])['answer'].tolist():
            other_paper, focal_paper = extract_json(answer)
            paper_answers[paper_id].focal_papers.append(focal_paper)
            paper_answers[paper_id].other_papers.append(other_paper)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for paper_id, answers in paper_answers.items():
        print(f"Processing Paper ID: {paper_id}")

        prompt = f"""
Please output the paper review in two paragraphs, strictly following the content requirements and do not add any other information:
1. Summarize the core content of the review of THIS PAPER.
2. Compare this paper with other papers in a comparative review, using a longer description to make a detailed comparison with other important and similar papers.

## THIS PAPER
```
{str(answers.focal_papers)}
```

## other papers
```
{str(answers.other_papers)}
```
"""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.choices[0].message.content
        except Exception as e:
            print(f"Error processing paper {paper_id}: {e}")
            summary = ""
                
        result = pd.DataFrame([[paper_id, summary]], columns=['paper_id', 'summary'])
        result.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

        time.sleep(3)


if __name__ == '__main__':
    main()
