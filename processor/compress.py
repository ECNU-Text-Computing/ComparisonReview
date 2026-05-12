import os

import pandas as pd
from datasets import load_from_disk


def main(venue: str) -> None:
    save_path = f"../data/{venue}/distribution/distribution.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if venue == "iclr_2025":
        dataset_path = f"../data/{venue}/iclr"
        submission_2023 = 4955
        accepted_2023 = 1575
        submission_2024 = 7304
        accepted_2024 = 2260
        dataset = load_from_disk(dataset_path)
        test_length = len(dataset['test'])
                 
        avg_accept_rate = (accepted_2023 / submission_2023 + accepted_2024 / submission_2024) / 2
        print(f"Average Accept Rate: {avg_accept_rate}")
        accept_start_index = round((1 - avg_accept_rate) * test_length)

    else:
        paper_path = f"../data/{venue}/papers.csv"
        df = pd.read_csv(paper_path)
        test_length = len(df)
        avg_accept_rate = df['decision'].value_counts()[1] / test_length
        print(f"Average Accept Rate: {avg_accept_rate}")
        accept_start_index = round((1 - avg_accept_rate) * test_length)

    y_true_list = [i / test_length * 10 for i in range(test_length)]
    decision_list = [0 if i < accept_start_index else 1 for i in range(test_length)]
    df = pd.DataFrame({
        'y_true': y_true_list,
        'decision': decision_list
    })
    print(df)

    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main("iclr_2025")


