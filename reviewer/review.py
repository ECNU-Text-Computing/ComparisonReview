import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Any
import random
from utils.data_utils import DISTRIBUTION, VALID_NUMBERS
from reviewer.judge import LLMJudger
from paper import Paper
from tqdm import tqdm
from metrics.score_metrics import ScoreEvaluator


MAX_SAMPLE_RATE = 0.055


class PeerReviewSystem:

    def __init__(
            self,
            comparison_model: str,
            generate: bool = False,
    ):
        if not generate:
            self.judger = None
        elif "Qwen2.5-7B-Instruct" in comparison_model:
            self.judger = LLMJudger(comparison_model, online=False)
        elif "Qwen3-235B-A22B-Instruct-2507" or "gpt-oss-120b" in comparison_model:
            self.judger = LLMJudger(comparison_model, online=True)
        else:
            raise ValueError("Invalid comparison model.")
        self.evaluator = ScoreEvaluator()

    @staticmethod
    def generate_paper_pairs(
            papers: list[Paper],
            comparison_path: str,
            pair_type: str = "random",
            seed: int = 42,
            max_sample_rate: float = MAX_SAMPLE_RATE,
            similarity_path: Optional[str] = None
    ) -> pd.DataFrame:
        n = len(papers)
        print(f"Total paper nums: {n}")
        max_paper_pairs = int(n * (n - 1) / 2 * max_sample_rate)
        if pair_type == "standard":
            pair_df = pd.read_csv(similarity_path)
            paper_pairs = [(papers[i], papers[j]) for i, j in pair_df[["paper_id_1", "paper_id_2"]].values]
            sampled_paper_pairs = paper_pairs[:max_paper_pairs]
        elif pair_type == "random":
            paper_pairs = [(papers[i], papers[j]) for i in range(n) for j in range(i + 1, n)]
            random.seed(seed)
            sampled_paper_pairs = random.sample(paper_pairs, min(len(paper_pairs), max_paper_pairs))
        else:
            raise ValueError("Invalid pair type.")
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

    @staticmethod
    def read_paper_pair(
            row: pd.Series
    ) -> Tuple[Paper, Paper]:
        paper_1 = Paper(
            paper_id=row["paper_id_1"],
            title=row["title_1"],
            abstract=row["abstract_1"],
            y_true=row["y_true_1"],
            decision=row["decision_1"],
        )
        paper_2 = Paper(
            paper_id=row["paper_id_2"],
            title=row["title_2"],
            abstract=row["abstract_2"],
            y_true=row["y_true_2"],
            decision=row["decision_2"],
        )
        return paper_1, paper_2

    def generate_comparisons(
            self,
            comparison_path: str,
            batch_size: int = 16
    ) -> pd.DataFrame:
        df = pd.read_csv(comparison_path, low_memory=False)
        print(f"Total comparison nums: {len(df)}")
        if "preference" not in df.columns:
            df["preference"] = pd.NA
            df["answer"] = pd.NA
            df["input_tokens"] = pd.NA
            df["output_tokens"] = pd.NA
            df.to_csv(comparison_path, index=False)

        def _df_batch_generator(_df, _start_index, _batch_size):
            _df = _df.reset_index(drop=True)
            _df_slice = _df.iloc[_start_index:]
            _total_batches = (len(_df_slice) + _batch_size - 1) // _batch_size
            with tqdm(total=_total_batches, desc="Processing batches") as _pbar:
                for _start in range(_start_index, len(_df), _batch_size):
                    _end = _start + _batch_size
                    _batch = _df.iloc[_start:_end]
                    yield _start, _end, _batch
                    _pbar.update(1)

        start_index = len(df)
        for index, row in df.iterrows():
            if pd.isna(row["preference"]):
                start_index = index
                break
        if start_index == len(df):
            return df
        print(f"Start comparison from line {start_index}.")
        for start, end, batch in _df_batch_generator(df, start_index, batch_size):
            paper_batch = [self.read_paper_pair(row) for _, row in batch.iterrows()]
            preferences, answers, token_counts = self.judger.batch_compare(paper_batch)
            batch.loc[:, "preference"] = preferences
            batch.loc[:, "answer"] = answers
            batch.loc[:, "input_tokens"] = [token_count.input_tokens for token_count in token_counts]
            batch.loc[:, "output_tokens"] = [token_count.output_tokens for token_count in token_counts]
            cols_to_update = ['preference', 'answer', 'input_tokens', 'output_tokens']
            df.iloc[start:end, df.columns.get_indexer(cols_to_update)] = batch[cols_to_update].values
            df.to_csv(comparison_path, index=False)
        return df

    @staticmethod
    def bradley_terry_ranking(
            comparison_paths: list[str],
            score_path: str,
            paper_path: str,
            distribution_path: str,
            rank_method: str = "bradley_terry",
            target_percentage: float = 0.05,
            keep_number: int = 1e8,
            random_seed: int = 42
    ) -> dict[int, float]:
        papers_df = pd.read_csv(paper_path, low_memory=False)
        if keep_number < len(papers_df):
            random.seed(random_seed)
            indices = list(range(len(papers_df)))
            random.shuffle(indices)
            papers_df = papers_df.iloc[indices[:keep_number]].copy()
            print(f"Keep {len(papers_df)} papers.")

        comparisons = []
        for comparison_path in comparison_paths:
            comparisons_df = pd.read_csv(comparison_path, low_memory=False)
            comparisons_df = comparisons_df[comparisons_df['preference'].notna()]
            if keep_number < len(papers_df):
                kept_paper_ids = set(papers_df['paper_id'].values)
                comparisons_df = comparisons_df[
                    (comparisons_df['paper_id_1'].isin(kept_paper_ids)) &
                    (comparisons_df['paper_id_2'].isin(kept_paper_ids))
                ].copy()
            comparisons_df['confidence'] = 1
            current_comparisons = list(zip(
                comparisons_df['paper_id_1'],
                comparisons_df['paper_id_2'],
                comparisons_df['preference'],
                comparisons_df['confidence']
            ))
            n_papers = len(papers_df)
            n_comparisons = len(comparisons_df)
            current_percentage = n_comparisons / (n_papers * (n_papers - 1) / 2)
            print(f"Available sample percentage: {current_percentage:.2%}")
            if current_percentage < target_percentage:
                print("Not enough data available. Please collect more data.")
                return
            elif current_percentage > target_percentage:
                current_comparisons = current_comparisons[:round((n_papers * (n_papers - 1) / 2) * target_percentage)]
                print(f"Target percentage: {target_percentage:.2%}")
            comparisons.extend(current_comparisons)

        paper_ids = set()
        for paper_id_1, paper_id_2, _, _ in comparisons:
            paper_ids.add(paper_id_1)
            paper_ids.add(paper_id_2)
        paper_id_list = sorted(list(paper_ids))
        id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_id_list)}

        if rank_method == "bradley_terry":

            def neg_log_likelihood(theta, comp, id_map):
                i = np.array([id_map[p1] for p1, _, _, _ in comp])
                j = np.array([id_map[p2] for _, p2, _, _ in comp])
                prefs = np.array([pref for _, _, pref, _ in comp])
                confs = np.array([conf for _, _, _, conf in comp])
                winner = np.where(prefs == 1, i, j)
                loser = np.where(prefs == 1, j, i)
                theta_w = theta[winner]
                theta_l = theta[loser]
                max_theta = np.maximum(theta_w, theta_l)
                log_sum_exp = max_theta + np.log(np.exp(theta_w - max_theta) + np.exp(theta_l - max_theta))
                return -np.sum(confs * (theta_w - log_sum_exp))

            theta_init = np.zeros(len(paper_ids))
            constraints = {'type': 'eq', 'fun': lambda theta: theta[-1]}
            result = minimize(neg_log_likelihood, theta_init, args=(comparisons, id_to_idx),
                              constraints=constraints, method='SLSQP')
            theta_estimated = result.x


        scores = {int(paper_id_list[i]): theta_estimated[i] for i in range(len(paper_id_list))}
        scores = dict(sorted(scores.items(), key=lambda x: x[1]))
        df_distribution = pd.read_csv(distribution_path)
        y_preds = {k: df_distribution['y_true'].iloc[i] for i, k in enumerate(scores.keys())}
        decision_preds = {k: df_distribution['decision'].iloc[i] for i, k in enumerate(scores.keys())}
        papers_df['y_pred'] = papers_df['paper_id'].map(y_preds)
        papers_df['decision_pred'] = papers_df['paper_id'].map(decision_preds)

        papers_df.to_csv(score_path, index=False)
        return scores


    def get_score(
            self,
            score_path: str,
            print_result: bool = True
    ) -> dict[str, Any]:
        papers_df = pd.read_csv(score_path, low_memory=False)
        papers_df = papers_df.dropna()
        y_true = papers_df['y_true'].tolist()
        y_pred = papers_df['y_pred'].tolist()
        decision = papers_df['decision'].tolist()
        decision_pred = papers_df['decision_pred'].tolist()
        result = self.evaluator.evaluate(y_true, y_pred, decision, decision_pred)
        self.evaluator.print_result(result)
        return result
