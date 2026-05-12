import numpy as np
from typing import Any, Optional
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, cohen_kappa_score, mean_squared_error
from sklearn.metrics import roc_auc_score, jaccard_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from ranx import Qrels, Run, evaluate
from metrics.base_metrics import BaseEvaluator


class ScoreEvaluator(BaseEvaluator):
    def __init__(
            self,
            enable_average_performance: bool = True,
            enable_average_rank: bool = False
    ):
        super().__init__()
        self.metric_map = {
            'decision_accuracy': 'D. Acc.',
            'decision_f1': 'D. F1',
            'decision_auc': 'D. AUC',
            'decision_cohen': "D. Cohen's Kappa",
            'spearman': "Spearman's Rho",
            'pairwise_accuracy': 'Pairwise Acc.',
            'map_20': 'MAP@20',
            'ndcg_20': 'NDCG@20',
        }

        self.enable_average_performance = enable_average_performance
        if self.enable_average_performance:
            self.metric_map.update({
                'avg_performance': "Avg. Performance",
            })
        self.enable_average_rank = enable_average_rank
        if self.enable_average_rank:
            self.metric_map.update({
                'avg_rank': "Avg. Rank",
            })

    @staticmethod
    def calculate_decision_accuracy(
            decision: list[int],
            decision_pred: list[int]
    ) -> float:
        decision_accuracy = accuracy_score(decision, decision_pred)
        return decision_accuracy

    @staticmethod
    def calculate_decision_jaccard(
            decision: list[int],
            decision_pred: list[int]
    ) -> float:
        jaccard = jaccard_score(decision, decision_pred)
        return jaccard

    @staticmethod
    def calculate_decision_f1(
            decision: list[int],
            decision_pred: list[int],
            average: str = 'macro'
    ) -> float:
        decision_f1 = f1_score(decision, decision_pred, average=average)
        return decision_f1

    @staticmethod
    def calculate_decision_cohen(
            y_true: list[float],
            y_pred: list[float]
    ) -> float:
        cohen = cohen_kappa_score(y_true, y_pred)
        return cohen

    @staticmethod
    def calculate_pairwise_accuracy(
            y_true: list[float],
            y_pred: list[float]
    ) -> float:
        true_orders = []
        pred_orders = []
        for i in range(len(y_true)):
            for j in range(i + 1, len(y_true)):
                if y_true[i] != y_true[j]:
                    true_orders.append(y_true[i] > y_true[j])
                    pred_orders.append(y_pred[i] > y_pred[j])
        return accuracy_score(true_orders, pred_orders) if true_orders else 0.0

    @staticmethod
    def calculate_decision_auc(
            decision: list[int],
            y_pred: list[float]
    ) -> float:
        try:
            auc = roc_auc_score(decision, y_pred)
            return auc
        except ValueError:
            return 0.0

    @staticmethod
    def calculate_spearman(
            y_true: list[float],
            y_pred: list[float]
    ) -> float:
        spearman, p_value = spearmanr(y_true, y_pred)
        if np.isnan(spearman):
            spearman = 0.0
        return spearman

    @staticmethod
    def calculate_kendall(
            y_true: list[float],
            y_pred: list[float]
    ) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        n = len(y_true)
        if n < 2:
            return 0.0
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = y_true[i] - y_true[j]
                dy = y_pred[i] - y_pred[j]
                if dx * dy > 0:
                    concordant += 1
                elif dx * dy < 0:
                    discordant += 1
        total_pairs = n * (n - 1) // 2
        if total_pairs == 0:
            return 0.0
        tau_a = (concordant - discordant) / total_pairs
        return float(tau_a)

    @staticmethod
    def calculate_map(
            y_true: list[float],
            y_pred: list[float],
            k: int = 1000
    ) -> float
        if len(y_true) == 0:
            return 0.0
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        threshold_percentile = (1 - 0.31364012144348424) * 100
        threshold = np.percentile(y_true, threshold_percentile)
        relevance = (y_true > threshold).astype(int)
        sorted_indices = np.argsort(-y_pred)
        top_k_indices = sorted_indices[:k]
        ranked_relevance = relevance[top_k_indices]
        num_relevant = np.sum(relevance)
        if num_relevant == 0:
            return 0.0
        ap = 0.0
        num_hits = 0
        for i, rel in enumerate(ranked_relevance, start=1):
            if rel == 1:
                num_hits += 1
                precision_at_i = num_hits / i
                ap += precision_at_i
        ap /= min(num_relevant, k)
        return float(ap)

    @staticmethod
    def calculate_ndcg(
            y_true: list[float],
            y_pred: list[float],
            k: int = 20
    ) -> float:
        if len(y_true) == 0:
            return 0.0
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        qrels_dict = {"q1": {str(i): float(rel) for i, rel in enumerate(y_true)}}
        run_dict = {"q1": {str(i): float(score) for i, score in enumerate(y_pred)}}
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)
        ndcg = evaluate(qrels, run, f"ndcg@{k}")
        return ndcg

    def evaluate(
            self,
            y_true: list[float],
            y_pred: list[float],
            decision: list[int],
            decision_pred: list[int]
    ) -> dict[str, Any]:
        result = {}
        if 'decision_accuracy' in self.metric_map:
            result['decision_accuracy'] = self.calculate_decision_accuracy(decision, decision_pred)
        if 'decision_jaccard' in self.metric_map:
            result['decision_jaccard'] = self.calculate_decision_jaccard(decision, decision_pred)
        if 'decision_f1' in self.metric_map:
            result['decision_f1'] = self.calculate_decision_f1(decision, decision_pred, average='macro')
        if 'decision_f1_weighted' in self.metric_map:
            result['decision_f1_weighted'] = self.calculate_decision_f1(decision, decision_pred, average='weighted')
        if 'decision_cohen' in self.metric_map:
            result['decision_cohen'] = self.calculate_decision_cohen(decision, decision_pred)
        if 'decision_auc' in self.metric_map:
            result['decision_auc'] = self.calculate_decision_auc(decision, y_pred)
        if 'pairwise_accuracy' in self.metric_map:
            result['pairwise_accuracy'] = self.calculate_pairwise_accuracy(y_true, y_pred)
        if 'mae' in self.metric_map:
            result['mae'] = self.calculate_mae(y_true, y_pred)
        if 'mse' in self.metric_map:
            result['mse'] = self.calculate_mse(y_true, y_pred)
        if 'spearman' in self.metric_map:
            result['spearman'] = self.calculate_spearman(y_true, y_pred)
        if 'kendall' in self.metric_map:
            result['kendall'] = self.calculate_kendall(y_true, y_pred)
        if 'map' in self.metric_map:
            result['map'] = self.calculate_map(y_true, y_pred)
        if 'ndcg' in self.metric_map:
            result['ndcg'] = self.calculate_ndcg(y_true, y_pred)
        if 'map_10' in self.metric_map:
            result['map_10'] = self.calculate_map(y_true, y_pred, k=10)
        if 'ndcg_10' in self.metric_map:
            result['ndcg_10'] = self.calculate_ndcg(y_true, y_pred, k=10)
        if 'map_20' in self.metric_map:
            result['map_20'] = self.calculate_map(y_true, y_pred, k=20)
        if 'ndcg_20' in self.metric_map:
            result['ndcg_20'] = self.calculate_ndcg(y_true, y_pred, k=20)
        if 'map_50' in self.metric_map:
            result['map_50'] = self.calculate_map(y_true, y_pred, k=50)
        if 'ndcg_50' in self.metric_map:
            result['ndcg_50'] = self.calculate_ndcg(y_true, y_pred, k=50)
        if 'mrr' in self.metric_map:
            result['mrr'] = self.calculate_mrr(y_true, y_pred)
        return result

    def batch_evaluate(
            self,
            y_true: list[float],
            y_preds: dict[str, list[float]],
            decision: list[int],
            decision_preds: dict[str, list[int]]
    ) -> dict[str, dict[str, Any]]:
        results = {}
        for name, y_pred, decision_pred in zip(y_preds.keys(), y_preds.values(), decision_preds.values()):
            results[name] = self.evaluate(y_true, y_pred, decision, decision_pred)
        return results

