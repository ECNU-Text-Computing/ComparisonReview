from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    def __init__(self):
        self.metric_map = None

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_evaluate(self, *args, **kwargs):
        pass

    def print_result(
            self,
            result: dict[str, Any],
            orientation: str = "vertical"
    ) -> None:
        if orientation == "horizontal":
            formated_metric_fullnames = [fullname if len(fullname) >= 6 else fullname + (6 - len(fullname)) * " "
                                         for fullname in list(self.metric_map.values())]
            print("| Name           | " + " | ".join(formated_metric_fullnames) + " |")
            print("|----------------|-" + "-|-".join("-" * len(metric) for metric in formated_metric_fullnames) + "-|")
            formated_scores = [f"{result[metric]:.4f}{(len(fullname) - 6) * ' '}"
                               for metric, fullname in zip(self.metric_map.keys(), formated_metric_fullnames)]
            print("| Value          | " + " | ".join(formated_scores) + " |")

        elif orientation == "vertical":
            print(f"| Metric         | Value  |")
            print(f"|----------------|--------|")
            for metric, fullname in self.metric_map.items():
                print(f"| {fullname:<14} | {result[metric]:.4f} |")
        print()
        return None

    def print_results(
            self,
            results: dict[str, dict[str, Any]],
            orientation: str = "vertical",
            mark: bool = False
    ) -> None:
        if orientation == "horizontal":
            max_name_width = max([len(fullname) for fullname in results.keys()])
            formated_metric_fullnames = [fullname if len(fullname) >= 6 else fullname + (6 - len(fullname)) * " "
                                         for fullname in list(self.metric_map.values())]
            print(f"| Name          {(max_name_width - 14) * ' '} | " + " | ".join(formated_metric_fullnames) + " |")
            print(f"|---------------{(max_name_width - 14) * '-'}-|-" +
                  "-|-".join("-" * len(metric) for metric in formated_metric_fullnames) + "-|")
            for name, result in results.items():
                formated_scores = []
                for metric, fullname in zip(self.metric_map.keys(), formated_metric_fullnames):
                    values = [results[n][metric] for n in results.keys()]
                    max_value = max(values)
                    sorted_values = sorted(set(values), reverse=True)
                    second_max_value = sorted_values[1] if len(sorted_values) > 1 else None
                    score_str = f"{result[metric]:.4f}"
                    if mark:
                        if result[metric] == max_value:
                            score_str = f"**{score_str}**"
                        elif second_max_value is not None and result[metric] == second_max_value:
                            score_str = f"<u>{score_str}</u>"
                    score_str += (len(fullname) - 6) * " "
                    formated_scores.append(score_str)
                print(f"| {name:<14}{(max_name_width - max(len(name), 14)) * ' '} | " + " | ".join(
                    formated_scores) + " |")

        elif orientation == "vertical":
            formeted_names = [name if len(name) >= 6 else name + (6 - len(name)) * " " for name in results.keys()]
            print("| Metric         | " + " | ".join(formeted_names) + " |")
            print("|----------------|-" + "-|-".join(["-" * len(name) for name in formeted_names]) + "-|")
            for metric, fullname in self.metric_map.items():
                values = [results[name][metric] for name in results.keys()]
                max_value = max(values)
                sorted_values = sorted(set(values), reverse=True)
                second_max_value = sorted_values[1] if len(sorted_values) > 1 else None
                formated_scores = []
                for name in results.keys():
                    score_str = f"{results[name][metric]:.4f}"
                    if mark:
                        if results[name][metric] == max_value:
                            score_str = f"**{score_str}**"
                        elif second_max_value is not None and results[name][metric] == second_max_value:
                            score_str = f"<u>{score_str}</u>"
                    score_str += (len(name) - 6) * " "
                    formated_scores.append(score_str)
                print(f"| {fullname:<14} | " + " | ".join(formated_scores) + " |")
        print()
        return None
