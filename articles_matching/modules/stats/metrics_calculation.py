"""Module with metrics calculation"""


import math
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

from articles_matching.modules.stats.utils import ignore_zero_division


class MetricCalculator:
    """
    ------------------------------------------------------------------
    | found_and_relevant_name     | found_and_not_relevant_name      |
    | not_found_and_relevant_name | not_found_and_not_relevant_name  |
    ------------------------------------------------------------------
    """

    found_and_relevant_name = 'found_and_relevant'
    found_and_not_relevant_name = 'found_and_not_relevant'
    not_found_and_relevant_name = 'not_found_and_relevant'
    not_found_and_not_relevant_name = 'not_found_not_relevant'

    def __init__(self):
        self.eps = 0.0000001

    def _get_values(
        self, pred: List[int], true: List[int], total_num: int
    ) -> Dict[str, int]:
        found_and_relevant_num = len(set(pred).intersection(true))
        found_and_not_relevant_num = len(set(pred) - set(true))
        not_found_and_relevant_num = len(set(true) - set(pred))

        not_found_and_not_relevant_num = (
            total_num
            - found_and_relevant_num
            - found_and_not_relevant_num
            - not_found_and_relevant_num
        )

        values = {
            self.found_and_relevant_name: found_and_relevant_num,
            self.found_and_not_relevant_name: found_and_not_relevant_num,
            self.not_found_and_relevant_name: not_found_and_relevant_num,
            self.not_found_and_not_relevant_name: not_found_and_not_relevant_num,
        }

        return values

    @ignore_zero_division
    def get_recall(self, pred: List[int], true: List[int], total_num: int) -> float:
        values = self._get_values(pred=pred, true=true, total_num=total_num)

        recall = values[self.found_and_relevant_name] / (
            values[self.found_and_relevant_name]
            + values[self.not_found_and_relevant_name]
        )

        return recall

    @ignore_zero_division
    def get_precision(self, pred: List[int], true: List[int], total_num: int) -> float:
        values = self._get_values(pred=pred, true=true, total_num=total_num)

        precision = values[self.found_and_relevant_name] / (
            values[self.found_and_relevant_name]
            + values[self.found_and_not_relevant_name]
        )

        return precision

    @ignore_zero_division
    def get_accuracy(self, pred: List[int], true: List[int], total_num: int) -> float:
        values = self._get_values(pred=pred, true=true, total_num=total_num)

        accuracy = (
            values[self.found_and_relevant_name]
            + values[self.not_found_and_not_relevant_name]
        ) / (
            values[self.found_and_relevant_name]
            + values[self.not_found_and_not_relevant_name]
            + values[self.found_and_not_relevant_name]
            + values[self.not_found_and_relevant_name]
        )

        return accuracy

    @ignore_zero_division
    def get_error(self, pred: List[int], true: List[int], total_num: int) -> float:
        values = self._get_values(pred=pred, true=true, total_num=total_num)

        accuracy = (
            values[self.found_and_not_relevant_name]
            + values[self.not_found_and_relevant_name]
        ) / (
            values[self.found_and_relevant_name]
            + values[self.not_found_and_not_relevant_name]
            + values[self.found_and_not_relevant_name]
            + values[self.not_found_and_relevant_name]
        )

        return accuracy

    @ignore_zero_division
    def get_f_score(self, pred: List[int], true: List[int], total_num: int) -> float:
        recall = self.get_recall(pred=pred, true=true, total_num=total_num)
        precision = self.get_precision(pred=pred, true=true, total_num=total_num)

        f_score = 2 * recall * precision / (recall + precision)

        return f_score

    @staticmethod
    def get_avg_precision(
        pred: List[int], true: List[int], pred_scores: List[float]
    ) -> float:
        binary_labels = [
            curr_pred == curr_true for curr_pred, curr_true in zip(pred, true)
        ]

        avg_precision = average_precision_score(
            y_true=binary_labels, y_score=pred_scores
        )

        if math.isnan(avg_precision):
            return 0.0

        return avg_precision

    @staticmethod
    def get_precision_recall_thresholds(
        pred: List[int], true: List[int], pred_scores: List[float]
    ) -> Tuple[List[float], List[float]]:
        y_true = np.float16(np.array(pred) == np.array(true))

        precision_values = []
        recall_values = []

        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for curr_threshold in thresholds:
            y_pred = np.float16(np.array(pred_scores) > curr_threshold)

            prec = precision_score(y_true=y_true, y_pred=y_pred)
            rec = recall_score(y_true=y_true, y_pred=y_pred)

            precision_values.append(prec)
            recall_values.append(rec)

        return precision_values, recall_values
