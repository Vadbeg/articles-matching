"""Module with stats calculation"""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import plotly.express as px
from tqdm import tqdm

from articles_matching.modules.model.vector_model import VectorModel
from articles_matching.modules.parser.wiki_parser import WikiParser
from articles_matching.modules.stats.base_predictions import BasePredictor
from articles_matching.modules.stats.metrics_calculation import MetricCalculator
from articles_matching.modules.stats.queries_dataset import QueriesDataset


class StatsCalculator:
    f_score = 'f_score'
    recall = 'recall'
    precision = 'precision'
    accuracy = 'accuracy'
    error = 'error'
    avg_precision_score = 'avg_precision_score'
    precisions_for_plot = 'precisions_for_plot'
    recalls_for_plot = 'recalls_for_plot'

    def __init__(
        self,
        base_predictor_url: str = 'localhost:9200',
        num_of_articles: int = 100,
        num_of_top_words: int = 100,
        verbose: bool = True,
    ):
        self.base_predictor = BasePredictor(url=base_predictor_url)
        self.vector_model = VectorModel()

        self.wiki_parser = WikiParser()
        self.metrics_calculator = MetricCalculator()

        self.num_of_articles = num_of_articles
        self.num_of_top_words = num_of_top_words
        self.verbose = verbose

    def models_validation(self) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        texts = self._load_texts()
        self._train_models(texts=texts)

        query_dataset = QueriesDataset(texts=texts, top_n=self.num_of_top_words)

        metrics: Dict[str, List[float]] = defaultdict(list)
        plot_metrics: Dict[str, List[List[float]]] = defaultdict(list)

        for curr_query in tqdm(
            query_dataset, postfix='Validation...', disable=not self.verbose
        ):
            base_text_ids, new_text_ids, pred_scores = self._get_predictions(
                query=curr_query
            )

            if len(base_text_ids) > 0:
                metrics = self._update_metrics(
                    metrics=metrics,
                    pred=base_text_ids,
                    true=new_text_ids,
                    pred_scores=pred_scores,
                    total_texts_num=len(texts),
                )
                plot_metrics = self._update_plot_metrics(
                    plot_metrics=plot_metrics,
                    pred=base_text_ids,
                    true=new_text_ids,
                    pred_scores=pred_scores,
                )

        processed_metrics = self._process_metrics(metrics=metrics)
        processed_plot_metrics = self._process_plot_metrics(plot_metrics=plot_metrics)

        return processed_metrics, processed_plot_metrics

    def _load_texts(self) -> List[str]:
        articles = self.wiki_parser.get_random_articles(
            num_of_articles=self.num_of_articles
        )
        texts = [
            curr_article.content
            for curr_article in tqdm(
                articles, postfix='Loading articles...', disable=not self.verbose
            )
        ]

        return texts

    @staticmethod
    def _get_texts_with_ids(texts: List[str]) -> List[Tuple[int, str]]:
        texts_with_id = [(idx, curr_text) for idx, curr_text in enumerate(texts)]

        return texts_with_id

    def _train_models(self, texts: List[str]) -> None:
        texts_with_id = self._get_texts_with_ids(texts=texts)

        self.base_predictor.add_texts_with_id(texts_with_id=texts_with_id)
        self.vector_model.train(corpus=texts)

    def _get_predictions(self, query: str) -> Tuple[List[int], List[int], List[float]]:
        base_predictions = self.base_predictor.predict(query=query)
        new_predictions = self.vector_model.predict(query=query, top_n=5)

        base_text_ids = []
        new_text_ids = []
        pred_scores = []

        for curr_base_pred, curr_new_pred in zip(base_predictions, new_predictions):
            base_text_ids.append(int(curr_base_pred[1]['id']))
            new_text_ids.append(curr_new_pred[0])
            pred_scores.append(curr_new_pred[1])

        max_pred_score = max(pred_scores)

        if max_pred_score > 0:
            pred_scores = [
                curr_pred_score / max_pred_score for curr_pred_score in pred_scores
            ]

        return base_text_ids, new_text_ids, pred_scores

    @staticmethod
    def _process_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
        processed_metrics = {}

        for metric_name, metric_values in metrics.items():
            if len(metric_values) > 0:
                if isinstance(metric_values[0], float):
                    curr_metric_value = np.mean(metric_values)
                else:
                    raise ValueError(f'Bad metrics type: {type(metric_values)}')

                processed_metrics[metric_name] = curr_metric_value
            else:
                processed_metrics[metric_name] = 0.0

        return processed_metrics

    @staticmethod
    def _process_plot_metrics(
        plot_metrics: Dict[str, List[List[float]]]
    ) -> Dict[str, List[float]]:
        processed_metrics = {}

        for metric_name, metric_values in plot_metrics.items():
            if len(metric_values) > 0:
                if isinstance(metric_values, list):
                    curr_metric_value = np.mean(metric_values, axis=0)
                else:
                    raise ValueError(f'Bad metrics type: {type(metric_values)}')

                processed_metrics[metric_name] = curr_metric_value
            else:
                processed_metrics[metric_name] = []

        return processed_metrics

    def _update_metrics(
        self,
        metrics: Dict[str, List[float]],
        pred: List[int],
        true: List[int],
        pred_scores: List[float],
        total_texts_num: int,
    ) -> Dict[str, List[float]]:
        metrics[self.f_score].append(
            self.metrics_calculator.get_f_score(
                pred=pred, true=true, total_num=total_texts_num
            )
        )
        metrics[self.recall].append(
            self.metrics_calculator.get_recall(
                pred=pred, true=true, total_num=total_texts_num
            )
        )
        metrics[self.precision].append(
            self.metrics_calculator.get_precision(
                pred=pred, true=true, total_num=total_texts_num
            )
        )
        metrics[self.accuracy].append(
            self.metrics_calculator.get_accuracy(
                pred=pred, true=true, total_num=total_texts_num
            )
        )
        metrics[self.error].append(
            self.metrics_calculator.get_error(
                pred=pred, true=true, total_num=total_texts_num
            )
        )
        metrics[self.avg_precision_score].append(
            self.metrics_calculator.get_avg_precision(
                pred=pred, true=true, pred_scores=pred_scores
            )
        )

        return metrics

    def _update_plot_metrics(
        self,
        plot_metrics: Dict[str, List[List[float]]],
        pred: List[int],
        true: List[int],
        pred_scores: List[float],
    ) -> Dict[str, List[List[float]]]:
        precisions, recalls = self.metrics_calculator.get_precision_recall_thresholds(
            pred=pred, true=true, pred_scores=pred_scores
        )
        plot_metrics[self.precisions_for_plot].append(precisions)
        plot_metrics[self.recalls_for_plot].append(recalls)

        return plot_metrics

    @staticmethod
    def get_precision_recall_plot(
        precisions: List[float], recalls: List[float]
    ) -> Dict:
        fig = px.area(
            x=precisions,
            y=recalls,
            labels=dict(x='Precision', y='Recall'),
            width=700,
            height=500,
        )
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')

        return fig.to_html(full_html=False)
