"""Module with base predictions"""

from typing import Dict, List, Tuple, Union

from elasticsearch import Elasticsearch, helpers


class BasePredictor:
    def __init__(self, url: str = 'localhost:9200') -> None:
        self.elastic = Elasticsearch(url)

        self._es_index = 'es-index'

    def add_texts_with_id(self, texts_with_id: List[Tuple[int, str]]) -> None:
        insert_data = [
            {
                '_index': self._es_index,
                'id': idx,
                'text': curr_word,
            }
            for idx, curr_word in texts
        ]
        helpers.bulk(self.elastic, insert_data)
        self._refresh_index()

    def clean_all_data(self) -> None:
        self.elastic.indices.delete(index=self._es_index)
        self.elastic.indices.create(index=self._es_index)
        self._refresh_index()

    def predict(self, query: str) -> List[Tuple[float, Dict[str, Union[int, str]]]]:
        query_body = {"bool": {"must": {"match": {"text": query}}}}
        result = self.elastic.search(index=self._es_index, query=query_body)
        prediction = self._process_prediction(prediction=result)

        return prediction

    @staticmethod
    def _process_prediction(
        prediction: Dict,
    ) -> List[Tuple[float, Dict[str, Union[int, str]]]]:
        processed_prediction = []

        for hit in prediction['hits']['hits']:
            curr_prediction = (hit['_score'], hit['_source'])
            processed_prediction.append(curr_prediction)

        return processed_prediction

    def _refresh_index(self) -> None:
        self.elastic.indices.refresh(index=self._es_index)


if __name__ == '__main__':
    base_predictor = BasePredictor()

    texts = [
        (1, 'Any idea why? Has the api changed for 2.1.1 for python?'),
        (2, 'From the docs, use this notation'),
        (3, 'If you have a document object (model) and you'),
    ]
    # base_predictor.add_texts_with_id(texts=texts)
    # base_predictor.clean_all_data()

    _res = base_predictor.predict(query='the docs')
    print(_res)
