"""Module with building of vectors for search"""


from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse
import sklearn.metrics.pairwise
from sklearn.feature_extraction.text import TfidfVectorizer


class VectorModel:
    def __init__(self):
        self.tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer()

        self._corpus_sparse_vectors: Optional[scipy.sparse.csr_matrix] = None

    @property
    def corpus_sparse_vectors(self) -> scipy.sparse.csr_matrix:
        if self._corpus_sparse_vectors is None:
            raise ValueError('Model is not trained yet')

        return self._corpus_sparse_vectors

    def train(self, corpus: List[str]) -> None:
        corpus_sparse_vectors = self.tfidf_vectorizer.fit_transform(
            raw_documents=corpus
        )
        self._corpus_sparse_vectors = corpus_sparse_vectors

    def predict(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        query_sparse_vector = self.tfidf_vectorizer.transform(raw_documents=[query])

        cos_sim: np.ndarray = sklearn.metrics.pairwise.cosine_similarity(
            query_sparse_vector, self.corpus_sparse_vectors
        )

        top_text_indexes = self._get_sorted_idxs(arr=cos_sim)
        top_text_indexes = top_text_indexes[:top_n]

        top_cos_values = cos_sim[:, top_text_indexes][0].tolist()

        top_text_id_cos_val = [
            (curr_text_id, curr_cos_value)
            for curr_text_id, curr_cos_value in zip(top_text_indexes, top_cos_values)
        ]

        return top_text_id_cos_val

    @staticmethod
    def _get_sorted_idxs(arr: np.ndarray) -> List[int]:
        top_indexes = arr.argsort(axis=1).tolist()
        top_indexes = top_indexes[0][::-1]

        return top_indexes


if __name__ == '__main__':
    vector_model = VectorModel()

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    vector_model.train(corpus=corpus)

    query = 'document first'
    idxs_and_cos_vals = vector_model.predict(query=query, top_n=2)

    print(f'QUERY: {query}')
    for idx, cos_value in idxs_and_cos_vals:
        print(f'{corpus[idx]} --> {cos_value}')
