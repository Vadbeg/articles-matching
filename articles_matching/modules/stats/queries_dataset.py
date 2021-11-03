"""Module with dataset"""

from collections import Counter
from itertools import combinations
from typing import List, Tuple


class QueriesDataset:
    def __init__(self, texts: List[str], top_n: int = 50) -> None:
        self.texts = texts

        self._top_n = top_n
        self._words_and_combinations = self._get_words_and_combinations(
            texts=texts, n=top_n
        )

    def _get_words_and_combinations(self, texts: List[str], n: int = 50) -> List[str]:
        common_words = self._get_n_most_common_words(texts=texts, n=n)
        words_combinations = [' '.join(item) for item in combinations(common_words, 2)]

        words_and_combinations = common_words + words_combinations

        return words_and_combinations

    def _get_n_most_common_words(self, texts: List[str], n: int = 50) -> List[str]:
        words_count = self._create_words_count(texts=texts, top_n=n)
        most_common_words = [curr_word[0] for curr_word in words_count]

        return most_common_words

    @staticmethod
    def _create_words_count(texts: List[str], top_n: int = 50) -> List[Tuple[str, int]]:
        joint_text = ' '.join(texts)
        words_count = Counter(joint_text.split(' '))
        words_and_count = [
            (k, v)
            for k, v in sorted(words_count.items(), key=lambda x: x[1], reverse=True)
        ]

        return words_and_count[:top_n]

    def __getitem__(self, item):
        return self._words_and_combinations[item]

    def __len__(self):
        return len(self._words_and_combinations)
