"""Module with main logic file"""


import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from articles_matching.modules.model.vector_model import VectorModel
from articles_matching.modules.parser.wiki_article import WikiArticle
from articles_matching.modules.parser.wiki_parser import WikiParser


class Logic:
    def __init__(self):
        self.wiki_parser = WikiParser()
        self.vector_model = VectorModel()

        self._articles: List[WikiArticle] = []

    @property
    def articles(self) -> Optional[List[WikiArticle]]:
        return self._articles

    def remove_all_articles(self) -> None:
        self._articles = []

    def load_random_articles(
        self, num_of_articles: int = 1, verbose: bool = False
    ) -> List[WikiArticle]:
        articles = self.wiki_parser.get_random_articles(num_of_articles=num_of_articles)

        for curr_article in tqdm(
            articles, disable=not verbose, postfix='Loading articles'
        ):
            curr_article.load_content()

        self._articles.extend(articles)

        return articles

    def train_model(self):
        if len(self._articles) > 0:
            articles_text = [curr_article.content for curr_article in self._articles]
            self.vector_model.train(corpus=articles_text)
        else:
            raise ValueError('No articles downloaded')

    def get_prediction(
        self, query: str, top_n: int = 1
    ) -> List[Tuple[WikiArticle, float]]:
        top_text_id_cos_val = self.vector_model.predict(query=query, top_n=top_n)

        predictions = [
            (self._articles[text_id], cos_val)
            for text_id, cos_val in top_text_id_cos_val
        ]

        return predictions

    @staticmethod
    def load_logic(path: Union[str, Path]) -> 'Logic':
        path = Path(path)
        with path.open(mode='rb') as file:
            logic = pickle.load(file=file)

        return logic

    @staticmethod
    def save_logic(logic: 'Logic', path: Union[str, Path]) -> None:
        path = Path(path)
        with path.open(mode='wb') as file:
            pickle.dump(obj=logic, file=file)
