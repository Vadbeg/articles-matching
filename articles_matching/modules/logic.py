"""Module with main logic file"""


from typing import List, Tuple

from tqdm import tqdm

from articles_matching.modules.model.vector_model import VectorModel
from articles_matching.modules.parser.wiki_article import WikiArticle
from articles_matching.modules.parser.wiki_parser import WikiParser


class Logic:
    def __init__(self):
        self.wiki_parser = WikiParser()
        self.vector_model = VectorModel()

    def load_random_articles(
        self, num_of_articles: int = 1, verbose: bool = False
    ) -> List[WikiArticle]:
        articles = self.wiki_parser.get_random_articles(num_of_articles=num_of_articles)

        for curr_article in tqdm(
            articles, disable=not verbose, postfix='Loading articles'
        ):
            curr_article.load_content()

        return articles

    def load_articles_and_train_model(
        self, num_of_articles: int = 1, verbose: bool = False
    ) -> List[WikiArticle]:
        articles = self.load_random_articles(
            num_of_articles=num_of_articles, verbose=verbose
        )
        articles_text = [curr_article.content for curr_article in articles]

        self.vector_model.train(corpus=articles_text)

        return articles

    def get_prediction(
        self, query: str, articles: List[WikiArticle], top_n: int = 1
    ) -> List[Tuple[WikiArticle, float]]:
        top_text_id_cos_val = self.vector_model.predict(query=query, top_n=top_n)

        predictions = [
            (articles[text_id], cos_val) for text_id, cos_val in top_text_id_cos_val
        ]

        return predictions
