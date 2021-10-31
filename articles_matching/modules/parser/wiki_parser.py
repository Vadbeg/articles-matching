"""Module with RSS parser"""

from typing import List

import wikipedia

from articles_matching.modules.parser.wiki_article import WikiArticle


class WikiParser:
    def __init__(self):
        pass

    def get_random_articles(self, num_of_articles: int = 1) -> List[WikiArticle]:
        article_names = wikipedia.random(pages=num_of_articles)
        all_articles: List[WikiArticle] = []

        for curr_article_name in article_names:
            article = self.get_article_by_name(name=curr_article_name)
            all_articles.append(article)

        return all_articles

    @staticmethod
    def get_article_by_name(name: str) -> WikiArticle:
        wiki_page: wikipedia.WikipediaPage = wikipedia.page(title=name)
        article = WikiArticle(wikipedia_page=wiki_page)

        return article
