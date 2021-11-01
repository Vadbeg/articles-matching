"""Module with RSS parser"""

from typing import List

import wikipedia
import wikipedia.exceptions

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

    def get_article_by_name(self, name: str) -> WikiArticle:
        wiki_page = self._get_wiki_page_by_name(name=name)
        article = WikiArticle(wikipedia_page=wiki_page)

        return article

    @staticmethod
    def _get_wiki_page_by_name(name: str) -> wikipedia.WikipediaPage:
        try:
            wiki_page: wikipedia.WikipediaPage = wikipedia.page(
                title=name, auto_suggest=False
            )
        except wikipedia.exceptions.DisambiguationError as exc:
            if len(exc.options) <= 0:
                raise ValueError('No article!')

            wiki_page = wikipedia.page(title=exc.options[0], auto_suggest=False)

        return wiki_page
