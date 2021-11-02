"""Module with RSS parser"""

from typing import List, Optional

import wikipedia
import wikipedia.exceptions

from articles_matching.modules.parser.wiki_article import WikiArticle


class WikiParser:
    def __init__(self):
        pass

    def get_random_articles(self, num_of_articles: int = 1) -> List[WikiArticle]:
        article_names = wikipedia.random(pages=num_of_articles)
        all_articles: List[WikiArticle] = []

        if isinstance(article_names, (str, int)):
            article_names = [article_names]

        for curr_article_name in article_names:
            article = self.get_article_by_name(name=curr_article_name)
            if article:
                all_articles.append(article)

        return all_articles

    def get_article_by_name(self, name: str) -> Optional[WikiArticle]:
        wiki_page = self._get_wiki_page_by_name(name=name)

        if wiki_page is None:
            return None

        article = WikiArticle(wikipedia_page=wiki_page)

        return article

    @staticmethod
    def _get_wiki_page_by_name(name: str) -> Optional[wikipedia.WikipediaPage]:
        try:
            wiki_page: wikipedia.WikipediaPage = wikipedia.page(
                title=name, auto_suggest=False
            )
        except wikipedia.exceptions.DisambiguationError as exc:
            if len(exc.options) <= 0:
                raise ValueError('No article!')

            wiki_page = wikipedia.page(title=exc.options[0], auto_suggest=False)
        except Exception as exc:
            print(f'No such page: \n{exc}')
            return None

        return wiki_page
