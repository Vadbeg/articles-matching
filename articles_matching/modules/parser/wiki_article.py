"""Module with wikipedia article wrapper"""

from typing import Optional

import wikipedia


class WikiArticle:
    def __init__(self, wikipedia_page: wikipedia.WikipediaPage):
        self.wikipedia_page = wikipedia_page

        self._content: Optional[str] = None
        self._title: Optional[str] = None
        self._url: Optional[str] = None

    @property
    def content(self) -> str:
        if self._content is None:
            self._content = self._get_article_content()

        return self._content

    @property
    def title(self) -> str:
        if self._title is None:
            self._title = self._get_article_title()

        return self._title

    @property
    def url(self) -> str:
        if self._url is None:
            self._url = self._get_article_url()

        return self._url

    def _get_article_content(self) -> str:
        content = self.wikipedia_page.content

        return content

    def _get_article_title(self) -> str:
        name = self.wikipedia_page.title

        return name

    def _get_article_url(self) -> str:
        url = self.wikipedia_page.url

        return url
