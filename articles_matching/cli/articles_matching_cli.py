"""CLI for articles matching"""

from typing import List, Tuple

import typer

from articles_matching.modules.logic import Logic
from articles_matching.modules.parser.wiki_article import WikiArticle


def start_articles_matching(
    query: str = typer.Option(..., help='Query to search in articles'),
    num_of_articles: int = typer.Option(
        ..., help='Number of articles to download for search'
    ),
    top_n: int = typer.Option(..., help='Number of articles to show in result'),
    verbose: bool = typer.Option(default=True, is_flag=True),
) -> None:
    logic = Logic()

    articles = logic.load_articles_and_train_model(
        num_of_articles=num_of_articles, verbose=verbose
    )
    prediction = logic.get_prediction(query=query, articles=articles, top_n=top_n)

    _print_prediction(prediction=prediction)


def _print_prediction(prediction: List[Tuple[WikiArticle, float]]) -> None:
    for curr_article, curr_cos_value in prediction:
        print(f'{curr_article.title}: {curr_cos_value}\n{curr_article.url}')
        print('-' * 10)
