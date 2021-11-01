"""Script to start simple sem seg model evaluation"""

import warnings

import typer

from articles_matching.cli.articles_matching_cli import start_articles_matching

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    typer.run(start_articles_matching)
