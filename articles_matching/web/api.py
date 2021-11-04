"""Module with API"""

import warnings
from pathlib import Path

import fastapi
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from articles_matching.modules.logic import Logic
from articles_matching.modules.stats_calculation import StatsCalculator

warnings.filterwarnings('ignore')

app = FastAPI()
app.mount(
    path='/static',
    app=StaticFiles(directory='articles_matching/web/static'),
    name='static',
)

templates = Jinja2Templates(directory='articles_matching/web/templates')


logic = Logic()
logic_path = Path('logic.pickle').absolute()

NUM_OF_ARTICLES = 10
NUM_OF_TOP_WORDS = 20

stats_calculator = StatsCalculator(
    base_predictor_url='localhost:9200',
    num_of_articles=NUM_OF_ARTICLES,
    num_of_top_words=NUM_OF_TOP_WORDS,
    verbose=True,
)


@app.get(path='/articles', response_class=JSONResponse)
def load_articles():
    logic.load_random_articles(num_of_articles=10, verbose=True)
    logic.save_logic(logic=logic, path=logic_path)

    return JSONResponse(content={}, status_code=fastapi.status.HTTP_201_CREATED)


@app.get(path='/articles/drop', response_class=JSONResponse)
def remove_all_articles():
    logic.remove_all_articles()
    logic.save_logic(logic=logic, path=logic_path)

    return JSONResponse(content={}, status_code=fastapi.status.HTTP_200_OK)


@app.get(path='/train', response_class=JSONResponse)
def train_model():
    try:
        logic.train_model()
        logic.save_logic(logic=logic, path=logic_path)
    except ValueError as exc:
        return JSONResponse(
            content=str(exc), status_code=fastapi.status.HTTP_404_NOT_FOUND
        )

    return JSONResponse(content={}, status_code=fastapi.status.HTTP_200_OK)


@app.get(path='/load_logic', response_class=JSONResponse)
def load_logic():
    status_code = fastapi.status.HTTP_204_NO_CONTENT

    if logic_path.exists():
        global logic
        logic = logic.load_logic(path=logic_path)
        status_code = fastapi.status.HTTP_201_CREATED

    return JSONResponse(content={}, status_code=status_code)


@app.get(path='/', response_class=RedirectResponse)
def root():
    return RedirectResponse(
        '/home',
    )


@app.get(path='/home', response_class=HTMLResponse)
def main_page(request: Request) -> Response:
    return templates.TemplateResponse('home.html', {'request': request})


@app.post(path='/home', response_class=RedirectResponse)
def redirect_search_query(
    query_text: str = Form(..., alias='query-text')
) -> RedirectResponse:
    url = (
        app.url_path_for(
            name='query',
        )
        + f'?query_text={query_text}'
    )

    return RedirectResponse(url=url, status_code=fastapi.status.HTTP_302_FOUND)


@app.get('/query', response_class=HTMLResponse)
def query(request: Request, query_text: str) -> Response:
    predictions = logic.get_prediction(query=query_text, top_n=5)

    articles = [curr_pred[0] for curr_pred in predictions]

    return templates.TemplateResponse(
        'search_result.html', {'request': request, 'articles': articles}
    )


@app.get('/all_articles', response_class=HTMLResponse)
def all_articles(request: Request) -> Response:
    articles = []

    if logic.articles:
        articles = [curr_pred for curr_pred in logic.articles]

    return templates.TemplateResponse(
        'search_result.html',
        {'request': request, 'articles': articles, 'num_of_articles': len(articles)},
    )


@app.get('/stats', response_class=HTMLResponse)
def show_stats(request: Request) -> Response:
    metrics, plot_metrics = stats_calculator.models_validation()

    precisions = plot_metrics[stats_calculator.precisions_for_plot]
    recalls = plot_metrics[stats_calculator.recalls_for_plot]

    predictions_recall_plot = stats_calculator.get_precision_recall_plot(
        precisions=precisions, recalls=recalls
    )

    metrics_to_show = [
        {'name': metric_name, 'value': metric_value}
        for metric_name, metric_value in metrics.items()
    ]

    return templates.TemplateResponse(
        'stats.html',
        {
            'request': request,
            'metrics': metrics_to_show,
            'predictions_recall_plot': predictions_recall_plot,
            'num_of_articles': NUM_OF_ARTICLES,
            'num_of_top_words': NUM_OF_TOP_WORDS,
        },
    )
