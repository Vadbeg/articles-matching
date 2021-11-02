"""Module with API"""

import fastapi
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from articles_matching.modules.logic import Logic

app = FastAPI()
app.mount(
    path='/static',
    app=StaticFiles(directory='articles_matching/web/static'),
    name='static',
)

templates = Jinja2Templates(directory='articles_matching/web/templates')


logic = Logic()


@app.get(path='/articles', response_class=JSONResponse)
def load_articles():
    logic.load_random_articles(num_of_articles=10, verbose=True)

    return JSONResponse(content={}, status_code=fastapi.status.HTTP_201_CREATED)


@app.get(path='/train', response_class=JSONResponse)
def train_model():
    try:
        logic.train_model()
    except ValueError as exc:
        return JSONResponse(
            content=str(exc), status_code=fastapi.status.HTTP_404_NOT_FOUND
        )

    return JSONResponse(content={}, status_code=fastapi.status.HTTP_200_OK)


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
