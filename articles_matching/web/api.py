"""Module with API"""

import fastapi
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount(
    path='/static',
    app=StaticFiles(directory='articles_matching/web/static'),
    name='static',
)

templates = Jinja2Templates(directory='articles_matching/web/templates')


@app.get(path='/', response_class=RedirectResponse)
def root():
    return RedirectResponse(
        '/home',
    )


@app.get(path='/home', response_class=HTMLResponse)
def main_page(request: Request) -> Response:
    return templates.TemplateResponse('index.html', {'request': request})


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
    return templates.TemplateResponse(
        'search_result.html', {'request': request, 'query_text': query_text}
    )
