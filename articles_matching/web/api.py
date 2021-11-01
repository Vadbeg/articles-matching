"""Module with API"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount(
    path='/static',
    app=StaticFiles(directory='articles_matching/web/static'),
    name='static',
)

templates = Jinja2Templates(directory='articles_matching/web/templates')


@app.get(path='/', response_class=HTMLResponse)
def main_page(request: Request) -> Response:
    return templates.TemplateResponse('index.html', {'request': request})
