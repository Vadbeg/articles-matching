FROM python:3.9

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
RUN chmod +x start_api.sh

ENTRYPOINT ./start_api.sh
