#!/bin/sh

uvicorn articles_matching.web.api:app --reload --host 0.0.0.0 --port 8000
