# Dockerfile
# pull the official docker image
FROM python:3.11.1-slim 
# set work directory
WORKDIR . ##/app
# set env variables
ENV PYTHONDONTWRITEBYTECODE 1 
ENV PYTHONUNBUFFERED 1
# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
# copy project
COPY . .

ENTRYPOINT gunicorn --worker-class uvicorn.workers.UvicornWorker \
  frontend_server:app \
  --workers 3 \
  --bind 0.0.0.0:8090 \
  --log-level info

