# syntax=docker/dockerfile:1

FROM python:3.9.17-slim-bullseye

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN python3.9 -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -U flask-cors

RUN pip install --upgrade sqlalchemy  # required to resolve ImportError: cannot import name 'URL' from 'sqlalchemy'
RUN pip install setuptools_scm jpype1 # install pre-reqs
RUN pip install sutime

COPY . /python-docker
ENV FLASK_APP=/python-docker/main.py
EXPOSE 5000

CMD [ "python3.9", "-m" , "flask", "run", "--host=0.0.0.0"]
