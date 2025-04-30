FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt

COPY . /app
ENV FLASK_APP=/app/main.py

EXPOSE 5000

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
