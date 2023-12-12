FROM python:3.9.7-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "./"]

EXPOSE 10000

ENTRYPOINT ["gunicorn", "--timeout", "20000", "--bind=0.0.0.0:10000", "predict:app"]