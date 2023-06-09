FROM python:3.9

WORKDIR /src

COPY python/iris_classification.py .
COPY python/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


CMD ["python","iris_classification.py"]
