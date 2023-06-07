FROM python:3.8

WORKDIR /src

COPY python/iris_classification.py .
COPY python/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python","iris_classification.py"]
