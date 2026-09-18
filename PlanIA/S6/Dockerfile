FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py inference.py model.joblib model_metadata.json ./

CMD exec gunicorn --bind :${PORT} --workers 1 --threads 8 --timeout 0 app:app --worker-class uvicorn.workers.UvicornWorker
