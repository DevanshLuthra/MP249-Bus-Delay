FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout 180 --retries 10 -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY params.yaml .
COPY models/model.joblib ./models/model.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
