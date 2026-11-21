
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir \
    fastapi==0.165.0 \
    uvicorn==0.54.0 \
    pandas==3.1.1 \
    numpy \
    scikit-learn==1.9.1 \
    joblib==1.6.0

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY models/bus_delay_model.joblib ./models/bus_delay_model.joblib

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
