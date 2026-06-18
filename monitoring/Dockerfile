FROM python:3.12-slim

WORKDIR /app

COPY monitoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY monitoring/predicted_metrics_exporter.py .
COPY models ./models
COPY data/processed/model1_dataset ./data/processed/model1_dataset
COPY data/processed/model2_dataset ./data/processed/model2_dataset

EXPOSE 8000

CMD ["python", "predicted_metrics_exporter.py"]