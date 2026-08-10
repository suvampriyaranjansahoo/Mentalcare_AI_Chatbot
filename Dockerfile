FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app/ ./flask_app/
COPY data/train.csv ./data/train.csv
COPY artifacts/models/final_model.joblib ./artifacts/models/final_model.joblib
COPY evaluation/faq_results.json ./evaluation/faq_results.json

ENV FLASK_DEBUG=False
ENV PORT=5000
EXPOSE 5000

CMD ["python", "flask_app/app.py"]
