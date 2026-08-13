FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app/ ./flask_app/
COPY app/ ./app/
COPY data/train.csv ./data/train.csv
COPY artifacts/models/final_model.joblib ./artifacts/models/final_model.joblib
COPY evaluation/faq_results.json ./evaluation/faq_results.json

ENV FLASK_DEBUG=False
ENV PORT=5000
EXPOSE 5000

RUN useradd --create-home appuser && chown -R appuser:appuser /app/data
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import os; from urllib.request import urlopen; urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '5000')}/health\", timeout=3)"
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-5000} flask_app.app:app"]
