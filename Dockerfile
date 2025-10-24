
FROM python:3.9-slim

# 
WORKDIR /app

# requircopyements and install dependancies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy  source code
COPY src/ ./src/
COPY models/ ./models/
COPY api.py .

# environnement variables for  Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

#  port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

