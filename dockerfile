# Base image
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the API (correct module path)
CMD ["uvicorn", "ML.API.recommend_api:app", "--host", "0.0.0.0", "--port", "8000"]
