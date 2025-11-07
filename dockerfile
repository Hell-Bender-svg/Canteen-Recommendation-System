# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Expose FastAPI port
EXPOSE 8000

# DEFAULT: Run the ML recommender service
# If you want to run chat_api_service instead, see notes below
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
