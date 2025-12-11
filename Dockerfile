# Base Image: Lightweight Python 3.10
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose ports for API (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Environment variable to ensure outputs are sent straight to terminal
ENV PYTHONUNBUFFERED=1

# Default Command: Start the FastAPI Microservice
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]