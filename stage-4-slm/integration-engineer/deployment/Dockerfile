# Offline Production Container for Clinical SLM Integration
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build-essential for llama-cpp OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Copy application source, frontend, and artifacts
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY runtime/ ./runtime/
COPY artifacts/ ./artifacts/
COPY cli.py .

# Expose local FastAPI port
EXPOSE 8000

# Strict offline environment declaration
ENV OFFLINE_MODE=true
ENV PYTHONUNBUFFERED=1

# Run FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
