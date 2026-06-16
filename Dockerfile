FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose API port
EXPOSE 8000

# Run as non-root user
RUN useradd -m app && chown -R app:app /app
USER app

# Start: run migration first, then start API
CMD ["sh", "-c", "alembic upgrade head && uvicorn api.server:app --host 0.0.0.0 --port 8000"]
