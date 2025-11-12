# Backend Dockerfile for Options Desk API
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies in separate layer
RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc g++
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r backend-requirements.txt

# Copy source code
COPY src/ ./src/
COPY backend/ ./backend/

# Install the options_desk package
RUN pip install -e .

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run the application
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
