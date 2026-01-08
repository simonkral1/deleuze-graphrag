FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git lfs
RUN git lfs install

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn flask python-dotenv

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "2", "--threads", "4", "app:app"]
