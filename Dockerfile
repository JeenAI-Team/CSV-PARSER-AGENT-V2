FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory for file uploads (OpenShift compatible)
RUN mkdir -p /app/temp && chmod -R 777 /app/temp

# OpenShift runs containers as arbitrary UID, so make /app writable
RUN chmod -R g+rwX /app

# Expose port
EXPOSE 8000

# Use non-root user (OpenShift will override UID but keep GID)
USER 1001

# Run the application
CMD ["python", "main.py"]
