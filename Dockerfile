FROM python:3.12-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/tmp \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code and entrypoint
COPY . .
COPY entrypoint.sh /entrypoint.sh

# OpenShift: arbitrary UID support (gid 0 with group-writable paths)
RUN dos2unix /entrypoint.sh \
    && chmod +x /entrypoint.sh \
    && mkdir -p /app/temp /app/.cache /tmp \
    && chgrp -R 0 /app /tmp /entrypoint.sh \
    && chmod -R g=u /app /tmp \
    && chmod g+rx /entrypoint.sh

# Non-root; OpenShift may override UID, but gid 0 remains
USER 1001:0

# Expose port
EXPOSE 8000

# Run the application
CMD ["/bin/sh", "/entrypoint.sh"]
