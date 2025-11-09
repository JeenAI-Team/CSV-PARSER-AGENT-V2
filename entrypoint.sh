#!/bin/bash
set -e

# Display configuration
echo "=========================================="
echo "CSV Parser Agent - Starting"
echo "=========================================="
echo "Model: ${GEMMA_3_DEPLOYMENT:-vllm-google/gemma-3-12b-it}"
echo "vLLM Endpoint: ${VLLM_ENDPOINT:-not-set}"
echo "Server Port: ${SERVER_PORT:-8000}"
echo "Agent to vLLM Timeout: ${AGENT_TO_VLLM_TIMEOUT:-540}s"
echo "Max Retries: ${MAX_RETRIES:-6}"
echo "Temperature: ${TEMPERATURE:-0.1}"
echo "=========================================="

# Create temp directory if not exists
mkdir -p /app/temp

# Start the application
exec python main.py
