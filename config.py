"""
Configuration for the Data Analysis Agent
Compatible with data-analysis-api environment variables
"""

import os
from dotenv import load_dotenv

load_dotenv()


# Server Configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SERVER_TIMEOUT = int(os.getenv("SERVER_TIMEOUT", "999"))
API_KEY = os.getenv("API_KEY", "")

# Model Configuration
GEMMA_3_DEPLOYMENT = os.getenv("GEMMA_3_DEPLOYMENT", "vllm-google/gemma-3-12b-it")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "")

# Extract model deployment from GEMMA_3_DEPLOYMENT (remove vllm- prefix)
if GEMMA_3_DEPLOYMENT.startswith("vllm-"):
    MODEL_DEPLOYMENT = GEMMA_3_DEPLOYMENT[5:]  # Remove 'vllm-' prefix
else:
    MODEL_DEPLOYMENT = GEMMA_3_DEPLOYMENT

# Agent Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))


def get_agent_config():
    """Get configuration dictionary for agent initialization."""
    if not VLLM_ENDPOINT:
        raise ValueError("VLLM_ENDPOINT environment variable is required")
    
    return {
        "vllm_endpoint": VLLM_ENDPOINT,
        "model_deployment": MODEL_DEPLOYMENT,
        "max_retries": MAX_RETRIES,
        "temperature": TEMPERATURE
    }


def validate_config():
    """Validate required configuration."""
    if not API_KEY:
        raise ValueError("API_KEY environment variable is required")
    if not VLLM_ENDPOINT:
        raise ValueError("VLLM_ENDPOINT environment variable is required")
    if not GEMMA_3_DEPLOYMENT:
        raise ValueError("GEMMA_3_DEPLOYMENT environment variable is required")
    
    return True

