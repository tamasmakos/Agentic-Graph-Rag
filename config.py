"""Configuration management for the application"""
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
import logging
import streamlit as st

# Load environment variables (fallback for local development)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

def get_groq_api_key() -> str:
    """Get Groq API key from Streamlit secrets or environment"""
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in secrets or environment")
    if not api_key.startswith("gsk_"):
        raise ValueError("Invalid API key format. Groq API keys should start with 'gsk_'")
    return api_key

def get_anthropic_api_key() -> str:
    """Get Anthropic API key from Streamlit secrets or environment"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in secrets or environment")
    if not api_key.startswith("sk-ant"):
        raise ValueError("Invalid API key format. Anthropic API keys should start with 'sk-ant'")
    return api_key

# Global configurations
GROQ_API_KEY = get_groq_api_key()
ANTHROPIC_API_KEY = get_anthropic_api_key()

# Text Processing Configuration
TEXT_PROCESSING_CONFIG = {
    "testing": {
        "enabled": True,  # Whether to use testing mode
        "start_idx": 0,   # Starting index for text slice
        "end_idx": 200000,  # Ending index for text slice
        "chunk_size": 1000,  # Size of each chunk
        "max_chunks": 5    # Maximum number of chunks to process
    },
    "production": {
        "enabled": False,
        "chunk_size": 2000,
        "max_chunks": None  # No limit
    }
}

# Model configurations
GROQ_MODEL = "llama-3.1-8b-instant"
CLAUDE_MODEL = "claude-3-haiku-20240307"

LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3
LLM_REQUEST_TIMEOUT = 30

# Get LLM provider from environment (default to groq)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Add rate limiting configuration
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 30,
    "max_tokens_per_minute": 7000,
    "max_request_tokens": 2000  # Reduced from 4000
}

# LLM Configuration for each provider
GROQ_CONFIG = {
    "model": GROQ_MODEL,
    "temperature": LLM_TEMPERATURE,
    "groq_api_key": GROQ_API_KEY,
    "max_tokens": 2000  # Reduced from 4096
}

ANTHROPIC_CONFIG = {
    "model": CLAUDE_MODEL,
    "temperature": LLM_TEMPERATURE,
    "anthropic_api_key": ANTHROPIC_API_KEY,
    "max_tokens": 2000  # Reduced from 4096
}

# Export this as LLM_CONFIG for backward compatibility
LLM_CONFIG = GROQ_CONFIG if LLM_PROVIDER == "groq" else ANTHROPIC_CONFIG

def get_llm():
    """Initialize LLM based on configured provider"""
    if LLM_PROVIDER == "groq":
        logger.info("Using Groq LLM provider")
        return ChatGroq(**GROQ_CONFIG)
    elif LLM_PROVIDER == "anthropic":
        logger.info("Using Anthropic LLM provider")
        return ChatAnthropic(**ANTHROPIC_CONFIG)
    else:
        raise ValueError(f"Invalid LLM provider: {LLM_PROVIDER}. Must be 'groq' or 'anthropic'")

def get_text_processing_config():
    """Get text processing configuration based on environment"""
    is_testing = os.getenv("TESTING", "true").lower() == "true"
    return TEXT_PROCESSING_CONFIG["testing" if is_testing else "production"]