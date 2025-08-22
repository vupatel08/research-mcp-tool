"""
Research Tracker MCP Server - Configuration Module

Configuration constants, exception classes, and logging setup.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
CACHE_TTL = 3600  # 1 hour cache TTL
MAX_URL_LENGTH = 2048
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_CALLS = 30  # max calls per window

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
HUGGINGFACE_API_BASE = "https://huggingface.co/api"
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_AUTH = os.environ.get("GITHUB_AUTH")

# Allowed domains for security
ALLOWED_DOMAINS = {
    "arxiv.org",
    "huggingface.co",
    "github.com",
    "github.io",
    "raw.githubusercontent.com"
}

if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables")


class MCPError(Exception):
    """Base exception for MCP-related errors"""
    pass


class ValidationError(MCPError):
    """Input validation error"""
    pass


class ExternalAPIError(MCPError):
    """External API call error"""
    pass
