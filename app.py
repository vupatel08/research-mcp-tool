"""
Research Tracker MCP Server

A clean, simple MCP server that provides research inference utilities.
Exposes functions to infer research metadata from paper URLs, repository links,
or research names using embedded inference logic.

Key Features:
- Author inference from papers and repositories
- Cross-platform resource discovery (papers, code, models, datasets)
- Research metadata extraction (names, dates, licenses, organizations)
- URL classification and relationship mapping
- Comprehensive research ecosystem analysis

All functions are optimized for MCP usage with clear type hints and docstrings.
"""

import logging
from config import logger
from ui import demo

if __name__ == "__main__":
    logger.info("Starting Research Tracker MCP Server")
    demo.launch(mcp_server=True, share=False)
