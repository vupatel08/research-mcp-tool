---
title: Research Tracker MCP
emoji: 🔬
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
---

# Research Tracker MCP Server

A robust, performant MCP (Model Context Protocol) server that provides research inference utilities following MCP best practices. This server extracts research metadata from paper URLs, repository links, or research names using intelligent scraping and API integration.

## Features

- **Author inference** from papers and repositories
- **Cross-platform resource discovery** (papers, code, models, datasets)
- **Research metadata extraction** (names, dates, licenses)
- **URL classification** and relationship mapping
- **Comprehensive research ecosystem analysis**
- **Rate limiting** to prevent API abuse
- **Request caching** with TTL for performance
- **Robust error handling** with typed exceptions
- **Security validation** for all URLs
- **Retry logic** with exponential backoff

## Available MCP Tools

All functions are optimized for MCP usage with clear type hints and docstrings:

- `infer_authors` - Extract author names from papers and repositories
- `infer_paper_url` - Find associated research paper URLs
- `infer_code_repository` - Discover code repository links
- `infer_research_name` - Extract research project names
- `classify_research_url` - Classify URL types (paper/code/model/etc.)
- `infer_organizations` - Identify affiliated organizations
- `infer_publication_date` - Extract publication dates
- `infer_model` - Find associated HuggingFace models
- `infer_dataset` - Find associated HuggingFace datasets
- `infer_space` - Find associated HuggingFace spaces
- `infer_license` - Extract license information
- `find_research_relationships` - Comprehensive research ecosystem analysis

## Input Support

- arXiv paper URLs (https://arxiv.org/abs/...)
- HuggingFace paper URLs (https://huggingface.co/papers/...) - **Preferred over arXiv for better resource discovery**
- GitHub repository URLs (https://github.com/...)
- HuggingFace model/dataset/space URLs
- Research paper titles and project names
- Project page URLs (github.io)

## MCP Best Practices Implementation

This server follows official MCP best practices:

1. **Security**: URL validation, domain allowlisting, input sanitization
2. **Performance**: Request caching, rate limiting, connection pooling
3. **Reliability**: Retry logic, graceful error handling, timeout management
4. **Documentation**: Comprehensive docstrings with examples for all tools
5. **Error Handling**: Typed exceptions for different failure scenarios

## Environment Variables

- `HF_TOKEN` - Hugging Face API token (required)
- `GITHUB_AUTH` - GitHub API token (optional, enables enhanced GitHub integration)

## Usage

The server automatically launches as an MCP server when run. All inference functions are exposed as MCP tools for seamless integration with Claude and other AI assistants.

### Example

Test with the 3D Arena paper:
```
Input: https://arxiv.org/abs/2506.18787
Finds: dataset (dylanebert/iso3d), space (dylanebert/LGM-tiny), and more
```

### Rate Limits

- 30 requests per minute per tool
- Automatic caching reduces duplicate requests
- Graceful error messages when limits exceeded

### Error Handling

The server provides clear error messages:
- `ValidationError`: Invalid or malicious URLs
- `ExternalAPIError`: External service failures
- `MCPError`: Rate limiting or other MCP issues