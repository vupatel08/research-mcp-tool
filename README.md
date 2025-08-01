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

A clean, simple MCP server that provides research inference utilities with no external dependencies. Self-contained server that extracts research metadata from paper URLs, repository links, or research names using embedded inference logic.

## Features

- **Author inference** from papers and repositories
- **Cross-platform resource discovery** (papers, code, models, datasets)
- **Research metadata extraction** (names, dates, licenses, organizations)
- **URL classification** and relationship mapping
- **Comprehensive research ecosystem analysis**

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
- GitHub repository URLs (https://github.com/...)
- HuggingFace model/dataset/space URLs
- Research paper titles and project names
- Project page URLs

## Environment Variables

- `HF_TOKEN` - Hugging Face API token (required)
- `GITHUB_AUTH` - GitHub API token (optional, enables enhanced GitHub integration)

## Usage

The server automatically launches as an MCP server when run. All inference functions are exposed as MCP tools for seamless integration with Claude and other AI assistants.