---
title: Research Tracker MCP
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
license: mit
---

# Research Tracker MCP Server

A Gradio-based MCP (Model Context Protocol) server that provides research inference utilities for AI assistants and tools. This server offers public-facing APIs to extract and infer research metadata from various sources like papers, repositories, and project pages.

## Features

### MCP Tools Available

- **`infer_authors`**: Extract author names from research papers, repositories, or project URLs
- **`infer_paper_url`**: Find associated research papers from GitHub repos, project pages, or partial information
- **`infer_code_repository`**: Locate code repositories from paper URLs or project information
- **`infer_research_name`**: Extract formal paper/project titles from various inputs
- **`classify_research_url`**: Classify URLs as Paper, Code, Model, Dataset, Space, or Project

### Supported Input Types

- **arXiv papers**: `https://arxiv.org/abs/2010.11929`
- **GitHub repositories**: `https://github.com/google-research/vision_transformer`
- **Hugging Face resources**: Models, Datasets, Spaces, Papers
- **Project pages**: GitHub Pages, personal websites
- **Research titles**: Natural language paper titles

## Usage

### As MCP Server

This space can be used as an MCP server by AI assistants that support the MCP protocol. Configure your MCP client with:

```json
{
  "mcpServers": {
    "research-tracker": {
      "url": "https://YOUR_SPACE_NAME.hf.space/gradio_api/mcp/sse"
    }
  }
}
```

### Web Interface

The space also provides a web interface for testing the inference functions directly in your browser.

## Architecture

This MCP server delegates all inference logic to the [Research Tracker Backend](https://huggingface.co/spaces/dylanebert/research-tracker-backend) to ensure consistency and avoid code duplication. It serves as a public-facing interface for research inference utilities without requiring database access.

## Examples

### Infer Authors from arXiv Paper
```python
infer_authors("https://arxiv.org/abs/2010.11929")
# Returns: ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", ...]
```

### Find Paper from GitHub Repository
```python
infer_paper_url("https://github.com/google-research/vision_transformer")
# Returns: "https://arxiv.org/abs/2010.11929"
```

### Classify URL Type
```python
classify_research_url("https://huggingface.co/google/vit-base-patch16-224")
# Returns: "Model"
```

## Requirements

- Python 3.11+
- Gradio with MCP support
- Internet connection for backend API calls

## Development

The server is built with:
- **Gradio**: Web interface and MCP protocol support
- **Requests**: HTTP client for backend communication
- **Backend Integration**: Calls to research-tracker-backend API

## License

MIT License - Feel free to use and modify for your research needs.
