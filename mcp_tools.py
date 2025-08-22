"""
Research Tracker MCP Server - MCP Tool Functions

Public-facing MCP tool functions with rate limiting and error handling.
"""

import logging
from typing import List, Dict, Any

from config import logger
from utils import rate_limit
from inference import (
    create_row_data, infer_paper_from_row, infer_name_from_row, 
    infer_code_from_row, infer_authors_from_row, infer_date_from_row,
    infer_model_from_row, infer_dataset_from_row, infer_space_from_row,
    infer_license_from_row, infer_field_type
)

logger = logging.getLogger(__name__)


# MCP tool functions
@rate_limit("mcp_tools")
def infer_authors(input_data: str) -> List[str]:
    """
    Infer authors from research paper or project information.
    
    This tool extracts author names from:
    - arXiv papers (via API)
    - HuggingFace paper pages (via scraping)
    - GitHub repositories (via API when GITHUB_AUTH is set)
    
    Args:
        input_data (str): A URL, paper title, or other research-related input.
                         Examples:
                         - "https://arxiv.org/abs/2103.00020"
                         - "https://huggingface.co/papers/2103.00020"
                         - "https://github.com/openai/CLIP"
        
    Returns:
        List[str]: A list of author names as strings, or empty list if no authors found.
                  Example: ["Alec Radford", "Jong Wook Kim", "Chris Hallacy"]
        
    Raises:
        ValidationError: If input_data is invalid or malformed
        ExternalAPIError: If external API calls fail after retries
    """
    if not input_data or not input_data.strip():
        return []
    
    try:
        cleaned_input = input_data.strip()
        row_data = create_row_data(cleaned_input)
        authors = infer_authors_from_row(row_data)
        
        valid_authors = []
        for author in authors:
            if isinstance(author, str) and len(author.strip()) > 0:
                cleaned_author = author.strip()
                if 2 <= len(cleaned_author) <= 100:
                    valid_authors.append(cleaned_author)
        
        logger.info(f"Successfully inferred {len(valid_authors)} authors from input")
        return valid_authors
        
    except Exception as e:
        logger.error(f"Error inferring authors: {e}")
        return []


@rate_limit("mcp_tools")
def infer_paper_url(input_data: str) -> str:
    """
    Infer the paper URL from various research-related inputs.
    
    This tool finds paper URLs by:
    - Validating existing paper URLs
    - Searching GitHub repositories for paper links
    - Converting between arXiv and HuggingFace paper formats
    - Searching by paper title when provided
    
    Args:
        input_data (str): A URL, repository link, or other research-related input
                         Examples:
                         - "https://github.com/openai/CLIP"
                         - "Vision Transformer"
                         - "https://huggingface.co/spaces/example"
        
    Returns:
        str: The paper URL (typically arXiv or Hugging Face papers), or empty string if not found
             Example: "https://huggingface.co/papers/2103.00020"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = infer_paper_from_row(row_data)
        return result or ""
        
    except Exception as e:
        logger.error(f"Error inferring paper: {e}")
        return ""


@rate_limit("mcp_tools")
def infer_code_repository(input_data: str) -> str:
    """
    Infer the code repository URL from research-related inputs.
    
    This tool discovers code repositories by:
    - Scraping HuggingFace paper pages for GitHub links
    - Searching GitHub for repositories by paper title
    - Extracting repository links from project pages
    
    Args:
        input_data (str): A URL, paper link, or other research-related input
                         Examples:
                         - "https://arxiv.org/abs/2010.11929"
                         - "https://huggingface.co/papers/2010.11929"
                         - "Vision Transformer"
        
    Returns:
        str: The code repository URL (typically GitHub), or empty string if not found
             Example: "https://github.com/google-research/vision_transformer"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = infer_code_from_row(row_data)
        return result or ""
        
    except Exception as e:
        logger.error(f"Error inferring code: {e}")
        return ""


def infer_research_name(input_data: str) -> str:
    """
    Infer the research paper or project name from various inputs.
    
    Args:
        input_data (str): A URL, repository link, or other research-related input
        
    Returns:
        str: The research name/title, or empty string if not found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = infer_name_from_row(row_data)
        return result or ""
        
    except Exception as e:
        logger.error(f"Error inferring name: {e}")
        return ""


@rate_limit("mcp_tools")
def classify_research_url(input_data: str) -> str:
    """
    Classify the type of research-related URL or input.
    
    This tool identifies resource types based on URL patterns:
    - Paper: arXiv, HuggingFace papers, PDF files
    - Code: GitHub repositories
    - Model: HuggingFace model pages
    - Dataset: HuggingFace dataset pages
    - Space: HuggingFace space/demo pages
    - Project: GitHub.io pages
    - Unknown: Unrecognized patterns
    
    Args:
        input_data (str): The URL or input to classify
                         Examples:
                         - "https://arxiv.org/abs/2103.00020" -> "Paper"
                         - "https://github.com/openai/CLIP" -> "Code"
                         - "https://huggingface.co/openai/clip-vit-base-patch32" -> "Model"
        
    Returns:
        str: The field type: "Paper", "Code", "Space", "Model", "Dataset", "Project", or "Unknown"
    """
    if not input_data or not input_data.strip():
        return "Unknown"
    
    try:
        field = infer_field_type(input_data)
        return field if field else "Unknown"
        
    except Exception as e:
        logger.error(f"Error classifying URL: {e}")
        return "Unknown"


def infer_publication_date(input_data: str) -> str:
    """
    Infer publication date from research paper or project information.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        str: Publication date as string (YYYY-MM-DD format), or empty string if not found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = infer_date_from_row(row_data)
        return result or ""
        
    except Exception as e:
        logger.error(f"Error inferring publication date: {e}")
        return ""


def infer_model(input_data: str) -> str:
    """
    Infer associated HuggingFace model from research paper or project information.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        str: HuggingFace model URL, or empty string if no model found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = infer_model_from_row(row_data)
        return result or ""
        
    except Exception as e:
        logger.error(f"Error inferring model: {e}")
        return ""


def infer_dataset(input_data: str) -> str:
    """
    Infer associated HuggingFace dataset from research paper or project information.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        str: HuggingFace dataset URL, or empty string if no dataset found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = infer_dataset_from_row(row_data)
        return result or ""
        
    except Exception as e:
        logger.error(f"Error inferring dataset: {e}")
        return ""


def infer_space(input_data: str) -> str:
    """
    Infer associated HuggingFace space from research paper or project information.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        str: HuggingFace space URL, or empty string if no space found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = infer_space_from_row(row_data)
        return result or ""
        
    except Exception as e:
        logger.error(f"Error inferring space: {e}")
        return ""


def infer_license(input_data: str) -> str:
    """
    Infer license information from research repository or project.
    
    Args:
        input_data (str): A URL, repository link, or other research-related input
        
    Returns:
        str: License name/type, or empty string if no license found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = infer_license_from_row(row_data)
        return result or ""
        
    except Exception as e:
        logger.error(f"Error inferring license: {e}")
        return ""
