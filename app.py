"""
Research Tracker MCP Server

A clean, simple MCP server that provides research inference utilities.
Exposes functions to infer research metadata from paper URLs, repository links,
or research names using the research-tracker-backend inference engine.

Key Features:
- Author inference from papers and repositories
- Cross-platform resource discovery (papers, code, models, datasets)
- Research metadata extraction (names, dates, licenses, organizations)
- URL classification and relationship mapping
- Comprehensive research ecosystem analysis

All functions are optimized for MCP usage with clear type hints and docstrings.
"""

import os
import requests
import gradio as gr
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_URL = "https://dylanebert-research-tracker-backend.hf.space"
HF_TOKEN = os.environ.get("HF_TOKEN")
REQUEST_TIMEOUT = 30

if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables")


def make_backend_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the research-tracker-backend."""
    url = f"{BACKEND_URL}/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}" if HF_TOKEN else "",
        "User-Agent": "Research-Tracker-MCP/1.0"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Backend request to {endpoint} failed: {e}")
        raise Exception(f"Backend request to {endpoint} failed: {str(e)}")


def create_row_data(input_data: str) -> Dict[str, Any]:
    """Create standardized row data structure for backend requests."""
    row_data = {
        "Name": None,
        "Authors": [],
        "Paper": None,
        "Code": None,
        "Project": None,
        "Space": None,
        "Model": None,
        "Dataset": None,
    }
    
    # Classify input based on URL patterns
    if input_data.startswith(("http://", "https://")):
        if "arxiv.org" in input_data or "huggingface.co/papers" in input_data:
            row_data["Paper"] = input_data
        elif "github.com" in input_data:
            row_data["Code"] = input_data
        elif "github.io" in input_data:
            row_data["Project"] = input_data
        elif "huggingface.co/spaces" in input_data:
            row_data["Space"] = input_data
        elif "huggingface.co/datasets" in input_data:
            row_data["Dataset"] = input_data
        elif "huggingface.co/" in input_data:
            row_data["Model"] = input_data
        else:
            row_data["Paper"] = input_data
    else:
        row_data["Name"] = input_data
    
    return row_data


def infer_authors(input_data: str) -> List[str]:
    """
    Infer authors from research paper or project information.
    
    This function attempts to extract author names from various inputs like
    paper URLs (arXiv, Hugging Face papers), project pages, or repository links.
    It uses the research-tracker-backend inference engine with sophisticated
    author extraction from paper metadata and repository contributor information.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input.
                         Supports arXiv URLs, GitHub repositories, HuggingFace resources,
                         project pages, and natural language paper titles.
        
    Returns:
        List[str]: A list of author names as strings, or empty list if no authors found.
                   Authors are returned in the order they appear in the original source.
    """
    if not input_data or not input_data.strip():
        return []
    
    try:
        cleaned_input = input_data.strip()
        row_data = create_row_data(cleaned_input)
        result = make_backend_request("infer-authors", row_data)
        
        # Extract and validate authors from response
        authors = result.get("authors", [])
        if isinstance(authors, str):
            # Handle comma-separated string format
            authors = [author.strip() for author in authors.split(",") if author.strip()]
        elif not isinstance(authors, list):
            authors = []
        
        # Filter out empty or invalid author names
        valid_authors = []
        for author in authors:
            if isinstance(author, str) and len(author.strip()) > 0:
                cleaned_author = author.strip()
                # Basic validation - authors should have reasonable length
                if 2 <= len(cleaned_author) <= 100:
                    valid_authors.append(cleaned_author)
        
        logger.info(f"Successfully inferred {len(valid_authors)} authors from input")
        return valid_authors
        
    except Exception as e:
        logger.error(f"Error inferring authors: {e}")
        return []


def infer_paper_url(input_data: str) -> str:
    """
    Infer the paper URL from various research-related inputs.
    
    Args:
        input_data (str): A URL, repository link, or other research-related input
        
    Returns:
        str: The paper URL (typically arXiv or Hugging Face papers), or empty string if not found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = make_backend_request("infer-paper", row_data)
        return result.get("paper", "")
        
    except Exception as e:
        logger.error(f"Error inferring paper: {e}")
        return ""


def infer_code_repository(input_data: str) -> str:
    """
    Infer the code repository URL from research-related inputs.
    
    Args:
        input_data (str): A URL, paper link, or other research-related input
        
    Returns:
        str: The code repository URL (typically GitHub), or empty string if not found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = make_backend_request("infer-code", row_data)
        return result.get("code", "")
        
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
        result = make_backend_request("infer-name", row_data)
        return result.get("name", "")
        
    except Exception as e:
        logger.error(f"Error inferring name: {e}")
        return ""


def classify_research_url(input_data: str) -> str:
    """
    Classify the type of research-related URL or input.
    
    This function determines what type of research resource a given URL
    or input represents (paper, code, model, dataset, etc.).
    
    Args:
        input_data (str): The URL or input to classify
        
    Returns:
        str: The field type: "Paper", "Code", "Space", "Model", "Dataset", "Project", or "Unknown"
    """
    if not input_data or not input_data.strip():
        return "Unknown"
    
    try:
        result = make_backend_request("infer-field", {"value": input_data})
        field = result.get("field", "Unknown")
        return field if field else "Unknown"
        
    except Exception as e:
        logger.error(f"Error classifying URL: {e}")
        return "Unknown"


def infer_organizations(input_data: str) -> List[str]:
    """
    Infer affiliated organizations from research paper or project information.
    
    This function attempts to extract organization names from research metadata,
    author affiliations, and repository information using NLP analysis to identify
    institutional affiliations from paper authors and project contributors.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        List[str]: A list of organization names, or empty list if no organizations found
    """
    if not input_data or not input_data.strip():
        return []
    
    try:
        row_data = create_row_data(input_data.strip())
        result = make_backend_request("infer-orgs", row_data)
        
        orgs = result.get("orgs", [])
        if isinstance(orgs, str):
            orgs = [org.strip() for org in orgs.split(",") if org.strip()]
        elif not isinstance(orgs, list):
            orgs = []
            
        return orgs
        
    except Exception as e:
        logger.error(f"Error inferring organizations: {e}")
        return []


def infer_publication_date(input_data: str) -> str:
    """
    Infer publication date from research paper or project information.
    
    This function attempts to extract publication dates from paper metadata,
    repository creation dates, or release information. Returns dates in
    standardized format (YYYY-MM-DD) when possible.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        str: Publication date as string (YYYY-MM-DD format), or empty string if not found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = make_backend_request("infer-date", row_data)
        return result.get("date", "")
        
    except Exception as e:
        logger.error(f"Error inferring publication date: {e}")
        return ""


def infer_model(input_data: str) -> str:
    """
    Infer associated HuggingFace model from research paper or project information.
    
    This function attempts to find HuggingFace models associated with research papers,
    GitHub repositories, or project pages. It searches for model references in papers,
    README files, and related documentation.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        str: HuggingFace model URL, or empty string if no model found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = make_backend_request("infer-model", row_data)
        return result.get("model", "")
        
    except Exception as e:
        logger.error(f"Error inferring model: {e}")
        return ""


def infer_dataset(input_data: str) -> str:
    """
    Infer associated HuggingFace dataset from research paper or project information.
    
    This function attempts to find HuggingFace datasets used or created by research papers,
    GitHub repositories, or projects. It analyzes paper content, repository documentation,
    and project descriptions.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        str: HuggingFace dataset URL, or empty string if no dataset found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = make_backend_request("infer-dataset", row_data)
        return result.get("dataset", "")
        
    except Exception as e:
        logger.error(f"Error inferring dataset: {e}")
        return ""


def infer_space(input_data: str) -> str:
    """
    Infer associated HuggingFace space from research paper or project information.
    
    This function attempts to find HuggingFace spaces (demos/applications) associated
    with research papers, models, or GitHub repositories. It looks for interactive
    demos and applications built around research.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        str: HuggingFace space URL, or empty string if no space found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = make_backend_request("infer-space", row_data)
        return result.get("space", "")
        
    except Exception as e:
        logger.error(f"Error inferring space: {e}")
        return ""


def infer_license(input_data: str) -> str:
    """
    Infer license information from research repository or project.
    
    This function attempts to extract license information from GitHub repositories,
    project documentation, or associated code. It checks license files, repository
    metadata, and project descriptions.
    
    Args:
        input_data (str): A URL, repository link, or other research-related input
        
    Returns:
        str: License name/type, or empty string if no license found
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        row_data = create_row_data(input_data.strip())
        result = make_backend_request("infer-license", row_data)
        return result.get("license", "")
        
    except Exception as e:
        logger.error(f"Error inferring license: {e}")
        return ""


def find_research_relationships(input_data: str) -> Dict[str, Any]:
    """
    Find ALL related research resources across platforms for comprehensive analysis.
    
    This function performs a comprehensive analysis of a research item to find
    all related resources including papers, code repositories, models, datasets,
    spaces, and metadata. It's designed for building research knowledge graphs
    and understanding the complete ecosystem around a research topic.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        Dict[str, Any]: Dictionary containing all discovered related resources:
        {
            "paper": str | None,           # Associated research paper
            "code": str | None,            # Code repository URL
            "name": str | None,            # Research/project name
            "authors": List[str],          # Author names
            "organizations": List[str],    # Affiliated organizations
            "date": str | None,           # Publication date
            "model": str | None,          # HuggingFace model URL
            "dataset": str | None,        # HuggingFace dataset URL
            "space": str | None,          # HuggingFace space URL
            "license": str | None,        # License information
            "field_type": str | None,     # Classification of input type
            "success_count": int,         # Number of successful inferences
            "total_inferences": int       # Total inferences attempted
        }
    """
    if not input_data or not input_data.strip():
        return {"error": "Input data cannot be empty", "success_count": 0, "total_inferences": 0}
    
    try:
        cleaned_input = input_data.strip()
        
        # Initialize result structure
        relationships = {
            "paper": None,
            "code": None,
            "name": None,
            "authors": [],
            "organizations": [],
            "date": None,
            "model": None,
            "dataset": None,
            "space": None,
            "license": None,
            "field_type": None,
            "success_count": 0,
            "total_inferences": 11  # Number of inference types we'll attempt
        }
        
        # Define inference operations
        inferences = [
            ("paper", infer_paper_url),
            ("code", infer_code_repository),
            ("name", infer_research_name),
            ("authors", infer_authors),
            ("organizations", infer_organizations),
            ("date", infer_publication_date),
            ("model", infer_model),
            ("dataset", infer_dataset),
            ("space", infer_space),
            ("license", infer_license),
            ("field_type", classify_research_url)
        ]
        
        logger.info(f"Finding research relationships for: {cleaned_input}")
        
        # Perform all inferences
        for field_name, inference_func in inferences:
            try:
                result = inference_func(cleaned_input)
                
                # Handle different return types
                if isinstance(result, list) and result:
                    relationships[field_name] = result
                    relationships["success_count"] += 1
                elif isinstance(result, str) and result.strip():
                    relationships[field_name] = result.strip()
                    relationships["success_count"] += 1
                # else: leave as None (unsuccessful inference)
                
            except Exception as e:
                logger.warning(f"Failed to infer {field_name}: {e}")
                # Continue with other inferences
        
        logger.info(f"Research relationship analysis completed: {relationships['success_count']}/{relationships['total_inferences']} successful")
        return relationships
        
    except Exception as e:
        logger.error(f"Error finding research relationships: {e}")
        return {"error": str(e), "success_count": 0, "total_inferences": 0}


# Create minimal Gradio interface focused on MCP tool exposure
with gr.Blocks(title="Research Tracker MCP Server") as demo:
    gr.Markdown("# Research Tracker MCP Server")
    gr.Markdown("""
    This server provides MCP tools for research inference and metadata extraction.
    
    **Available MCP Tools:**
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
    
    **Input Support:**
    - arXiv paper URLs (https://arxiv.org/abs/...)
    - GitHub repository URLs (https://github.com/...)
    - HuggingFace model/dataset/space URLs
    - Research paper titles and project names
    - Project page URLs
    """)
    
    # Expose all core functions as MCP tools
    gr.api(infer_authors)
    gr.api(infer_paper_url)
    gr.api(infer_code_repository)
    gr.api(infer_research_name)
    gr.api(classify_research_url)
    gr.api(infer_organizations)
    gr.api(infer_publication_date)
    gr.api(infer_model)
    gr.api(infer_dataset)
    gr.api(infer_space)
    gr.api(infer_license)
    gr.api(find_research_relationships)


if __name__ == "__main__":
    logger.info("Starting Research Tracker MCP Server")
    demo.launch(mcp_server=True, share=False)