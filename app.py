"""
Research Tracker MCP Server

A Gradio-based MCP server that provides research inference utilities.
Delegates inference logic to the research-tracker-backend for consistency.
"""

import os
import requests
import gradio as gr
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_URL = "https://dylanebert-research-tracker-backend.hf.space"
HF_TOKEN = os.environ.get("HF_TOKEN")
REQUEST_TIMEOUT = 30

if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables")


def validate_input(input_data: str, input_name: str = "input") -> str:
    """
    Validate and sanitize input data.
    
    Args:
        input_data: The input string to validate
        input_name: Name of the input for error messages
        
    Returns:
        Cleaned input string
        
    Raises:
        ValueError: If input is invalid
    """
    if not input_data:
        raise ValueError(f"{input_name} cannot be empty or None")
    
    cleaned = input_data.strip()
    if not cleaned:
        raise ValueError(f"{input_name} cannot be empty after trimming")
    
    # Basic URL validation if it looks like a URL
    if cleaned.startswith(("http://", "https://")):
        if len(cleaned) > 2000:
            raise ValueError(f"{input_name} URL is too long (max 2000 characters)")
        # Check for suspicious patterns
        suspicious_patterns = ["javascript:", "data:", "file:", "ftp:"]
        if any(pattern in cleaned.lower() for pattern in suspicious_patterns):
            raise ValueError(f"{input_name} contains invalid URL scheme")
    
    return cleaned


def make_backend_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a request to the research-tracker-backend with comprehensive error handling.
    
    Args:
        endpoint: The backend endpoint to call (e.g., 'infer-authors')
        data: The data to send in the request body
    
    Returns:
        The response data from the backend
        
    Raises:
        Exception: If the request fails or returns an error
    """
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not available - backend requests may fail")
    
    url = f"{BACKEND_URL}/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}" if HF_TOKEN else ""
    }
    
    try:
        logger.debug(f"Making request to {endpoint} with data: {data}")
        response = requests.post(url, json=data, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 401:
            raise Exception("Authentication failed - please check HF_TOKEN")
        elif response.status_code == 403:
            raise Exception("Access forbidden - insufficient permissions")
        elif response.status_code == 404:
            raise Exception(f"Backend endpoint {endpoint} not found")
        elif response.status_code == 422:
            raise Exception("Invalid request data format")
        elif response.status_code >= 500:
            raise Exception(f"Backend server error (status {response.status_code})")
        
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Backend response: {result}")
        return result
        
    except requests.exceptions.Timeout:
        raise Exception(f"Backend request to {endpoint} timed out after {REQUEST_TIMEOUT}s")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Failed to connect to backend - service may be unavailable")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Backend request to {endpoint} failed: {str(e)}")
    except ValueError as e:
        raise Exception(f"Invalid JSON response from backend: {str(e)}")


def create_row_data(input_data: str) -> Dict[str, Any]:
    """
    Create standardized row data structure for backend requests.
    
    This function analyzes the input and places it in the appropriate field
    based on URL patterns and content analysis.
    
    Args:
        input_data: The input string to analyze
        
    Returns:
        Dictionary with appropriate field populated
    """
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
            # Likely a model URL (huggingface.co/org/model-name)
            row_data["Model"] = input_data
        else:
            # Unknown URL type - try as paper
            row_data["Paper"] = input_data
    else:
        # Non-URL input - likely a paper title or project name
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
        input_data: A URL, paper title, or other research-related input.
                   Supports arXiv URLs, GitHub repositories, HuggingFace resources,
                   project pages, and natural language paper titles.
        
    Returns:
        A list of author names as strings, or empty list if no authors found.
        Authors are returned in the order they appear in the original source.
        
    Examples:
        >>> infer_authors("https://arxiv.org/abs/2010.11929")
        ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", ...]
        
        >>> infer_authors("https://github.com/google-research/vision_transformer")
        ["Alexey Dosovitskiy", "Lucas Beyer", ...]
        
        >>> infer_authors("Vision Transformer")
        ["Alexey Dosovitskiy", "Lucas Beyer", ...]
        
    Raises:
        No exceptions are raised - errors are logged and empty list returned.
    """
    try:
        # Validate and clean input
        cleaned_input = validate_input(input_data, "input_data")
        
        # Create structured data for backend
        row_data = create_row_data(cleaned_input)
        
        # Call the backend
        result = make_backend_request("infer-authors", row_data)
        
        # Extract and validate authors from response
        authors = result.get("authors", [])
        if isinstance(authors, str):
            # Handle comma-separated string format
            authors = [author.strip() for author in authors.split(",") if author.strip()]
        elif not isinstance(authors, list):
            logger.warning(f"Unexpected authors format: {type(authors)}")
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
        
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error inferring authors: {e}")
        return []


def infer_paper_url(input_data: str) -> str:
    """
    Infer the paper URL from various research-related inputs.
    
    This function attempts to find the associated research paper from
    inputs like GitHub repositories, project pages, or partial URLs.
    
    Args:
        input_data: A URL, repository link, or other research-related input
        
    Returns:
        The paper URL (typically arXiv or Hugging Face papers), or empty string if not found
        
    Examples:
        >>> infer_paper_url("https://github.com/google-research/vision_transformer")
        "https://arxiv.org/abs/2010.11929"
        
        >>> infer_paper_url("Vision Transformer")
        "https://arxiv.org/abs/2010.11929"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        # Create row data structure
        row_data = {
            "Name": input_data if not input_data.startswith("http") else None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-paper", row_data)
        
        # Extract paper URL from response
        paper_url = result.get("paper", "")
        return paper_url if paper_url else ""
        
    except Exception as e:
        logger.error(f"Error inferring paper: {e}")
        return ""


def infer_code_repository(input_data: str) -> str:
    """
    Infer the code repository URL from research-related inputs.
    
    This function attempts to find the associated code repository from
    inputs like paper URLs, project pages, or partial information.
    
    Args:
        input_data: A URL, paper link, or other research-related input
        
    Returns:
        The code repository URL (typically GitHub), or empty string if not found
        
    Examples:
        >>> infer_code_repository("https://arxiv.org/abs/2010.11929")
        "https://github.com/google-research/vision_transformer"
        
        >>> infer_code_repository("Vision Transformer")
        "https://github.com/google-research/vision_transformer"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        # Create row data structure
        row_data = {
            "Name": input_data if not input_data.startswith("http") else None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-code", row_data)
        
        # Extract code URL from response
        code_url = result.get("code", "")
        return code_url if code_url else ""
        
    except Exception as e:
        logger.error(f"Error inferring code: {e}")
        return ""


def infer_research_name(input_data: str) -> str:
    """
    Infer the research paper or project name from various inputs.
    
    This function attempts to extract the formal name/title of a research
    paper or project from URLs, repositories, or partial information.
    
    Args:
        input_data: A URL, repository link, or other research-related input
        
    Returns:
        The research name/title, or empty string if not found
        
    Examples:
        >>> infer_research_name("https://arxiv.org/abs/2010.11929")
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
        
        >>> infer_research_name("https://github.com/google-research/vision_transformer")
        "Vision Transformer"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        # Create row data structure
        row_data = {
            "Name": None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-name", row_data)
        
        # Extract name from response
        name = result.get("name", "")
        return name if name else ""
        
    except Exception as e:
        logger.error(f"Error inferring name: {e}")
        return ""


def classify_research_url(url: str) -> str:
    """
    Classify the type of research-related URL or input.
    
    This function determines what type of research resource a given URL
    or input represents (paper, code, model, dataset, etc.).
    
    Args:
        url: The URL or input to classify
        
    Returns:
        The field type: "Paper", "Code", "Space", "Model", "Dataset", "Project", or "Unknown"
        
    Examples:
        >>> classify_research_url("https://arxiv.org/abs/2010.11929")
        "Paper"
        
        >>> classify_research_url("https://github.com/google-research/vision_transformer")
        "Code"
        
        >>> classify_research_url("https://huggingface.co/google/vit-base-patch16-224")
        "Model"
    """
    if not url or not url.strip():
        return "Unknown"
    
    try:
        # Call the backend
        result = make_backend_request("infer-field", {"value": url})
        
        # Extract field from response
        field = result.get("field", "Unknown")
        return field if field else "Unknown"
        
    except Exception as e:
        logger.error(f"Error classifying URL: {e}")
        return "Unknown"


def infer_organizations(input_data: str) -> List[str]:
    """
    Infer affiliated organizations from research paper or project information.
    
    This function attempts to extract organization names from research metadata,
    author affiliations, and repository information. It uses NLP analysis to
    identify institutional affiliations from paper authors and project contributors.
    
    Args:
        input_data: A URL, paper title, or other research-related input
        
    Returns:
        A list of organization names, or empty list if no organizations found
        
    Examples:
        >>> infer_organizations("https://arxiv.org/abs/2010.11929")
        ["Google Research", "University of Amsterdam", "ETH Zurich"]
        
        >>> infer_organizations("https://github.com/openai/gpt-2")
        ["OpenAI"]
    """
    if not input_data or not input_data.strip():
        return []
    
    try:
        # Create row data structure
        row_data = {
            "Name": input_data if not input_data.startswith("http") else None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-orgs", row_data)
        
        # Extract organizations from response
        orgs = result.get("orgs", [])
        if isinstance(orgs, str):
            # Handle comma-separated string format
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
        input_data: A URL, paper title, or other research-related input
        
    Returns:
        Publication date as string (YYYY-MM-DD format), or empty string if not found
        
    Examples:
        >>> infer_publication_date("https://arxiv.org/abs/2010.11929")
        "2020-10-22"
        
        >>> infer_publication_date("https://github.com/google-research/vision_transformer")
        "2020-10-22"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        # Create row data structure
        row_data = {
            "Name": input_data if not input_data.startswith("http") else None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-date", row_data)
        
        # Extract date from response
        date = result.get("date", "")
        return date if date else ""
        
    except Exception as e:
        logger.error(f"Error inferring publication date: {e}")
        return ""


def infer_model(input_data: str) -> str:
    """
    Infer associated HuggingFace model from research paper or project information.
    
    This function attempts to find HuggingFace models associated with research
    papers, GitHub repositories, or project pages. It searches for model
    references in papers, README files, and related documentation.
    
    Args:
        input_data: A URL, paper title, or other research-related input
        
    Returns:
        HuggingFace model URL, or empty string if no model found
        
    Examples:
        >>> infer_model("https://arxiv.org/abs/2010.11929")
        "https://huggingface.co/google/vit-base-patch16-224"
        
        >>> infer_model("Vision Transformer")
        "https://huggingface.co/google/vit-base-patch16-224"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        # Create row data structure
        row_data = {
            "Name": input_data if not input_data.startswith("http") else None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-model", row_data)
        
        # Extract model URL from response
        model = result.get("model", "")
        return model if model else ""
        
    except Exception as e:
        logger.error(f"Error inferring model: {e}")
        return ""


def infer_dataset(input_data: str) -> str:
    """
    Infer associated HuggingFace dataset from research paper or project information.
    
    This function attempts to find HuggingFace datasets used or created by
    research papers, GitHub repositories, or projects. It analyzes paper
    content, repository documentation, and project descriptions.
    
    Args:
        input_data: A URL, paper title, or other research-related input
        
    Returns:
        HuggingFace dataset URL, or empty string if no dataset found
        
    Examples:
        >>> infer_dataset("https://arxiv.org/abs/1706.03762")
        "https://huggingface.co/datasets/wmt14"
        
        >>> infer_dataset("https://github.com/huggingface/transformers")
        "https://huggingface.co/datasets/glue"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        # Create row data structure
        row_data = {
            "Name": input_data if not input_data.startswith("http") else None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-dataset", row_data)
        
        # Extract dataset URL from response
        dataset = result.get("dataset", "")
        return dataset if dataset else ""
        
    except Exception as e:
        logger.error(f"Error inferring dataset: {e}")
        return ""


def infer_space(input_data: str) -> str:
    """
    Infer associated HuggingFace space from research paper or project information.
    
    This function attempts to find HuggingFace spaces (demos/applications) 
    associated with research papers, models, or GitHub repositories. It looks
    for interactive demos and applications built around research.
    
    Args:
        input_data: A URL, paper title, or other research-related input
        
    Returns:
        HuggingFace space URL, or empty string if no space found
        
    Examples:
        >>> infer_space("https://huggingface.co/google/vit-base-patch16-224")
        "https://huggingface.co/spaces/google/vit-demo"
        
        >>> infer_space("https://arxiv.org/abs/2010.11929")
        "https://huggingface.co/spaces/google/vision-transformer-demo"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        # Create row data structure
        row_data = {
            "Name": input_data if not input_data.startswith("http") else None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-space", row_data)
        
        # Extract space URL from response
        space = result.get("space", "")
        return space if space else ""
        
    except Exception as e:
        logger.error(f"Error inferring space: {e}")
        return ""


def infer_license(input_data: str) -> str:
    """
    Infer license information from research repository or project.
    
    This function attempts to extract license information from GitHub
    repositories, project documentation, or associated code. It checks
    license files, repository metadata, and project descriptions.
    
    Args:
        input_data: A URL, repository link, or other research-related input
        
    Returns:
        License name/type, or empty string if no license found
        
    Examples:
        >>> infer_license("https://github.com/google-research/vision_transformer")
        "Apache License 2.0"
        
        >>> infer_license("https://github.com/openai/gpt-2")
        "MIT License"
    """
    if not input_data or not input_data.strip():
        return ""
    
    try:
        # Create row data structure
        row_data = {
            "Name": input_data if not input_data.startswith("http") else None,
            "Authors": [],
            "Paper": input_data if "arxiv" in input_data or "huggingface.co/papers" in input_data else None,
            "Code": input_data if "github.com" in input_data else None,
            "Project": input_data if "github.io" in input_data else None,
            "Space": input_data if "huggingface.co/spaces" in input_data else None,
            "Model": input_data if "huggingface.co/models" in input_data else None,
            "Dataset": input_data if "huggingface.co/datasets" in input_data else None,
        }
        
        # Call the backend
        result = make_backend_request("infer-license", row_data)
        
        # Extract license from response
        license_info = result.get("license", "")
        return license_info if license_info else ""
        
    except Exception as e:
        logger.error(f"Error inferring license: {e}")
        return ""


def batch_infer_research(input_list: List[str], inference_type: str = "authors") -> List[Dict[str, Any]]:
    """
    Perform batch inference on multiple research items for scale analysis.
    
    This function processes multiple research URLs or titles simultaneously,
    applying the specified inference type to each item. Useful for analyzing
    large research datasets, comparing multiple papers, or building research
    knowledge graphs.
    
    Args:
        input_list: List of URLs, paper titles, or research-related inputs to process
        inference_type: Type of inference to perform on each item.
                       Options: "authors", "paper", "code", "name", "organizations",
                       "date", "model", "dataset", "space", "license", "classify"
        
    Returns:
        List of dictionaries, each containing:
        - "input": The original input string
        - "result": The inference result (format depends on inference_type)
        - "success": Boolean indicating if inference succeeded
        - "error": Error message if inference failed
        
    Examples:
        >>> papers = [
        ...     "https://arxiv.org/abs/2010.11929",
        ...     "https://arxiv.org/abs/1706.03762",
        ...     "https://github.com/openai/gpt-2"
        ... ]
        >>> results = batch_infer_research(papers, "authors")
        >>> for result in results:
        ...     print(f"{result['input']}: {len(result['result'])} authors")
        
        >>> urls = ["https://huggingface.co/bert-base-uncased", "https://github.com/pytorch/pytorch"]
        >>> classifications = batch_infer_research(urls, "classify")
        
    Notes:
        - Processing is done sequentially to avoid overwhelming the backend
        - Failed inferences return empty results rather than raising exceptions
        - Large batches may take significant time - consider chunking for very large datasets
    """
    if not input_list:
        return []
    
    # Map inference types to their corresponding functions
    inference_functions = {
        "authors": infer_authors,
        "paper": infer_paper_url,
        "code": infer_code_repository,
        "name": infer_research_name,
        "organizations": infer_organizations,
        "date": infer_publication_date,
        "model": infer_model,
        "dataset": infer_dataset,
        "space": infer_space,
        "license": infer_license,
        "classify": classify_research_url,
    }
    
    if inference_type not in inference_functions:
        logger.error(f"Invalid inference type: {inference_type}")
        return []
    
    inference_func = inference_functions[inference_type]
    results = []
    
    logger.info(f"Starting batch inference of type '{inference_type}' on {len(input_list)} items")
    
    for i, input_item in enumerate(input_list):
        try:
            if not input_item or not isinstance(input_item, str):
                results.append({
                    "input": str(input_item),
                    "result": None,
                    "success": False,
                    "error": "Invalid input: must be non-empty string"
                })
                continue
            
            # Perform inference
            result = inference_func(input_item)
            
            results.append({
                "input": input_item,
                "result": result,
                "success": True,
                "error": None
            })
            
            logger.debug(f"Batch item {i+1}/{len(input_list)} completed successfully")
            
        except Exception as e:
            logger.error(f"Batch inference failed for item {i+1}: {e}")
            results.append({
                "input": input_item,
                "result": None,
                "success": False,
                "error": str(e)
            })
    
    successful_count = sum(1 for r in results if r["success"])
    logger.info(f"Batch inference completed: {successful_count}/{len(input_list)} successful")
    
    return results


def find_research_relationships(input_data: str) -> Dict[str, Any]:
    """
    Find ALL related research resources across platforms for comprehensive analysis.
    
    This function performs a comprehensive analysis of a research item to find
    all related resources including papers, code repositories, models, datasets,
    spaces, and metadata. It's designed for building research knowledge graphs
    and understanding the complete ecosystem around a research topic.
    
    Args:
        input_data: A URL, paper title, or other research-related input
        
    Returns:
        Dictionary containing all discovered related resources:
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
        
    Examples:
        >>> relationships = find_research_relationships("https://arxiv.org/abs/2010.11929")
        >>> print(f"Found {relationships['success_count']} related resources")
        >>> print(f"Authors: {relationships['authors']}")
        >>> print(f"Code: {relationships['code']}")
        >>> print(f"Model: {relationships['model']}")
        
        >>> ecosystem = find_research_relationships("Vision Transformer")
        >>> if ecosystem['paper']:
        ...     print(f"Paper: {ecosystem['paper']}")
        >>> if ecosystem['code']:
        ...     print(f"Implementation: {ecosystem['code']}")
    """
    try:
        # Validate input
        cleaned_input = validate_input(input_data, "input_data")
        
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
        
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        return {"error": str(e), "success_count": 0, "total_inferences": 0}
    except Exception as e:
        logger.error(f"Error finding research relationships: {e}")
        return {"error": str(e), "success_count": 0, "total_inferences": 0}


def validate_research_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Validate accessibility and format of research URLs at scale.
    
    This function checks multiple research URLs for accessibility, format
    validity, and basic content analysis. Useful for data cleaning,
    link validation, and quality assurance of research datasets.
    
    Args:
        urls: List of URLs to validate
        
    Returns:
        List of validation results, each containing:
        - "url": The original URL
        - "accessible": Boolean indicating if URL is reachable
        - "status_code": HTTP status code (if applicable)
        - "format_valid": Boolean indicating if URL format is valid
        - "platform": Detected platform (arxiv, github, huggingface, etc.)
        - "error": Error message if validation failed
        
    Examples:
        >>> urls = [
        ...     "https://arxiv.org/abs/2010.11929",
        ...     "https://github.com/google-research/vision_transformer",
        ...     "https://invalid-url-example"
        ... ]
        >>> validation_results = validate_research_urls(urls)
        >>> accessible_urls = [r for r in validation_results if r["accessible"]]
        >>> print(f"{len(accessible_urls)}/{len(urls)} URLs are accessible")
    """
    if not urls:
        return []
    
    results = []
    logger.info(f"Validating {len(urls)} research URLs")
    
    for url in urls:
        result = {
            "url": url,
            "accessible": False,
            "status_code": None,
            "format_valid": False,
            "platform": "unknown",
            "error": None
        }
        
        try:
            # Basic format validation
            if not isinstance(url, str) or not url.strip():
                result["error"] = "Invalid URL format: empty or non-string"
                results.append(result)
                continue
            
            cleaned_url = url.strip()
            
            # URL format validation
            if not cleaned_url.startswith(("http://", "https://")):
                result["error"] = "Invalid URL format: must start with http:// or https://"
                results.append(result)
                continue
                
            result["format_valid"] = True
            
            # Platform detection
            if "arxiv.org" in cleaned_url:
                result["platform"] = "arxiv"
            elif "github.com" in cleaned_url:
                result["platform"] = "github"
            elif "huggingface.co" in cleaned_url:
                result["platform"] = "huggingface"
            elif "github.io" in cleaned_url:
                result["platform"] = "github_pages"
            
            # Accessibility check
            try:
                response = requests.head(cleaned_url, timeout=10, allow_redirects=True)
                result["status_code"] = response.status_code
                result["accessible"] = 200 <= response.status_code < 400
                
            except requests.exceptions.Timeout:
                result["error"] = "Timeout: URL not accessible within 10 seconds"
            except requests.exceptions.ConnectionError:
                result["error"] = "Connection error: Unable to reach URL"
            except requests.exceptions.RequestException as e:
                result["error"] = f"Request failed: {str(e)}"
                
        except Exception as e:
            result["error"] = f"Validation error: {str(e)}"
        
        results.append(result)
    
    accessible_count = sum(1 for r in results if r["accessible"])
    logger.info(f"URL validation completed: {accessible_count}/{len(urls)} accessible")
    
    return results


# Create Gradio interface
def create_demo():
    """Create the Gradio demo interface for testing."""
    
    with gr.Blocks(title="Research Tracker MCP Server") as demo:
        gr.Markdown("# Research Tracker MCP Server")
        gr.Markdown("Test the comprehensive research inference utilities available through MCP. This server provides cross-platform research analysis, batch processing, and relationship discovery.")
        
        # Core inference functions
        with gr.TabItem("Core Inference"):
            with gr.Tab("Authors"):
                with gr.Row():
                    author_input = gr.Textbox(
                        label="Input (URL, paper title, etc.)",
                        placeholder="https://arxiv.org/abs/2010.11929",
                        lines=1
                    )
                    author_output = gr.JSON(label="Authors")
                author_btn = gr.Button("Infer Authors")
                author_btn.click(infer_authors, inputs=author_input, outputs=author_output)
            
            with gr.Tab("Paper"):
                with gr.Row():
                    paper_input = gr.Textbox(
                        label="Input (GitHub repo, project name, etc.)",
                        placeholder="https://github.com/google-research/vision_transformer",
                        lines=1
                    )
                    paper_output = gr.Textbox(label="Paper URL")
                paper_btn = gr.Button("Infer Paper")
                paper_btn.click(infer_paper_url, inputs=paper_input, outputs=paper_output)
            
            with gr.Tab("Code"):
                with gr.Row():
                    code_input = gr.Textbox(
                        label="Input (paper URL, project name, etc.)",
                        placeholder="https://arxiv.org/abs/2010.11929",
                        lines=1
                    )
                    code_output = gr.Textbox(label="Code Repository URL")
                code_btn = gr.Button("Infer Code")
                code_btn.click(infer_code_repository, inputs=code_input, outputs=code_output)
            
            with gr.Tab("Name"):
                with gr.Row():
                    name_input = gr.Textbox(
                        label="Input (URL, repo, etc.)",
                        placeholder="https://github.com/google-research/vision_transformer",
                        lines=1
                    )
                    name_output = gr.Textbox(label="Research Name/Title")
                name_btn = gr.Button("Infer Name")
                name_btn.click(infer_research_name, inputs=name_input, outputs=name_output)
            
            with gr.Tab("Classify"):
                with gr.Row():
                    classify_input = gr.Textbox(
                        label="URL to classify",
                        placeholder="https://huggingface.co/google/vit-base-patch16-224",
                        lines=1
                    )
                    classify_output = gr.Textbox(label="URL Type")
                classify_btn = gr.Button("Classify URL")
                classify_btn.click(classify_research_url, inputs=classify_input, outputs=classify_output)
        
        # Extended inference functions
        with gr.TabItem("Extended Inference"):
            with gr.Tab("Organizations"):
                with gr.Row():
                    orgs_input = gr.Textbox(
                        label="Input (paper URL, repo, etc.)",
                        placeholder="https://arxiv.org/abs/2010.11929",
                        lines=1
                    )
                    orgs_output = gr.JSON(label="Organizations")
                orgs_btn = gr.Button("Infer Organizations")
                orgs_btn.click(infer_organizations, inputs=orgs_input, outputs=orgs_output)
            
            with gr.Tab("Publication Date"):
                with gr.Row():
                    date_input = gr.Textbox(
                        label="Input (paper URL, repo, etc.)",
                        placeholder="https://arxiv.org/abs/2010.11929",
                        lines=1
                    )
                    date_output = gr.Textbox(label="Publication Date")
                date_btn = gr.Button("Infer Date")
                date_btn.click(infer_publication_date, inputs=date_input, outputs=date_output)
            
            with gr.Tab("Model"):
                with gr.Row():
                    model_input = gr.Textbox(
                        label="Input (paper URL, project name, etc.)",
                        placeholder="https://arxiv.org/abs/2010.11929",
                        lines=1
                    )
                    model_output = gr.Textbox(label="HuggingFace Model URL")
                model_btn = gr.Button("Infer Model")
                model_btn.click(infer_model, inputs=model_input, outputs=model_output)
            
            with gr.Tab("Dataset"):
                with gr.Row():
                    dataset_input = gr.Textbox(
                        label="Input (paper URL, project name, etc.)",
                        placeholder="https://arxiv.org/abs/1706.03762",
                        lines=1
                    )
                    dataset_output = gr.Textbox(label="HuggingFace Dataset URL")
                dataset_btn = gr.Button("Infer Dataset")
                dataset_btn.click(infer_dataset, inputs=dataset_input, outputs=dataset_output)
            
            with gr.Tab("Space"):
                with gr.Row():
                    space_input = gr.Textbox(
                        label="Input (model URL, paper, etc.)",
                        placeholder="https://huggingface.co/google/vit-base-patch16-224",
                        lines=1
                    )
                    space_output = gr.Textbox(label="HuggingFace Space URL")
                space_btn = gr.Button("Infer Space")
                space_btn.click(infer_space, inputs=space_input, outputs=space_output)
            
            with gr.Tab("License"):
                with gr.Row():
                    license_input = gr.Textbox(
                        label="Input (repository URL, project, etc.)",
                        placeholder="https://github.com/google-research/vision_transformer",
                        lines=1
                    )
                    license_output = gr.Textbox(label="License Information")
                license_btn = gr.Button("Infer License")
                license_btn.click(infer_license, inputs=license_input, outputs=license_output)
        
        # Research intelligence functions
        with gr.TabItem("Research Intelligence"):
            with gr.Tab("Research Relationships"):
                gr.Markdown("Find ALL related resources for comprehensive research analysis")
                with gr.Row():
                    relationships_input = gr.Textbox(
                        label="Input (URL, paper title, etc.)",
                        placeholder="https://arxiv.org/abs/2010.11929",
                        lines=1
                    )
                    relationships_output = gr.JSON(label="Related Resources")
                relationships_btn = gr.Button("Find Research Relationships")
                relationships_btn.click(find_research_relationships, inputs=relationships_input, outputs=relationships_output)
            
            with gr.Tab("Batch Processing"):
                gr.Markdown("Process multiple research items simultaneously")
                with gr.Row():
                    with gr.Column():
                        batch_input = gr.Textbox(
                            label="Input URLs/Titles (one per line)",
                            placeholder="https://arxiv.org/abs/2010.11929\nhttps://github.com/openai/gpt-2\nVision Transformer",
                            lines=5
                        )
                        batch_type = gr.Dropdown(
                            choices=["authors", "paper", "code", "name", "organizations", "date", "model", "dataset", "space", "license", "classify"],
                            value="authors",
                            label="Inference Type"
                        )
                    batch_output = gr.JSON(label="Batch Results")
                
                def process_batch(input_text, inference_type):
                    if not input_text.strip():
                        return []
                    input_list = [line.strip() for line in input_text.strip().split('\n') if line.strip()]
                    return batch_infer_research(input_list, inference_type)
                
                batch_btn = gr.Button("Process Batch")
                batch_btn.click(process_batch, inputs=[batch_input, batch_type], outputs=batch_output)
            
            with gr.Tab("URL Validation"):
                gr.Markdown("Validate accessibility and format of research URLs")
                with gr.Row():
                    with gr.Column():
                        url_input = gr.Textbox(
                            label="URLs to validate (one per line)",
                            placeholder="https://arxiv.org/abs/2010.11929\nhttps://github.com/google-research/vision_transformer\nhttps://huggingface.co/google/vit-base-patch16-224",
                            lines=5
                        )
                    url_output = gr.JSON(label="Validation Results")
                
                def validate_urls(input_text):
                    if not input_text.strip():
                        return []
                    url_list = [line.strip() for line in input_text.strip().split('\n') if line.strip()]
                    return validate_research_urls(url_list)
                
                url_btn = gr.Button("Validate URLs")
                url_btn.click(validate_urls, inputs=url_input, outputs=url_output)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(mcp_server=True, share=False)