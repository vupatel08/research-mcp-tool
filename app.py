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


def make_backend_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a request to the research-tracker-backend.
    
    Args:
        endpoint: The backend endpoint to call (e.g., 'infer-authors')
        data: The data to send in the request body
    
    Returns:
        The response data from the backend
        
    Raises:
        Exception: If the request fails or returns an error
    """
    url = f"{BACKEND_URL}/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}" if HF_TOKEN else ""
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise Exception(f"Request to {endpoint} timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request to {endpoint} failed: {str(e)}")


def infer_authors(input_data: str) -> List[str]:
    """
    Infer authors from research paper or project information.
    
    This function attempts to extract author names from various inputs like
    paper URLs (arXiv, Hugging Face papers), project pages, or repository links.
    It uses the research-tracker-backend inference engine.
    
    Args:
        input_data: A URL, paper title, or other research-related input
        
    Returns:
        A list of author names, or empty list if no authors found
        
    Examples:
        >>> infer_authors("https://arxiv.org/abs/2103.00020")
        ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", ...]
        
        >>> infer_authors("https://github.com/google-research/vision_transformer")
        ["Alexey Dosovitskiy", "Lucas Beyer", ...]
    """
    if not input_data or not input_data.strip():
        return []
    
    try:
        # Create a minimal row data structure for the backend
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
        
        # If we can't classify the input, try it as a paper
        if not any(row_data.values()):
            row_data["Paper"] = input_data
        
        # Call the backend
        result = make_backend_request("infer-authors", row_data)
        
        # Extract authors from response
        authors = result.get("authors", [])
        if isinstance(authors, str):
            # Handle comma-separated string format
            authors = [author.strip() for author in authors.split(",") if author.strip()]
        elif not isinstance(authors, list):
            authors = []
            
        return authors
        
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


# Create Gradio interface
def create_demo():
    """Create the Gradio demo interface for testing."""
    
    with gr.Blocks(title="Research Tracker MCP Server") as demo:
        gr.Markdown("# Research Tracker MCP Server")
        gr.Markdown("Test the research inference utilities that are available through MCP.")
        
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
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(mcp_server=True, share=False)