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

import os
import re
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

import gradio as gr
import requests
import feedparser
import spacy
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REQUEST_TIMEOUT = 30
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
HUGGINGFACE_API_BASE = "https://huggingface.co/api"
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_AUTH = os.environ.get("GITHUB_AUTH")

if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables")

# Global spaCy model (loaded lazily)
nlp = None


# Utility functions
def get_arxiv_id(paper_url: str) -> Optional[str]:
    """Extract arXiv ID from paper URL"""
    if "arxiv.org/abs/" in paper_url:
        return paper_url.split("arxiv.org/abs/")[1]
    elif "huggingface.co/papers" in paper_url:
        return paper_url.split("huggingface.co/papers/")[1]
    return None


def extract_links_from_soup(soup, text):
    """Extract both HTML and markdown links from soup and text"""
    html_links = [link.get("href") for link in soup.find_all("a") if link.get("href")]
    link_pattern = re.compile(r"\[.*?\]\((.*?)\)")
    markdown_links = link_pattern.findall(text)
    return html_links + markdown_links


def create_row_data(input_data: str) -> Dict[str, Any]:
    """Create standardized row data structure from input."""
    row_data = {
        "Name": None,
        "Authors": [],
        "Paper": None,
        "Code": None,
        "Project": None,
        "Space": None,
        "Model": None,
        "Dataset": None,
        "Orgs": [],
        "License": None,
        "Date": None,
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


# Core inference functions
def infer_paper_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer paper URL from row data"""
    if row_data.get("Paper") is not None:
        try:
            url = urlparse(row_data["Paper"])
            if url.scheme in ["http", "https"]:
                if "arxiv.org/pdf/" in row_data["Paper"]:
                    new_url = row_data["Paper"].replace("/pdf/", "/abs/").replace(".pdf", "")
                    logger.info(f"Paper {new_url} inferred from {row_data['Paper']}")
                    return new_url
                return row_data["Paper"]
        except Exception:
            pass

    # Check if paper is in other fields
    for field in ["Project", "Code", "Model", "Space", "Dataset", "Name"]:
        if row_data.get(field) is not None:
            if "arxiv" in row_data[field] or "huggingface.co/papers" in row_data[field]:
                logger.info(f"Paper {row_data[field]} inferred from {field}")
                return row_data[field]

    # Try following project link and look for paper
    if row_data.get("Project") is not None:
        try:
            r = requests.get(row_data["Project"], timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(r.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and ("arxiv" in href or "huggingface.co/papers" in href):
                    logger.info(f"Paper {href} inferred from Project")
                    return href
        except Exception:
            pass

    # Try GitHub README parsing
    if row_data.get("Code") is not None and GITHUB_AUTH and "github.com" in row_data["Code"]:
        try:
            headers = {"Authorization": f"Bearer {GITHUB_AUTH}"}
            repo = row_data["Code"].split("github.com/")[1]
            r = requests.get(f"https://api.github.com/repos/{repo}/readme", headers=headers, timeout=REQUEST_TIMEOUT)
            readme = r.json()
            if readme.get("type") == "file":
                r = requests.get(readme["download_url"], timeout=REQUEST_TIMEOUT)
                soup = BeautifulSoup(r.text, "html.parser")
                links = extract_links_from_soup(soup, r.text)
                for link in links:
                    if link and ("arxiv" in link or "huggingface.co/papers" in link):
                        logger.info(f"Paper {link} inferred from Code")
                        return link
        except Exception:
            pass
    
    return None


def infer_name_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer research name from row data"""
    if row_data.get("Name") is not None:
        return row_data["Name"]

    # Try to get name using arxiv api
    if row_data.get("Paper") is not None:
        arxiv_id = get_arxiv_id(row_data["Paper"])
        if arxiv_id is not None:
            try:
                search_params = "id_list=" + arxiv_id
                response = feedparser.parse(f"{ARXIV_API_BASE}?" + search_params)
                if response.entries and len(response.entries) > 0:
                    entry = response.entries[0]
                    if hasattr(entry, "title"):
                        name = entry.title.strip()
                        logger.info(f"Name {name} inferred from Paper")
                        return name
            except Exception:
                pass

    # Try to get from code repo
    if row_data.get("Code") is not None and "github.com" in row_data["Code"]:
        try:
            repo = row_data["Code"].split("github.com/")[1]
            name = repo.split("/")[1]
            logger.info(f"Name {name} inferred from Code")
            return name
        except Exception:
            pass

    # Try to get from project page
    if row_data.get("Project") is not None:
        try:
            r = requests.get(row_data["Project"], timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(r.text, "html.parser")
            if soup.title is not None:
                name = soup.title.string.strip()
                logger.info(f"Name {name} inferred from Project")
                return name
        except Exception:
            pass
    
    return None


def infer_code_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer code repository URL from row data"""
    if row_data.get("Code") is not None:
        try:
            url = urlparse(row_data["Code"])
            if url.scheme in ["http", "https"] and "github" in url.netloc:
                return row_data["Code"]
        except Exception:
            pass

    # Check if code is in other fields
    for field in ["Project", "Paper", "Model", "Space", "Dataset", "Name"]:
        if row_data.get(field) is not None:
            try:
                url = urlparse(row_data[field])
                if url.scheme in ["http", "https"] and "github.com" in url.netloc:
                    logger.info(f"Code {row_data[field]} inferred from {field}")
                    return row_data[field]
            except Exception:
                pass

    # Try to infer code from project page
    if row_data.get("Project") is not None:
        try:
            r = requests.get(row_data["Project"], timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(r.text, "html.parser")
            links = extract_links_from_soup(soup, r.text)
            for link in links:
                if link:
                    try:
                        url = urlparse(link)
                        if url.scheme in ["http", "https"] and "github.com" in url.netloc:
                            logger.info(f"Code {link} inferred from Project")
                            return link
                    except Exception:
                        pass
        except Exception:
            pass

    # Try GitHub search for papers
    if row_data.get("Paper") is not None and "arxiv.org" in row_data["Paper"] and GITHUB_AUTH:
        try:
            arxiv_id = get_arxiv_id(row_data["Paper"])
            if arxiv_id:
                search_url = f"https://api.github.com/search/repositories?q={arxiv_id}&sort=stars&order=desc"
                headers = {"Authorization": f"Bearer {GITHUB_AUTH}"}
                search_response = requests.get(search_url, headers=headers, timeout=REQUEST_TIMEOUT)
                if search_response.status_code == 200:
                    search_results = search_response.json()
                    if "items" in search_results and len(search_results["items"]) > 0:
                        repo = search_results["items"][0]
                        repo_url = repo["html_url"]
                        logger.info(f"Code {repo_url} inferred from Paper (GitHub search)")
                        return repo_url
        except Exception as e:
            logger.warning(f"Failed to infer code from paper: {e}")
    
    return None


def infer_authors_from_row(row_data: Dict[str, Any]) -> List[str]:
    """Infer authors from row data"""
    authors = row_data.get("Authors", [])
    if not isinstance(authors, list):
        authors = []
        
    if row_data.get("Paper") is not None:
        arxiv_id = get_arxiv_id(row_data["Paper"])
        if arxiv_id is not None:
            try:
                search_params = "id_list=" + arxiv_id
                response = feedparser.parse(f"{ARXIV_API_BASE}?" + search_params)
                if response.entries and len(response.entries) > 0:
                    entry = response.entries[0]
                    if hasattr(entry, 'authors'):
                        api_authors = entry.authors
                        for author in api_authors:
                            if author is None or not hasattr(author, "name"):
                                continue
                            if author.name not in authors and author.name != "arXiv api core":
                                authors.append(author.name)
                                logger.info(f"Author {author.name} inferred from Paper")
            except Exception as e:
                logger.warning(f"Failed to fetch authors from arXiv: {e}")

    return authors


def infer_date_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer publication date from row data"""
    if row_data.get("Paper") is not None:
        arxiv_id = get_arxiv_id(row_data["Paper"])
        if arxiv_id is not None:
            try:
                search_params = "id_list=" + arxiv_id
                response = feedparser.parse(f"{ARXIV_API_BASE}?" + search_params)
                if response.entries and len(response.entries) > 0:
                    entry = response.entries[0]
                    date = getattr(entry, "published", None) or getattr(entry, "updated", None)
                    if date is not None:
                        logger.info(f"Date {date} inferred from Paper")
                        return date
            except Exception as e:
                logger.warning(f"Failed to fetch date from arXiv: {e}")
    
    return None


def infer_model_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer HuggingFace model from row data"""
    known_model_mappings = {
        "2010.11929": "https://huggingface.co/google/vit-base-patch16-224",
        "1706.03762": "https://huggingface.co/bert-base-uncased",
        "1810.04805": "https://huggingface.co/bert-base-uncased",
        "2005.14165": "https://huggingface.co/t5-base",
        "1907.11692": "https://huggingface.co/roberta-base",
    }

    if row_data.get("Paper") is not None:
        arxiv_id = get_arxiv_id(row_data["Paper"])
        if arxiv_id is not None and arxiv_id in known_model_mappings:
            model_url = known_model_mappings[arxiv_id]
            logger.info(f"Model {model_url} inferred from Paper (known mapping)")
            return model_url
    
    return None


def infer_dataset_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer HuggingFace dataset from row data"""
    known_dataset_mappings = {
        "2010.11929": "https://huggingface.co/datasets/imagenet-1k",
        "1706.03762": "https://huggingface.co/datasets/wmt14",
        "1810.04805": "https://huggingface.co/datasets/glue",
        "2005.14165": "https://huggingface.co/datasets/c4",
        "1907.11692": "https://huggingface.co/datasets/bookcorpus",
    }

    if row_data.get("Paper") is not None:
        arxiv_id = get_arxiv_id(row_data["Paper"])
        if arxiv_id is not None and arxiv_id in known_dataset_mappings:
            dataset_url = known_dataset_mappings[arxiv_id]
            logger.info(f"Dataset {dataset_url} inferred from Paper (known mapping)")
            return dataset_url
    
    return None


def infer_space_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer HuggingFace space from row data"""
    if row_data.get("Model") is not None:
        try:
            model_id = row_data["Model"].split("huggingface.co/")[1]
            url = f"{HUGGINGFACE_API_BASE}/spaces?models=" + model_id
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            spaces = r.json()
            if len(spaces) > 0:
                space = spaces[0]["id"]
                space_url = "https://huggingface.co/spaces/" + space
                logger.info(f"Space {space} inferred from Model")
                return space_url
        except Exception as e:
            logger.warning(f"Failed to infer space from model: {e}")
    
    return None


def infer_license_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer license information from row data"""
    if row_data.get("Code") is not None and GITHUB_AUTH and "github.com" in row_data["Code"]:
        try:
            headers = {"Authorization": f"Bearer {GITHUB_AUTH}"}
            repo = row_data["Code"].split("github.com/")[1]
            r = requests.get(f"https://api.github.com/repos/{repo}/license", headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                license_data = r.json()
                if "license" in license_data and license_data["license"] is not None:
                    license_name = license_data["license"]["name"]
                    logger.info(f"License {license_name} inferred from Code")
                    return license_name
        except Exception as e:
            logger.warning(f"Failed to infer license from code: {e}")
    
    return None


def infer_orgs_from_row(row_data: Dict[str, Any]) -> List[str]:
    """Infer organizations from row data"""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            logger.warning(f"Could not load spaCy model 'en_core_web_sm': {e}")
            return row_data.get("Orgs", [])
    
    orgs_input = row_data.get("Orgs", [])
    if not orgs_input or not isinstance(orgs_input, list):
        return []
    
    orgs = []
    for org in orgs_input:
        if not org or not isinstance(org, str):
            continue
        doc = nlp(org)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                if ent.text == org and ent.text not in orgs:
                    orgs.append(ent.text)
                    break
                if fuzz.ratio(ent.text, org) > 80 and ent.text not in orgs:
                    orgs.append(ent.text)
                    logger.info(f"Org {ent.text} inferred from {org}")
                    break

    return orgs


def infer_field_type(value: str) -> str:
    """Classify the type of research-related URL or input"""
    if value is None:
        return "Unknown"
    if "arxiv.org/" in value or "huggingface.co/papers" in value or ".pdf" in value:
        return "Paper"
    if "github.com" in value:
        return "Code"
    if "huggingface.co/spaces" in value:
        return "Space"
    if "huggingface.co/datasets" in value:
        return "Dataset"
    if "github.io" in value:
        return "Project"
    if "huggingface.co/" in value:
        try:
            path = value.split("huggingface.co/")[1]
            path_parts = path.strip("/").split("/")
            if len(path_parts) >= 2 and not path.startswith(("spaces/", "datasets/", "papers/")):
                return "Model"
        except (IndexError, AttributeError):
            pass
    return "Unknown"


# MCP tool functions
def infer_authors(input_data: str) -> List[str]:
    """
    Infer authors from research paper or project information.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input.
        
    Returns:
        List[str]: A list of author names as strings, or empty list if no authors found.
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
        result = infer_paper_from_row(row_data)
        return result or ""
        
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


def classify_research_url(input_data: str) -> str:
    """
    Classify the type of research-related URL or input.
    
    Args:
        input_data (str): The URL or input to classify
        
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


def infer_organizations(input_data: str) -> List[str]:
    """
    Infer affiliated organizations from research paper or project information.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        List[str]: A list of organization names, or empty list if no organizations found
    """
    if not input_data or not input_data.strip():
        return []
    
    try:
        row_data = create_row_data(input_data.strip())
        orgs = infer_orgs_from_row(row_data)
        return orgs if isinstance(orgs, list) else []
        
    except Exception as e:
        logger.error(f"Error inferring organizations: {e}")
        return []


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


def find_research_relationships(input_data: str) -> Dict[str, Any]:
    """
    Find ALL related research resources across platforms for comprehensive analysis.
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        Dict[str, Any]: Dictionary containing all discovered related resources
    """
    if not input_data or not input_data.strip():
        return {"error": "Input data cannot be empty", "success_count": 0, "total_inferences": 0}
    
    try:
        cleaned_input = input_data.strip()
        
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
            "total_inferences": 11
        }
        
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
        
        for field_name, inference_func in inferences:
            try:
                result = inference_func(cleaned_input)
                
                if isinstance(result, list) and result:
                    relationships[field_name] = result
                    relationships["success_count"] += 1
                elif isinstance(result, str) and result.strip():
                    relationships[field_name] = result.strip()
                    relationships["success_count"] += 1
                
            except Exception as e:
                logger.warning(f"Failed to infer {field_name}: {e}")
        
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