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
import time
from urllib.parse import urlparse, quote
from typing import List, Dict, Any, Optional, Union
from functools import wraps
from datetime import datetime, timedelta

import gradio as gr
import requests
import feedparser
from bs4 import BeautifulSoup

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
    "github.io"
}

if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables")

# Enhanced cache with TTL for scraping results
_scrape_cache = {}  # {url: {"data": ..., "timestamp": ...}}
_rate_limit_tracker = {}  # {key: [timestamps]}


class MCPError(Exception):
    """Base exception for MCP-related errors"""
    pass


class ValidationError(MCPError):
    """Input validation error"""
    pass


class ExternalAPIError(MCPError):
    """External API call error"""
    pass


def validate_url(url: str) -> bool:
    """Validate URL for security and correctness"""
    if not url or len(url) > MAX_URL_LENGTH:
        return False
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Extract domain
        domain = parsed.netloc.lower()
        if ":" in domain:
            domain = domain.split(":")[0]
        
        # Check against allowed domains
        return any(domain.endswith(allowed) for allowed in ALLOWED_DOMAINS)
    except Exception:
        return False


def rate_limit(key: str):
    """Simple rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Clean old timestamps
            if key in _rate_limit_tracker:
                _rate_limit_tracker[key] = [
                    ts for ts in _rate_limit_tracker[key]
                    if now - ts < RATE_LIMIT_WINDOW
                ]
            else:
                _rate_limit_tracker[key] = []
            
            # Check rate limit
            if len(_rate_limit_tracker[key]) >= RATE_LIMIT_CALLS:
                raise MCPError(f"Rate limit exceeded. Max {RATE_LIMIT_CALLS} calls per {RATE_LIMIT_WINDOW} seconds.")
            
            _rate_limit_tracker[key].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def make_github_request(endpoint: str, headers: Optional[Dict] = None) -> Optional[requests.Response]:
    """Make GitHub API request with proper authentication and error handling"""
    if not GITHUB_AUTH:
        return None
        
    url = f"https://api.github.com{endpoint}" if endpoint.startswith("/") else endpoint
    
    if not headers:
        headers = {}
    headers["Authorization"] = f"Bearer {GITHUB_AUTH}"
    
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response
        elif response.status_code == 404:
            return None
        else:
            logger.warning(f"GitHub API returned {response.status_code} for {url}")
            return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"GitHub API request failed: {e}")
        return None


def cached_request(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    """Make HTTP request with caching, retries, and validation"""
    if not validate_url(url):
        raise ValidationError(f"Invalid or disallowed URL: {url}")
    
    # Check cache
    if url in _scrape_cache:
        cache_entry = _scrape_cache[url]
        # Handle both old and new cache formats
        if isinstance(cache_entry, dict) and "timestamp" in cache_entry:
            if time.time() - cache_entry["timestamp"] < CACHE_TTL:
                logger.debug(f"Cache hit for {url}")
                return cache_entry["data"]
        else:
            # Old cache format, clear it
            del _scrape_cache[url]
    
    # Make request with retries
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                # Cache successful response
                _scrape_cache[url] = {
                    "data": response,
                    "timestamp": time.time()
                }
                return response
            elif response.status_code == 404:
                return None
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt + 1}: {e}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
    
    raise ExternalAPIError(f"Failed to fetch {url} after {MAX_RETRIES} attempts")


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


def scrape_huggingface_paper_page(paper_url: str) -> Dict[str, Any]:
    """
    Scrape HuggingFace paper page to find associated resources with caching
    
    Returns:
        Dict containing found resources: {
            "models": [], "datasets": [], "spaces": [], "code": []
        }
    """
    # Default empty resources
    empty_resources = {"models": [], "datasets": [], "spaces": [], "code": []}
    
    if not paper_url or "huggingface.co/papers" not in paper_url:
        return empty_resources
    
    try:
        response = cached_request(paper_url)
        if not response:
            return empty_resources
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all links on the page
        links = set()  # Use set to avoid duplicates
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # Convert relative URLs to absolute
            if href.startswith("/"):
                href = "https://huggingface.co" + href
            elif href.startswith("huggingface.co"):
                href = "https://" + href
            links.add(href)
        
        # Categorize links efficiently
        resources = {"models": [], "datasets": [], "spaces": [], "code": []}
        for link in links:
            if "huggingface.co/" in link:
                if "/models/" in link:
                    resources["models"].append(link)
                elif "/datasets/" in link:
                    resources["datasets"].append(link)
                elif "/spaces/" in link:
                    resources["spaces"].append(link)
            elif "github.com" in link:
                resources["code"].append(link)
        
        # Cache the result
        _scrape_cache[paper_url] = resources
        
        logger.info(f"Scraped {len(resources['models'])} models, {len(resources['datasets'])} datasets, "
                   f"{len(resources['spaces'])} spaces, {len(resources['code'])} code repos from {paper_url}")
        
    except ValidationError as e:
        logger.error(f"Validation error scraping HF paper page: {e}")
        return empty_resources
    except ExternalAPIError as e:
        logger.error(f"External API error scraping HF paper page: {e}")
        return empty_resources
    except Exception as e:
        logger.error(f"Unexpected error scraping HF paper page: {e}")
        return empty_resources
    
    return resources


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
                # Convert arXiv PDF to abs format
                if "arxiv.org/pdf/" in row_data["Paper"]:
                    new_url = row_data["Paper"].replace("/pdf/", "/abs/").replace(".pdf", "")
                    logger.info(f"Paper {new_url} inferred from {row_data['Paper']}")
                    return new_url
                
                # If this is an arXiv URL, try HuggingFace papers first for better resource discovery
                if "arxiv.org/abs/" in row_data["Paper"]:
                    arxiv_id = row_data["Paper"].split("arxiv.org/abs/")[1]
                    hf_paper_url = f"https://huggingface.co/papers/{arxiv_id}"
                    try:
                        # Test if HuggingFace paper page exists and has content
                        response = cached_request(hf_paper_url)
                        if response and len(response.text) > 1000:  # Basic check for content
                            logger.info(f"Paper {hf_paper_url} inferred from arXiv (HuggingFace preferred)")
                            return hf_paper_url
                    except (ValidationError, ExternalAPIError):
                        pass  # Fall back to original arXiv URL
                
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
            response = cached_request(row_data["Project"])
            if response:
                soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and ("arxiv" in href or "huggingface.co/papers" in href):
                    logger.info(f"Paper {href} inferred from Project")
                    return href
        except (ValidationError, ExternalAPIError) as e:
            logger.debug(f"Failed to scrape project page: {e}")

    # Try GitHub README parsing
    if row_data.get("Code") is not None and GITHUB_AUTH and "github.com" in row_data["Code"]:
        try:
            repo = row_data["Code"].split("github.com/")[1]
            # GitHub API requires special handling
            r = make_github_request(f"/repos/{repo}/readme")
            if r:
                readme = r.json()
                if readme.get("type") == "file" and readme.get("download_url"):
                    response = cached_request(readme["download_url"])
                    if response:
                        soup = BeautifulSoup(response.text, "html.parser")
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

    # Try scraping HuggingFace paper page for code links
    if row_data.get("Paper") is not None:
        arxiv_id = get_arxiv_id(row_data["Paper"])
        
        # Try scraping HuggingFace paper page
        if "huggingface.co/papers" in row_data["Paper"]:
            resources = scrape_huggingface_paper_page(row_data["Paper"])
            if resources["code"]:
                code_url = resources["code"][0]  # Take first code repo found
                logger.info(f"Code {code_url} inferred from HuggingFace paper page")
                return code_url
        
        # If we have arXiv URL, try the HuggingFace version first
        elif "arxiv.org/abs/" in row_data["Paper"] and arxiv_id:
            hf_paper_url = f"https://huggingface.co/papers/{arxiv_id}"
            resources = scrape_huggingface_paper_page(hf_paper_url)
            if resources["code"]:
                code_url = resources["code"][0]
                logger.info(f"Code {code_url} inferred from HuggingFace paper page (via arXiv)")
                return code_url

    # Fallback: Try GitHub search for papers
    if row_data.get("Paper") is not None and GITHUB_AUTH:
        arxiv_id = get_arxiv_id(row_data["Paper"])
        if arxiv_id:
            try:
                search_endpoint = f"/search/repositories?q={arxiv_id}&sort=stars&order=desc"
                search_response = make_github_request(search_endpoint)
                if search_response:
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
    """Infer HuggingFace model from row data by scraping paper page"""
    if row_data.get("Paper") is not None:
        # Try scraping HuggingFace paper page
        if "huggingface.co/papers" in row_data["Paper"]:
            resources = scrape_huggingface_paper_page(row_data["Paper"])
            if resources["models"]:
                model_url = resources["models"][0]  # Take first model found
                logger.info(f"Model {model_url} inferred from HuggingFace paper page")
                return model_url
        
        # If we have arXiv URL, try the HuggingFace version
        elif "arxiv.org/abs/" in row_data["Paper"]:
            arxiv_id = get_arxiv_id(row_data["Paper"])
            if arxiv_id:
                hf_paper_url = f"https://huggingface.co/papers/{arxiv_id}"
                resources = scrape_huggingface_paper_page(hf_paper_url)
                if resources["models"]:
                    model_url = resources["models"][0]
                    logger.info(f"Model {model_url} inferred from HuggingFace paper page (via arXiv)")
                    return model_url
    
    return None


def infer_dataset_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer HuggingFace dataset from row data by scraping paper page"""
    if row_data.get("Paper") is not None:
        # Try scraping HuggingFace paper page
        if "huggingface.co/papers" in row_data["Paper"]:
            resources = scrape_huggingface_paper_page(row_data["Paper"])
            if resources["datasets"]:
                dataset_url = resources["datasets"][0]  # Take first dataset found
                logger.info(f"Dataset {dataset_url} inferred from HuggingFace paper page")
                return dataset_url
        
        # If we have arXiv URL, try the HuggingFace version
        elif "arxiv.org/abs/" in row_data["Paper"]:
            arxiv_id = get_arxiv_id(row_data["Paper"])
            if arxiv_id:
                hf_paper_url = f"https://huggingface.co/papers/{arxiv_id}"
                resources = scrape_huggingface_paper_page(hf_paper_url)
                if resources["datasets"]:
                    dataset_url = resources["datasets"][0]
                    logger.info(f"Dataset {dataset_url} inferred from HuggingFace paper page (via arXiv)")
                    return dataset_url
    
    return None


def infer_space_from_row(row_data: Dict[str, Any]) -> Optional[str]:
    """Infer HuggingFace space from row data by scraping paper page"""
    if row_data.get("Paper") is not None:
        # Try scraping HuggingFace paper page
        if "huggingface.co/papers" in row_data["Paper"]:
            resources = scrape_huggingface_paper_page(row_data["Paper"])
            if resources["spaces"]:
                space_url = resources["spaces"][0]  # Take first space found
                logger.info(f"Space {space_url} inferred from HuggingFace paper page")
                return space_url
        
        # If we have arXiv URL, try the HuggingFace version
        elif "arxiv.org/abs/" in row_data["Paper"]:
            arxiv_id = get_arxiv_id(row_data["Paper"])
            if arxiv_id:
                hf_paper_url = f"https://huggingface.co/papers/{arxiv_id}"
                resources = scrape_huggingface_paper_page(hf_paper_url)
                if resources["spaces"]:
                    space_url = resources["spaces"][0]
                    logger.info(f"Space {space_url} inferred from HuggingFace paper page (via arXiv)")
                    return space_url
    
    # Fallback: try to infer from model using HF API
    if row_data.get("Model") is not None:
        try:
            model_id = row_data["Model"].split("huggingface.co/")[1]
            url = f"{HUGGINGFACE_API_BASE}/spaces?models=" + model_id
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
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
            repo = row_data["Code"].split("github.com/")[1]
            r = make_github_request(f"/repos/{repo}/license")
            if r:
                license_data = r.json()
                if "license" in license_data and license_data["license"] is not None:
                    license_name = license_data["license"]["name"]
                    logger.info(f"License {license_name} inferred from Code")
                    return license_name
        except Exception as e:
            logger.warning(f"Failed to infer license from code: {e}")
    
    return None




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
                         - "CLIP: Connecting Text and Images"
        
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


@rate_limit("mcp_tools")
def find_research_relationships(input_data: str) -> Dict[str, Any]:
    """
    Find ALL related research resources across platforms for comprehensive analysis.
    Optimized version that scrapes once and uses cached results for all inferences.
    
    This is a comprehensive tool that combines all individual inference tools to provide
    a complete picture of a research project's ecosystem. It discovers:
    - Paper URLs (arXiv, HuggingFace)
    - Code repositories (GitHub)
    - Models, datasets, and demo spaces (HuggingFace)
    - Author information and publication dates
    - License information
    
    Args:
        input_data (str): A URL, paper title, or other research-related input
        
    Returns:
        Dict[str, Any]: Dictionary containing all discovered related resources
    """
    if not input_data or not input_data.strip():
        return {"error": "Input data cannot be empty", "success_count": 0, "total_inferences": 10}
    
    try:
        cleaned_input = input_data.strip()
        logger.info(f"Finding research relationships for: {cleaned_input}")
        
        # Initialize results
        relationships = {
            "paper": None,
            "code": None,
            "name": None,
            "authors": [],
            "date": None,
            "model": None,
            "dataset": None,
            "space": None,
            "license": None,
            "field_type": None,
            "success_count": 0,
            "total_inferences": 10
        }
        
        # Create row data and get paper URL first
        row_data = create_row_data(cleaned_input)
        paper_url = infer_paper_from_row(row_data)
        if paper_url:
            relationships["paper"] = paper_url
            relationships["success_count"] += 1
            row_data["Paper"] = paper_url  # Update row data with found paper
        
        # If we have a HuggingFace paper URL, scrape it once for all resources
        hf_resources = None
        if paper_url and "huggingface.co/papers" in paper_url:
            hf_resources = scrape_huggingface_paper_page(paper_url)
        elif paper_url and "arxiv.org/abs/" in paper_url:
            # Try HuggingFace version
            arxiv_id = get_arxiv_id(paper_url)
            if arxiv_id:
                hf_paper_url = f"https://huggingface.co/papers/{arxiv_id}"
                hf_resources = scrape_huggingface_paper_page(hf_paper_url)
        
        # Now perform all other inferences efficiently
        # Code inference
        code_url = infer_code_from_row(row_data)
        if not code_url and hf_resources and hf_resources["code"]:
            code_url = hf_resources["code"][0]
        if code_url:
            relationships["code"] = code_url
            relationships["success_count"] += 1
            row_data["Code"] = code_url
        
        # Name inference
        name = infer_name_from_row(row_data)
        if name:
            relationships["name"] = name
            relationships["success_count"] += 1
        
        # Authors inference
        authors = infer_authors_from_row(row_data)
        if authors:
            relationships["authors"] = authors
            relationships["success_count"] += 1
        
        # Date inference
        date = infer_date_from_row(row_data)
        if date:
            relationships["date"] = date
            relationships["success_count"] += 1
        
        # Model inference (use cached HF resources first)
        model_url = None
        if hf_resources and hf_resources["models"]:
            model_url = hf_resources["models"][0]
        else:
            model_url = infer_model_from_row(row_data)
        if model_url:
            relationships["model"] = model_url
            relationships["success_count"] += 1
        
        # Dataset inference (use cached HF resources first)
        dataset_url = None
        if hf_resources and hf_resources["datasets"]:
            dataset_url = hf_resources["datasets"][0]
        else:
            dataset_url = infer_dataset_from_row(row_data)
        if dataset_url:
            relationships["dataset"] = dataset_url
            relationships["success_count"] += 1
        
        # Space inference (use cached HF resources first)
        space_url = None
        if hf_resources and hf_resources["spaces"]:
            space_url = hf_resources["spaces"][0]
        else:
            space_url = infer_space_from_row(row_data)
        if space_url:
            relationships["space"] = space_url
            relationships["success_count"] += 1
        
        # License inference
        license_info = infer_license_from_row(row_data)
        if license_info:
            relationships["license"] = license_info
            relationships["success_count"] += 1
        
        # Field type inference
        field_type = infer_field_type(cleaned_input)
        if field_type and field_type != "Unknown":
            relationships["field_type"] = field_type
            relationships["success_count"] += 1
        
        logger.info(f"Research relationship analysis completed: {relationships['success_count']}/{relationships['total_inferences']} successful")
        return relationships
        
    except Exception as e:
        logger.error(f"Error finding research relationships: {e}")
        return {"error": str(e), "success_count": 0, "total_inferences": 10}


def format_list_output(items):
    """Format list items for display"""
    if not items or not isinstance(items, list):
        return "None"
    return "\n".join([f"• {item}" for item in items])

def process_research_relationships(input_data):
    """Process research input and return formatted results"""
    if not input_data or not input_data.strip():
        return "Please enter a valid URL or research name", "", "", "", "", "", "", "", "", ""
    
    try:
        result = find_research_relationships(input_data.strip())
        
        # Extract individual fields with fallback to empty string
        paper = result.get("paper", "") or ""
        code = result.get("code", "") or ""
        name = result.get("name", "") or ""
        authors = format_list_output(result.get("authors", []))
        date = result.get("date", "") or ""
        model = result.get("model", "") or ""
        dataset = result.get("dataset", "") or ""
        space = result.get("space", "") or ""
        license_info = result.get("license", "") or ""
        field_type = result.get("field_type", "") or ""
        
        return paper, code, name, authors, date, model, dataset, space, license_info, field_type
        
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}"
        return error_msg, "", "", "", "", "", "", "", "", ""

# Create Gradio interface with both UI and MCP tool exposure
with gr.Blocks(title="Research Tracker MCP Server") as demo:
    gr.Markdown("# 🔬 Research Tracker MCP Server")
    gr.Markdown("""
    **MCP Server for AI Research Intelligence** - This interface demonstrates the `find_research_relationships` tool, which combines all available MCP inference tools into a comprehensive analysis.
    
    ## Individual MCP Tools Available:
    Each output field below represents a separate MCP tool that can be used independently:
    - `infer_paper_url` → Paper URL
    - `infer_code_repository` → Code Repository  
    - `infer_research_name` → Research Name
    - `infer_authors` → Authors
    - `infer_publication_date` → Publication Date
    - `infer_model` → HuggingFace Model
    - `infer_dataset` → HuggingFace Dataset
    - `infer_space` → HuggingFace Space
    - `infer_license` → License
    - `classify_research_url` → Field Type
    
    💡 **For programmatic access**: Use the "Use via API or MCP" button below to integrate these tools with Claude or other AI assistants.
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Demo Input (Paper URL, Repository, or Research Name)",
                placeholder="https://arxiv.org/abs/2506.18787",
                lines=2,
                info="Try: arXiv URLs, HuggingFace paper URLs, GitHub repos, or research titles"
            )
            submit_btn = gr.Button("🔍 Demonstrate find_research_relationships", variant="primary")
    
    gr.Markdown("## Research Relationships")
    
    with gr.Row():
        with gr.Column():
            paper_output = gr.Textbox(label="Paper URL", interactive=False)
            code_output = gr.Textbox(label="Code Repository", interactive=False)
            name_output = gr.Textbox(label="Research Name", interactive=False)
            authors_output = gr.Textbox(label="Authors", lines=3, interactive=False)
            
        with gr.Column():
            date_output = gr.Textbox(label="Publication Date", interactive=False)
            model_output = gr.Textbox(label="HuggingFace Model", interactive=False)
            dataset_output = gr.Textbox(label="HuggingFace Dataset", interactive=False)
            
        with gr.Column():
            space_output = gr.Textbox(label="HuggingFace Space", interactive=False)
            license_output = gr.Textbox(label="License", interactive=False)
            field_type_output = gr.Textbox(label="Field Type", interactive=False)
    
    # Connect the interface with examples
    submit_btn.click(
        fn=process_research_relationships,
        inputs=[input_text],
        outputs=[
            paper_output, code_output, name_output, authors_output, 
            date_output, model_output, dataset_output,
            space_output, license_output, field_type_output
        ]
    )
    
    # Add examples using Gradio's built-in system
    gr.Examples(
        examples=[
            ["https://arxiv.org/abs/2506.18787"],
            ["https://huggingface.co/papers/2010.11929"],
            ["https://github.com/facebookresearch/segment-anything"],
        ],
        inputs=[input_text],
        outputs=[
            paper_output, code_output, name_output, authors_output, 
            date_output, model_output, dataset_output,
            space_output, license_output, field_type_output
        ],
        fn=process_research_relationships,
        cache_examples=False,
        label="Example Inputs"
    )
    
    # Also trigger on Enter key
    input_text.submit(
        fn=process_research_relationships,
        inputs=[input_text],
        outputs=[
            paper_output, code_output, name_output, authors_output, 
            date_output, model_output, dataset_output,
            space_output, license_output, field_type_output
        ]
    )
    
    # Expose all core functions as MCP tools
    gr.api(infer_authors)
    gr.api(infer_paper_url)
    gr.api(infer_code_repository)
    gr.api(infer_research_name)
    gr.api(classify_research_url)
    gr.api(infer_publication_date)
    gr.api(infer_model)
    gr.api(infer_dataset)
    gr.api(infer_space)
    gr.api(infer_license)
    gr.api(find_research_relationships)


if __name__ == "__main__":
    logger.info("Starting Research Tracker MCP Server")
    demo.launch(mcp_server=True, share=False)
