"""
Research Tracker MCP Server - Core Inference Functions

Core inference logic for extracting research metadata from various inputs.
"""

import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests
import feedparser
from bs4 import BeautifulSoup

from config import ARXIV_API_BASE, HUGGINGFACE_API_BASE, HF_TOKEN, GITHUB_AUTH, logger
from utils import (
    get_arxiv_id, is_valid_paper_url, select_best_github_repo, 
    extract_links_from_soup, scrape_huggingface_paper_page,
    make_github_request, cached_request
)

logger = logging.getLogger(__name__)


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
                    except Exception:
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
                if href and is_valid_paper_url(href):
                    logger.info(f"Paper {href} inferred from Project")
                    return href
        except Exception as e:
            logger.debug(f"Failed to scrape project page: {e}")

    # Try GitHub README parsing
    if row_data.get("Code") is not None and "github.com" in row_data["Code"]:
        try:
            repo = row_data["Code"].split("github.com/")[1]
            
            # First try with GitHub API if available
            if GITHUB_AUTH:
                readme_response = make_github_request(f"/repos/{repo}/readme")
                if readme_response:
                    readme = readme_response.json()
                    if readme.get("type") == "file" and readme.get("download_url"):
                        response = cached_request(readme["download_url"])
                        if response:
                            soup = BeautifulSoup(response.text, "html.parser")
                            links = extract_links_from_soup(soup, response.text)
                            for link in links:
                                if link and is_valid_paper_url(link):
                                    logger.info(f"Paper {link} inferred from Code (via GitHub API)")
                                    return link
            
            # Fallback: try scraping the GitHub page directly
            try:
                github_url = row_data["Code"]
                response = cached_request(github_url)
                if response:
                    soup = BeautifulSoup(response.text, "html.parser")
                    links = extract_links_from_soup(soup, response.text)
                    for link in links:
                        if link and is_valid_paper_url(link):
                            logger.info(f"Paper {link} inferred from Code (via GitHub scraping)")
                            return link
            except Exception:
                pass
                
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
            r = requests.get(row_data["Project"], timeout=30)
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
            r = requests.get(row_data["Project"], timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")
            links = extract_links_from_soup(soup, r.text)
            
            # Filter GitHub links
            github_links = []
            for link in links:
                if link:
                    try:
                        url = urlparse(link)
                        if url.scheme in ["http", "https"] and "github.com" in url.netloc:
                            github_links.append(link)
                    except Exception:
                        pass
            
            if github_links:
                # Extract context keywords from the project page
                context_keywords = []
                if soup.title:
                    context_keywords.extend(soup.title.get_text().split())
                
                # Use URL parts as context
                project_url_parts = row_data["Project"].split('/')
                context_keywords.extend([part for part in project_url_parts if part and len(part) > 2])
                
                best_repo = select_best_github_repo(github_links, context_keywords)
                if best_repo:
                    logger.info(f"Code {best_repo} inferred from Project")
                    return best_repo
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
            r = requests.get(url, timeout=30)
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
