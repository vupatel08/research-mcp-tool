"""
Research Tracker MCP Server - Utility Functions

URL validation, HTTP requests, caching, rate limiting, and helper functions.
"""

import os
import re
import logging
import time
from urllib.parse import urlparse, quote
from typing import List, Dict, Any, Optional, Union
from functools import wraps
from datetime import datetime, timedelta

import requests
import feedparser
from bs4 import BeautifulSoup

from config import (
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY, CACHE_TTL, 
    MAX_URL_LENGTH, RATE_LIMIT_WINDOW, RATE_LIMIT_CALLS,
    ALLOWED_DOMAINS, GITHUB_AUTH, logger
)

# Enhanced cache with TTL for scraping results
_scrape_cache = {}  # {url: {"data": ..., "timestamp": ...}}
_rate_limit_tracker = {}  # {key: [timestamps]}


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
                from config import MCPError
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
    from config import ValidationError, ExternalAPIError
    
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
        return paper_url.split("arxiv.org/abs/")[1].split('.pdf')[0]
    elif "arxiv.org/pdf/" in paper_url:
        return paper_url.split("arxiv.org/pdf/")[1].split('.pdf')[0]
    elif "huggingface.co/papers" in paper_url:
        return paper_url.split("huggingface.co/papers/")[1]
    return None


def clean_url(url):
    """Clean malformed URLs by removing trailing HTML fragments and invalid characters"""
    if not url:
        return url
    
    # Remove HTML closing tags and attributes that often get attached
    import re
    
    # Remove anything after quote marks followed by HTML-like content
    url = re.sub(r'["\']\s*>.*$', '', url)
    
    # Remove trailing HTML fragments
    url = re.sub(r'["\']?\s*</.*$', '', url)
    
    # Remove trailing punctuation and whitespace
    url = url.rstrip('",;\'"()<>[] \t\n\r')
    
    # Basic URL validation - should start with http/https and contain valid characters
    if not re.match(r'^https?://[^\s<>"\'\[\]{}|\\^`]+$', url):
        return None
    
    return url


def is_valid_paper_url(url):
    """Check if a URL is a valid paper URL, excluding badges and non-paper content"""
    if not url:
        return False
    
    url_lower = url.lower()
    
    # Exclude badges, shields, and other non-paper URLs
    if any(pattern in url_lower for pattern in [
        'img.shields.io', 'badge', 'logo', 'icon', 'button',
        'github.com/microsoft/trellis/issues', '/releases/', '/actions/',
        '/wiki/', '/tree/', '/blob/', '.svg', '.png', '.jpg', '.gif'
    ]):
        return False
    
    # Valid paper URL patterns
    if any(pattern in url_lower for pattern in [
        'arxiv.org/abs/', 'arxiv.org/pdf/', 'huggingface.co/papers/'
    ]):
        return True
    
    return False


def select_best_github_repo(github_links, context_keywords=None):
    """Select the best GitHub repository from a list of GitHub URLs"""
    if not github_links:
        return None
    
    if context_keywords is None:
        context_keywords = []
    
    # Score repositories based on various factors
    scored_repos = []
    
    for link in github_links:
        if not link:
            continue
            
        score = 0
        link_lower = link.lower()
        
        # Skip user profiles (github.com/username without repo)
        path_parts = link.split('github.com/')[-1].split('/')
        if len(path_parts) < 2 or not path_parts[1]:
            continue  # Skip user profiles
        
        # Skip issue/PR/wiki pages - prefer main repo
        if any(x in link_lower for x in ['/issues', '/pull', '/wiki', '/releases', '/actions']):
            score -= 10
        
        # Prefer repositories that match context keywords
        for keyword in context_keywords:
            if keyword.lower() in link_lower:
                score += 20
        
        # Prefer Microsoft/official org repos if in a Microsoft context
        if 'microsoft' in link_lower and any(k.lower() in link_lower for k in context_keywords):
            score += 15
        
        # Prefer main branch/root repo URLs
        if link_lower.endswith('.git') or '/tree/' not in link_lower:
            score += 5
        
        scored_repos.append((score, link))
    
    if scored_repos:
        # Return the highest scored repository
        scored_repos.sort(key=lambda x: x[0], reverse=True)
        return scored_repos[0][1]
    
    return None


def extract_links_from_soup(soup, text):
    """Extract both HTML and markdown links from soup and text"""
    html_links = [link.get("href") for link in soup.find_all("a") if link.get("href")]
    link_pattern = re.compile(r"\[.*?\]\((.*?)\)")
    markdown_links = link_pattern.findall(text)
    
    # Also extract direct URLs that aren't in markdown format
    url_pattern = re.compile(r'https?://[^\s\)]+')
    direct_urls = url_pattern.findall(text)
    
    # Combine all links, clean them, and remove duplicates
    all_links = html_links + markdown_links + direct_urls
    cleaned_links = [clean_url(link) for link in all_links if link]
    return list(set([link for link in cleaned_links if link]))


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
        
    except Exception as e:
        logger.error(f"Unexpected error scraping HF paper page: {e}")
        return empty_resources
    
    return resources
