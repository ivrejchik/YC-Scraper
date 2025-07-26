"""
LinkedIn Profile Discovery Module

This module provides core data structures and utilities for discovering LinkedIn
company profiles through search queries and matching them with YC company data.
"""

import re
import time
import random
import logging
import requests
import difflib
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, urlunparse, quote_plus
from bs4 import BeautifulSoup

try:
    from googlesearch import search as google_search
    GOOGLESEARCH_AVAILABLE = True
except ImportError:
    GOOGLESEARCH_AVAILABLE = False
    logger.warning("googlesearch-python not available, falling back to manual search")

from .logging_config import get_logger, log_performance, log_data_operation, log_exception

logger = get_logger(__name__)


@dataclass
class LinkedInResult:
    """
    Represents a LinkedIn company profile discovered through search.
    
    Attributes:
        company_name: Company name extracted from LinkedIn profile
        linkedin_url: LinkedIn company page URL
        description: Company description from LinkedIn
        confidence_score: Confidence score for company name matching (0.0-1.0)
    """
    company_name: str
    linkedin_url: str
    description: str
    confidence_score: float
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not isinstance(self.confidence_score, (int, float)):
            raise ValueError("confidence_score must be a number")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        if self.linkedin_url and not is_valid_linkedin_url(self.linkedin_url):
            raise ValueError(f"Invalid LinkedIn URL: {self.linkedin_url}")


@dataclass
class MatchResult:
    """
    Result of matching LinkedIn profiles with YC API companies.
    
    Attributes:
        matched_companies: YC companies with LinkedIn URLs filled
        new_companies: LinkedIn-only companies not found in YC API
        unmatched_yc: YC companies without LinkedIn matches
        unmatched_linkedin: LinkedIn results without YC matches
    """
    matched_companies: List[dict]
    new_companies: List[dict]
    unmatched_yc: List[dict]
    unmatched_linkedin: List[LinkedInResult]


def is_valid_linkedin_url(url: str) -> bool:
    """
    Validate if a URL is a valid LinkedIn company or profile URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid LinkedIn URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    if not url:
        return False
    
    try:
        parsed = urlparse(url)
        
        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Must be HTTP or HTTPS
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Must be LinkedIn domain
        domain = parsed.netloc.lower()
        valid_domains = [
            'linkedin.com',
            'www.linkedin.com',
            'm.linkedin.com'
        ]
        
        if domain not in valid_domains:
            return False
        
        # Path should indicate company or profile page
        path = parsed.path.lower()
        valid_path_patterns = [
            r'^/company/',
            r'^/in/',
            r'^/pub/',
            r'^/profile/',
            r'^/companies/',
            r'^/$',  # Root LinkedIn URL
            r'^$'    # Empty path
        ]
        
        # Check if path matches any valid pattern
        for pattern in valid_path_patterns:
            if re.match(pattern, path):
                return True
        
        return False
        
    except Exception:
        return False


def normalize_linkedin_url(url: str) -> str:
    """
    Normalize LinkedIn URL to a standard format.
    
    Args:
        url: LinkedIn URL to normalize
        
    Returns:
        Normalized LinkedIn URL
    """
    if not url or not isinstance(url, str):
        return ""
    
    url = url.strip()
    if not url:
        return ""
    
    try:
        parsed = urlparse(url)
        
        # Ensure HTTPS scheme
        scheme = 'https'
        
        # Normalize domain to www.linkedin.com
        netloc = parsed.netloc.lower()
        if netloc in ['linkedin.com', 'm.linkedin.com']:
            netloc = 'www.linkedin.com'
        elif netloc == 'www.linkedin.com':
            netloc = 'www.linkedin.com'
        else:
            # Keep original if not a standard LinkedIn domain
            netloc = parsed.netloc
        
        # Clean up path - remove trailing slash unless it's root
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'
        
        # Remove query parameters and fragments for cleaner URLs
        normalized = urlunparse((
            scheme,
            netloc,
            path,
            '',  # params
            '',  # query
            ''   # fragment
        ))
        
        return normalized
        
    except Exception:
        return url


def extract_company_name_from_linkedin_url(url: str) -> Optional[str]:
    """
    Extract company name from LinkedIn company URL.
    
    Args:
        url: LinkedIn company URL
        
    Returns:
        Company name if extractable, None otherwise
    """
    if not url or not is_valid_linkedin_url(url):
        return None
    
    try:
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        # Handle /company/company-name format
        if path.startswith('company/'):
            company_slug = path[8:]  # Remove 'company/' prefix
            if company_slug:
                # Convert slug to readable name (basic conversion)
                name = company_slug.replace('-', ' ').replace('_', ' ')
                # Capitalize words
                name = ' '.join(word.capitalize() for word in name.split())
                return name
        
        return None
        
    except Exception:
        return None


def validate_linkedin_result(result: LinkedInResult) -> List[str]:
    """
    Validate LinkedInResult data integrity.
    
    Args:
        result: LinkedInResult instance to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate company name
    if not result.company_name or not isinstance(result.company_name, str):
        errors.append("company_name must be a non-empty string")
    elif len(result.company_name.strip()) == 0:
        errors.append("company_name cannot be empty or whitespace only")
    elif len(result.company_name) > 200:
        errors.append("company_name must be 200 characters or less")
    
    # Validate LinkedIn URL
    if not result.linkedin_url or not isinstance(result.linkedin_url, str):
        errors.append("linkedin_url must be a non-empty string")
    elif not is_valid_linkedin_url(result.linkedin_url):
        errors.append(f"Invalid LinkedIn URL: {result.linkedin_url}")
    
    # Validate description
    if result.description is not None:
        if not isinstance(result.description, str):
            errors.append("description must be a string")
        elif len(result.description) > 1000:
            errors.append("description must be 1000 characters or less")
    
    # Validate confidence score
    if not isinstance(result.confidence_score, (int, float)):
        errors.append("confidence_score must be a number")
    elif not (0.0 <= result.confidence_score <= 1.0):
        errors.append("confidence_score must be between 0.0 and 1.0")
    
    return errors


def create_linkedin_result(company_name: str, linkedin_url: str, 
                          description: str = "", confidence_score: float = 0.0) -> LinkedInResult:
    """
    Create a LinkedInResult with validation.
    
    Args:
        company_name: Company name
        linkedin_url: LinkedIn company URL
        description: Company description (optional)
        confidence_score: Matching confidence score (optional)
        
    Returns:
        LinkedInResult instance
        
    Raises:
        ValueError: If validation fails
    """
    # Normalize the LinkedIn URL
    normalized_url = normalize_linkedin_url(linkedin_url)
    
    result = LinkedInResult(
        company_name=company_name.strip(),
        linkedin_url=normalized_url,
        description=description.strip() if description else "",
        confidence_score=float(confidence_score)
    )
    
    # Validate the result
    validation_errors = validate_linkedin_result(result)
    if validation_errors:
        raise ValueError(f"Invalid LinkedIn result: {', '.join(validation_errors)}")
    
    return result


class LinkedInSearchClient:
    """
    Client for searching LinkedIn company profiles using web scraping.
    
    This class implements web scraping for "site:linkedin.com YC [batch]" search queries
    to discover LinkedIn company profiles and extract company information.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LinkedIn search client.
        
        Args:
            config: Configuration dictionary with search settings
        """
        self.config = config or {}
        
        # Extract configuration settings
        linkedin_config = self.config.get('linkedin', {})
        discovery_config = self.config.get('linkedin_discovery', {})
        rate_limiting_config = self.config.get('rate_limiting', {})
        
        # Test mode configuration
        self.test_mode = discovery_config.get('test_mode', False)
        
        # User agents for rotation
        self.user_agents = linkedin_config.get('user_agents', [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ])
        
        # HTTP headers
        self.headers = linkedin_config.get('headers', {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        })
        
        # Search configuration
        self.search_query_template = discovery_config.get('search_query_template', 
                                                         "site:linkedin.com {company_name} YC {batch}")
        self.max_results_per_search = discovery_config.get('max_results_per_search', 10)
        
        # Rate limiting configuration
        self.search_delay_min = discovery_config.get('search_delay_min', 2.0)
        self.search_delay_max = discovery_config.get('search_delay_max', 4.0)
        self.max_retries = discovery_config.get('max_retries', 3)
        self.retry_delay = discovery_config.get('retry_delay', 5.0)
        self.request_timeout = rate_limiting_config.get('request_timeout', 30)
        
        # Session for connection reuse (only if not in test mode)
        if not self.test_mode:
            self.session = requests.Session()
            self.session.headers.update(self.headers)
        
        # Initialize test data if in test mode
        if self.test_mode:
            self._initialize_test_data()
        
        # Log initialization details
        logger.info(f"LinkedInSearchClient initialized (test_mode: {self.test_mode})")
        if not self.test_mode:
            logger.debug(f"Configuration: max_results={self.max_results_per_search}, "
                        f"delay_range={self.search_delay_min}-{self.search_delay_max}s, "
                        f"max_retries={self.max_retries}, timeout={self.request_timeout}s")
    
    def _initialize_test_data(self):
        """Initialize test data for test mode."""
        try:
            # Load existing company data to use their LinkedIn URLs for testing
            from .data_manager import DataManager
            data_manager = DataManager()
            companies_df = data_manager.load_existing_data()
            
            # Extract companies with LinkedIn URLs
            self.test_linkedin_data = []
            for _, company in companies_df.iterrows():
                # Handle potential NaN values
                linkedin_url = company.get('linkedin_url', '')
                if pd.isna(linkedin_url):
                    linkedin_url = ''
                else:
                    linkedin_url = str(linkedin_url).strip()
                
                # Only include companies with valid LinkedIn URLs
                if linkedin_url and is_valid_linkedin_url(linkedin_url):
                    # Handle other potential NaN values
                    description = company.get('description', '')
                    if pd.isna(description):
                        description = ''
                    else:
                        description = str(description)
                    
                    website = company.get('website', '')
                    if pd.isna(website):
                        website = ''
                    else:
                        website = str(website)
                    
                    self.test_linkedin_data.append({
                        'name': str(company['name']),
                        'linkedin_url': linkedin_url,
                        'description': description,
                        'website': website
                    })
            
            logger.info(f"Test mode: Loaded {len(self.test_linkedin_data)} LinkedIn URLs for testing")
            
        except Exception as e:
            logger.warning(f"Failed to load test data: {e}")
            import traceback
            logger.debug(f"Test data loading error details: {traceback.format_exc()}")
            self.test_linkedin_data = []
    
    def _search_test_mode(self, search_term: str) -> List[LinkedInResult]:
        """
        Simulate search results using existing LinkedIn data for testing.
        
        Args:
            search_term: Search term (company name or batch identifier)
            
        Returns:
            List of LinkedInResult objects from test data
        """
        if not hasattr(self, 'test_linkedin_data') or not self.test_linkedin_data:
            logger.warning("No test data available for test mode")
            return []
        
        results = []
        search_term_lower = search_term.lower()
        
        # Apply rate limiting even in test mode to simulate real behavior
        time.sleep(random.uniform(0.1, 0.3))
        
        # Find matching companies from test data
        for company_data in self.test_linkedin_data[:self.max_results_per_search]:
            company_name = company_data['name']
            
            # Simple matching logic for test mode
            # Match if search term is in company name or if it's a batch search
            if (search_term_lower in company_name.lower() or 
                'yc' in search_term_lower or 
                len(results) < 3):  # Always return some results for testing
                
                try:
                    result = LinkedInResult(
                        company_name=company_name,
                        linkedin_url=company_data['linkedin_url'],
                        description=company_data.get('description', ''),
                        confidence_score=0.8
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error creating test result for {company_name}: {e}")
                    continue
        
        logger.debug(f"Test mode: Generated {len(results)} results for '{search_term}'")
        return results
    
    def search_yc_companies(self, batch: str = "S25") -> List[LinkedInResult]:
        """
        Search for YC companies on LinkedIn using batch-specific search query.
        
        Args:
            batch: YC batch identifier (e.g., "S25", "W25")
            
        Returns:
            List of LinkedInResult objects found in search
        """
        start_time = time.time()
        logger.info(f"Starting LinkedIn search for YC {batch} companies")
        
        # Use test mode if enabled
        if self.test_mode:
            return self._search_test_mode(f"YC {batch}")
        
        # Construct search query for the entire batch
        search_query = f"site:linkedin.com YC {batch}"
        logger.info(f"Search query: {search_query}")
        
        try:
            results = self._perform_search(search_query)
            
            # Log performance and results
            duration = time.time() - start_time
            log_performance(logger, f"LinkedIn search for YC {batch}", duration, {
                'results_count': len(results),
                'search_query': search_query
            })
            
            # Log data operation results
            log_data_operation(logger, f"LinkedIn search YC {batch}", len(results), True)
            
            logger.info(f"Successfully found {len(results)} LinkedIn profiles for YC {batch}")
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            log_exception(logger, f"Error searching for YC {batch} companies after {duration:.2f}s")
            log_data_operation(logger, f"LinkedIn search YC {batch}", 0, False, [str(e)])
            return []
    
    def search_company(self, company_name: str, batch: str = "S25") -> List[LinkedInResult]:
        """
        Search for a specific company on LinkedIn.
        
        Args:
            company_name: Name of the company to search for
            batch: YC batch identifier
            
        Returns:
            List of LinkedInResult objects found in search
        """
        start_time = time.time()
        logger.debug(f"Searching LinkedIn for company: {company_name} (YC {batch})")
        
        # Use test mode if enabled
        if self.test_mode:
            return self._search_test_mode(company_name)
        
        # Construct search query using template
        search_query = self.search_query_template.format(
            company_name=company_name,
            batch=batch
        )
        logger.debug(f"Company search query: {search_query}")
        
        try:
            results = self._perform_search(search_query)
            
            # Log performance and results
            duration = time.time() - start_time
            log_performance(logger, f"LinkedIn company search: {company_name}", duration, {
                'results_count': len(results),
                'search_query': search_query,
                'batch': batch
            })
            
            logger.debug(f"Found {len(results)} results for {company_name}")
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            log_exception(logger, f"Error searching for company {company_name} after {duration:.2f}s")
            return []
    
    def _perform_search(self, query: str) -> List[LinkedInResult]:
        """
        Perform the actual search using multiple search engines with fallback.
        
        Args:
            query: Search query string
            
        Returns:
            List of LinkedInResult objects
        """
        results = []
        
        # Try multiple search engines in order of preference
        search_engines = [
            self._search_with_googlesearch_library,  # Use googlesearch library first
            self._search_with_google,
            self._search_with_bing,
            self._search_with_duckduckgo
        ]
        
        for search_engine in search_engines:
            try:
                results = search_engine(query)
                if results:
                    logger.debug(f"Successfully found {len(results)} results using {search_engine.__name__}")
                    break
                else:
                    logger.debug(f"No results from {search_engine.__name__}, trying next engine")
            except Exception as e:
                logger.warning(f"Search engine {search_engine.__name__} failed: {e}")
                continue
        
        if not results:
            logger.warning(f"No results found for query: {query}")
        
        return results
    
    def _search_with_googlesearch_library(self, query: str) -> List[LinkedInResult]:
        """
        Search using the googlesearch-python library.
        
        This is the most reliable method as it uses the official Google search API
        through the googlesearch library.
        
        Args:
            query: Search query string
            
        Returns:
            List of LinkedInResult objects
        """
        if not GOOGLESEARCH_AVAILABLE:
            logger.debug("googlesearch library not available, skipping")
            return []
        
        results = []
        
        try:
            logger.debug(f"Performing Google search using googlesearch library: {query}")
            
            # Apply rate limiting
            self._apply_rate_limiting()
            
            # Use googlesearch library to get search results
            search_results = google_search(
                query,
                num_results=self.max_results_per_search
            )
            
            # Process each search result
            for url in search_results:
                try:
                    # Check if this is a LinkedIn URL
                    if 'linkedin.com' in url and is_valid_linkedin_url(url):
                        # Extract company information from the URL
                        linkedin_result = self._extract_company_info_from_url(url)
                        if linkedin_result:
                            results.append(linkedin_result)
                            
                except Exception as e:
                    logger.warning(f"Error processing search result URL {url}: {e}")
                    continue
            
            logger.debug(f"Successfully found {len(results)} LinkedIn URLs using googlesearch library")
            return results
            
        except Exception as e:
            logger.warning(f"googlesearch library search failed: {e}")
            return []
    
    def _extract_company_info_from_url(self, linkedin_url: str) -> Optional[LinkedInResult]:
        """
        Extract company information from a LinkedIn URL without scraping the page.
        
        This method extracts basic company information from the URL structure
        and creates a LinkedInResult object.
        
        Args:
            linkedin_url: LinkedIn company URL
            
        Returns:
            LinkedInResult object if extraction successful, None otherwise
        """
        try:
            # Normalize the LinkedIn URL
            normalized_url = normalize_linkedin_url(linkedin_url)
            
            if not normalized_url:
                return None
            
            # Extract company name from URL
            company_name = extract_company_name_from_linkedin_url(normalized_url)
            
            # If we can't extract a name from URL, use a generic approach
            if not company_name:
                # Try to extract from URL path
                parsed = urlparse(normalized_url)
                path_parts = parsed.path.strip('/').split('/')
                
                if len(path_parts) >= 2 and path_parts[0] == 'company':
                    # Convert company slug to readable name
                    company_slug = path_parts[1]
                    company_name = company_slug.replace('-', ' ').replace('_', ' ')
                    company_name = ' '.join(word.capitalize() for word in company_name.split())
                else:
                    # Fallback to generic name
                    company_name = "LinkedIn Company"
            
            # Create LinkedInResult with basic information
            result = LinkedInResult(
                company_name=company_name,
                linkedin_url=normalized_url,
                description="",  # No description available from URL alone
                confidence_score=0.7  # Medium confidence since we only have URL info
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Error extracting company info from URL {linkedin_url}: {e}")
            return None
    
    def _search_with_google(self, query: str) -> List[LinkedInResult]:
        """Search using Google."""
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self._apply_rate_limiting()
                
                # Rotate user agent
                headers = self.headers.copy()
                headers['User-Agent'] = random.choice(self.user_agents)
                
                logger.debug(f"Performing Google search (attempt {attempt + 1}): {query}")
                
                # Make the search request
                response = self.session.get(
                    search_url,
                    headers=headers,
                    timeout=self.request_timeout,
                    allow_redirects=True
                )
                
                response.raise_for_status()
                
                # Parse Google search results
                results = self._parse_google_results(response.text, query)
                
                if results:
                    logger.debug(f"Successfully parsed {len(results)} Google results")
                    return results
                    
            except requests.RequestException as e:
                logger.warning(f"Google search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All Google search attempts failed for query: {query}")
            
            except Exception as e:
                logger.error(f"Unexpected error during Google search: {e}")
                break
        
        return []
    
    def _search_with_bing(self, query: str) -> List[LinkedInResult]:
        """Search using Bing."""
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
        
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self._apply_rate_limiting()
                
                # Rotate user agent
                headers = self.headers.copy()
                headers['User-Agent'] = random.choice(self.user_agents)
                
                logger.debug(f"Performing Bing search (attempt {attempt + 1}): {query}")
                
                # Make the search request
                response = self.session.get(
                    search_url,
                    headers=headers,
                    timeout=self.request_timeout,
                    allow_redirects=True
                )
                
                response.raise_for_status()
                
                # Parse Bing search results
                results = self._parse_bing_results(response.text, query)
                
                if results:
                    logger.debug(f"Successfully parsed {len(results)} Bing results")
                    return results
                    
            except requests.RequestException as e:
                logger.warning(f"Bing search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All Bing search attempts failed for query: {query}")
            
            except Exception as e:
                logger.error(f"Unexpected error during Bing search: {e}")
                break
        
        return []
    
    def _search_with_duckduckgo(self, query: str) -> List[LinkedInResult]:
        """Search using DuckDuckGo (original implementation)."""
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self._apply_rate_limiting()
                
                # Rotate user agent
                headers = self.headers.copy()
                headers['User-Agent'] = random.choice(self.user_agents)
                
                logger.debug(f"Performing DuckDuckGo search (attempt {attempt + 1}): {query}")
                
                # Make the search request
                response = self.session.get(
                    search_url,
                    headers=headers,
                    timeout=self.request_timeout,
                    allow_redirects=True
                )
                
                response.raise_for_status()
                
                # Parse DuckDuckGo search results
                results = self._parse_duckduckgo_results(response.text, query)
                
                if results:
                    logger.debug(f"Successfully parsed {len(results)} DuckDuckGo results")
                    return results
                    
            except requests.RequestException as e:
                logger.warning(f"DuckDuckGo search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All DuckDuckGo search attempts failed for query: {query}")
            
            except Exception as e:
                logger.error(f"Unexpected error during DuckDuckGo search: {e}")
                break
        
        return []
    
    def _parse_google_results(self, html_content: str, query: str) -> List[LinkedInResult]:
        """Parse Google search results."""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Google search result selectors
            result_links = soup.find_all('a', href=True)
            
            for link in result_links[:self.max_results_per_search]:
                href = link.get('href', '')
                
                # Clean up Google redirect URLs
                if href.startswith('/url?q='):
                    # Extract actual URL from Google redirect
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    if 'q' in parsed:
                        href = parsed['q'][0]
                
                # Check if this is a LinkedIn URL
                if 'linkedin.com' in href and is_valid_linkedin_url(href):
                    # Extract company information
                    linkedin_result = self.extract_company_info(link, href)
                    if linkedin_result:
                        results.append(linkedin_result)
            
            logger.debug(f"Extracted {len(results)} LinkedIn URLs from Google results")
            
        except Exception as e:
            logger.error(f"Error parsing Google search results: {e}")
        
        return results
    
    def _parse_bing_results(self, html_content: str, query: str) -> List[LinkedInResult]:
        """Parse Bing search results."""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Bing search result selectors
            result_links = soup.find_all('a', href=True)
            
            for link in result_links[:self.max_results_per_search]:
                href = link.get('href', '')
                
                # Check if this is a LinkedIn URL
                if 'linkedin.com' in href and is_valid_linkedin_url(href):
                    # Extract company information
                    linkedin_result = self.extract_company_info(link, href)
                    if linkedin_result:
                        results.append(linkedin_result)
            
            logger.debug(f"Extracted {len(results)} LinkedIn URLs from Bing results")
            
        except Exception as e:
            logger.error(f"Error parsing Bing search results: {e}")
        
        return results
    
    def _parse_duckduckgo_results(self, html_content: str, query: str) -> List[LinkedInResult]:
        """Parse DuckDuckGo search results."""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find search result links
            # DuckDuckGo uses different selectors than Google
            result_links = soup.find_all('a', class_='result__a')
            
            for link in result_links[:self.max_results_per_search]:
                href = link.get('href', '')
                
                # Check if this is a LinkedIn URL
                if 'linkedin.com' in href and is_valid_linkedin_url(href):
                    # Extract company information
                    linkedin_result = self.extract_company_info(link, href)
                    if linkedin_result:
                        results.append(linkedin_result)
            
            logger.debug(f"Extracted {len(results)} LinkedIn URLs from DuckDuckGo results")
            
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo search results: {e}")
        
        return results
    
    def extract_company_info(self, link_element, linkedin_url: str) -> Optional[LinkedInResult]:
        """
        Extract company information from a search result link.
        
        Args:
            link_element: BeautifulSoup element containing the link
            linkedin_url: LinkedIn URL from the link
            
        Returns:
            LinkedInResult object if extraction successful, None otherwise
        """
        try:
            # Normalize the LinkedIn URL
            normalized_url = normalize_linkedin_url(linkedin_url)
            
            if not normalized_url:
                return None
            
            # Extract company name from URL as fallback
            company_name = extract_company_name_from_linkedin_url(normalized_url)
            
            # Try to get company name from link text
            link_text = link_element.get_text(strip=True)
            if link_text and len(link_text) < 200:
                # Use link text if it looks like a company name
                if not any(word in link_text.lower() for word in ['linkedin', 'profile', 'company']):
                    company_name = link_text
            
            # Get description from parent element if available
            description = ""
            parent = link_element.find_parent()
            if parent:
                desc_element = parent.find('span', class_='result__snippet')
                if desc_element:
                    description = desc_element.get_text(strip=True)[:1000]  # Limit length
            
            # Use URL-based name if no better name found
            if not company_name:
                company_name = extract_company_name_from_linkedin_url(normalized_url) or "Unknown Company"
            
            # Create and validate result
            result = LinkedInResult(
                company_name=company_name,
                linkedin_url=normalized_url,
                description=description,
                confidence_score=0.8  # Default confidence for search results
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Error extracting company info from {linkedin_url}: {e}")
            return None
    
    def validate_linkedin_url(self, url: str) -> bool:
        """
        Validate if a URL is a valid LinkedIn company URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid LinkedIn URL, False otherwise
        """
        return is_valid_linkedin_url(url)
    
    def _apply_rate_limiting(self):
        """Apply rate limiting delay between requests."""
        delay = random.uniform(self.search_delay_min, self.search_delay_max)
        logger.debug(f"Applying rate limiting delay: {delay:.2f} seconds")
        time.sleep(delay)
    
    def close(self):
        """Close the HTTP session."""
        if hasattr(self, 'session'):
            self.session.close()
            logger.debug("LinkedInSearchClient session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class LinkedInDiscoveryService:
    """
    Main orchestrator for LinkedIn profile discovery and data merging.
    
    This service coordinates the entire discovery workflow:
    1. Fetches current YC API data
    2. Searches LinkedIn for YC companies
    3. Matches LinkedIn profiles with YC companies
    4. Merges data and creates new company records
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LinkedIn discovery service.
        
        Args:
            config: Configuration dictionary with service settings
        """
        self.config = config or {}
        
        # Initialize components
        self.linkedin_client = LinkedInSearchClient(config)
        self.company_matcher = CompanyMatcher(config)
        
        # Import YcClient and DataManager here to avoid circular imports
        from .yc_client import YcClient
        from .data_manager import DataManager
        from .models import Company, create_company_from_yc_data
        
        self.yc_client = YcClient()
        self.data_manager = DataManager()
        self.Company = Company
        self.create_company_from_yc_data = create_company_from_yc_data
        
        logger.info("LinkedInDiscoveryService initialized")
    
    def discover_and_merge(self, batch: str = "S25") -> Dict[str, Any]:
        """
        Main discovery workflow that fetches YC data, searches LinkedIn,
        matches companies, and merges the results.
        
        Args:
            batch: YC batch identifier (e.g., "S25", "W25")
            
        Returns:
            Dictionary containing discovery results and statistics
        """
        start_time = time.time()
        logger.info(f"=== Starting LinkedIn discovery and merge for YC {batch} ===")
        
        # Initialize progress tracking
        progress = {
            'total_steps': 4,
            'current_step': 0,
            'step_names': ['Fetch YC API data', 'Search LinkedIn', 'Match and merge', 'Finalize results']
        }
        
        try:
            # Step 1: Fetch YC API data
            progress['current_step'] = 1
            step_start = time.time()
            logger.info(f"Step {progress['current_step']}/{progress['total_steps']}: {progress['step_names'][0]}")
            
            yc_data = self.fetch_yc_api_data(batch)
            
            step_duration = time.time() - step_start
            log_performance(logger, f"Step 1 - Fetch YC API data", step_duration, {
                'companies_fetched': len(yc_data),
                'batch': batch
            })
            logger.info(f"✓ Step 1 completed: Fetched {len(yc_data)} companies from YC API")
            
            # Step 2: Search LinkedIn for YC companies
            progress['current_step'] = 2
            step_start = time.time()
            logger.info(f"Step {progress['current_step']}/{progress['total_steps']}: {progress['step_names'][1]}")
            
            linkedin_results = self.linkedin_client.search_yc_companies(batch)
            
            step_duration = time.time() - step_start
            log_performance(logger, f"Step 2 - Search LinkedIn", step_duration, {
                'linkedin_profiles_found': len(linkedin_results),
                'batch': batch
            })
            logger.info(f"✓ Step 2 completed: Found {len(linkedin_results)} LinkedIn profiles")
            
            # Step 3: Match and merge data
            progress['current_step'] = 3
            step_start = time.time()
            logger.info(f"Step {progress['current_step']}/{progress['total_steps']}: {progress['step_names'][2]}")
            
            merged_companies = self.match_and_merge_data(yc_data, linkedin_results)
            
            step_duration = time.time() - step_start
            log_performance(logger, f"Step 3 - Match and merge", step_duration, {
                'merged_companies': len(merged_companies),
                'yc_input': len(yc_data),
                'linkedin_input': len(linkedin_results)
            })
            logger.info(f"✓ Step 3 completed: Created {len(merged_companies)} merged company records")
            
            # Step 4: Prepare results and statistics
            progress['current_step'] = 4
            step_start = time.time()
            logger.info(f"Step {progress['current_step']}/{progress['total_steps']}: {progress['step_names'][3]}")
            
            # Calculate detailed statistics
            total_duration = time.time() - start_time
            results = {
                'success': True,
                'batch': batch,
                'yc_companies_count': len(yc_data),
                'linkedin_results_count': len(linkedin_results),
                'merged_companies_count': len(merged_companies),
                'companies': merged_companies,
                'processing_time_seconds': round(total_duration, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            step_duration = time.time() - step_start
            log_performance(logger, f"Step 4 - Finalize results", step_duration)
            
            # Log comprehensive summary statistics
            self._log_discovery_summary(results, total_duration)
            
            logger.info(f"✓ Step 4 completed: Results finalized")
            logger.info(f"=== Discovery and merge completed successfully for YC {batch} in {total_duration:.2f}s ===")
            
            return results
            
        except Exception as e:
            total_duration = time.time() - start_time
            log_exception(logger, f"Error in discover_and_merge for YC {batch} after {total_duration:.2f}s")
            
            # Log failure statistics
            log_data_operation(logger, f"LinkedIn discovery YC {batch}", 0, False, [str(e)])
            
            return {
                'success': False,
                'error': str(e),
                'batch': batch,
                'processing_time_seconds': round(total_duration, 2),
                'failed_at_step': progress['current_step'],
                'step_name': progress['step_names'][progress['current_step'] - 1] if progress['current_step'] > 0 else 'initialization',
                'timestamp': datetime.now().isoformat()
            }
    
    def fetch_yc_api_data(self, batch: str = "S25") -> List[Dict[str, Any]]:
        """
        Fetch current YC company data using existing YcClient.
        
        Args:
            batch: YC batch identifier
            
        Returns:
            List of YC company dictionaries
        """
        start_time = time.time()
        logger.info(f"Fetching YC API data for batch {batch}")
        
        try:
            # Parse and set the batch on the YC client
            logger.debug(f"Parsing batch identifier: {batch}")
            
            if batch == "S25":
                season, year = "Summer", 2025
            elif batch == "W25":
                season, year = "Winter", 2025
            elif batch == "S24":
                season, year = "Summer", 2024
            elif batch == "W24":
                season, year = "Winter", 2024
            else:
                # Parse batch format like "S25" or "W24"
                if len(batch) >= 3:
                    season_code = batch[0].upper()
                    year_code = batch[1:]
                    
                    season = "Summer" if season_code == "S" else "Winter"
                    year = 2000 + int(year_code) if int(year_code) < 50 else 1900 + int(year_code)
                    
                    logger.debug(f"Parsed batch {batch} as {season} {year}")
                else:
                    logger.warning(f"Unknown batch format: {batch}, defaulting to Summer 2025")
                    season, year = "Summer", 2025
            
            # Set the batch on the YC client
            logger.debug(f"Setting YC client batch to {season} {year}")
            self.yc_client.set_batch(season, year)
            
            # Fetch the data
            logger.debug("Calling YC API to fetch batch data")
            yc_companies = self.yc_client.fetch_yc_batch_data()
            
            # Log performance and results
            duration = time.time() - start_time
            log_performance(logger, f"Fetch YC API data for {batch}", duration, {
                'companies_fetched': len(yc_companies),
                'batch': batch,
                'season': season,
                'year': year
            })
            
            # Log data operation success
            log_data_operation(logger, f"Fetch YC API data {batch}", len(yc_companies), True)
            
            logger.info(f"✓ Successfully fetched {len(yc_companies)} companies from YC API for {batch}")
            return yc_companies
            
        except Exception as e:
            duration = time.time() - start_time
            log_exception(logger, f"Error fetching YC API data for {batch} after {duration:.2f}s")
            log_data_operation(logger, f"Fetch YC API data {batch}", 0, False, [str(e)])
            raise
    
    def match_and_merge_data(self, yc_data: List[Dict[str, Any]], 
                           linkedin_results: List[LinkedInResult]) -> List[Any]:
        """
        Match LinkedIn profiles with YC companies and merge the data.
        
        This method:
        1. Matches LinkedIn profiles with YC companies to fill missing LinkedIn URLs
        2. Creates new Company records for LinkedIn-only companies
        3. Returns a list of all companies (YC + LinkedIn-only)
        
        Args:
            yc_data: List of YC company dictionaries
            linkedin_results: List of LinkedIn search results
            
        Returns:
            List of Company objects with merged data
        """
        start_time = time.time()
        logger.info(f"Starting match and merge: {len(yc_data)} YC companies with {len(linkedin_results)} LinkedIn results")
        
        # Track processing statistics
        processing_stats = {
            'matched_companies': 0,
            'unmatched_yc_companies': 0,
            'new_linkedin_companies': 0,
            'processing_errors': []
        }
        
        try:
            # Use CompanyMatcher to perform the matching
            logger.debug("Performing company matching using CompanyMatcher")
            match_result = self.company_matcher.match_companies(yc_data, linkedin_results)
            
            # Log matching statistics
            match_stats = self.company_matcher.get_match_statistics(match_result)
            logger.info(f"Matching statistics: {match_stats}")
            
            merged_companies = []
            
            # Process matched companies (YC companies with LinkedIn URLs filled)
            logger.debug(f"Processing {len(match_result.matched_companies)} matched companies")
            for i, yc_company_dict in enumerate(match_result.matched_companies):
                try:
                    # Create Company object from YC data
                    company = self.create_company_from_yc_data(yc_company_dict)
                    merged_companies.append(company)
                    processing_stats['matched_companies'] += 1
                    
                    if i % 10 == 0:  # Log progress every 10 companies
                        logger.debug(f"Processed {i+1}/{len(match_result.matched_companies)} matched companies")
                    
                except Exception as e:
                    error_msg = f"Error creating company from matched data: {e}"
                    logger.warning(error_msg)
                    processing_stats['processing_errors'].append(error_msg)
                    continue
            
            # Process unmatched YC companies (no LinkedIn URL found)
            logger.debug(f"Processing {len(match_result.unmatched_yc)} unmatched YC companies")
            for i, yc_company_dict in enumerate(match_result.unmatched_yc):
                try:
                    # Create Company object from YC data (linkedin_url will be empty)
                    company = self.create_company_from_yc_data(yc_company_dict)
                    merged_companies.append(company)
                    processing_stats['unmatched_yc_companies'] += 1
                    
                    if i % 10 == 0:  # Log progress every 10 companies
                        logger.debug(f"Processed {i+1}/{len(match_result.unmatched_yc)} unmatched YC companies")
                    
                except Exception as e:
                    error_msg = f"Error creating company from unmatched YC data: {e}"
                    logger.warning(error_msg)
                    processing_stats['processing_errors'].append(error_msg)
                    continue
            
            # Process new companies (LinkedIn-only companies not found in YC API)
            logger.debug(f"Processing {len(match_result.new_companies)} new LinkedIn-only companies")
            for i, new_company_dict in enumerate(match_result.new_companies):
                try:
                    # Create Company object for LinkedIn-only company
                    company = self.Company(
                        name=new_company_dict.get('name', ''),
                        website=new_company_dict.get('website', ''),
                        description=new_company_dict.get('description', ''),
                        yc_page='',  # Empty for LinkedIn-only companies
                        linkedin_url=new_company_dict.get('linkedin_url', ''),
                        yc_batch_on_linkedin=False,  # Will be updated by LinkedIn scraper
                        linkedin_only=True,  # This is a LinkedIn-only company
                        last_updated=datetime.now().isoformat()
                    )
                    merged_companies.append(company)
                    processing_stats['new_linkedin_companies'] += 1
                    
                    if i % 10 == 0:  # Log progress every 10 companies
                        logger.debug(f"Processed {i+1}/{len(match_result.new_companies)} new LinkedIn companies")
                    
                except Exception as e:
                    error_msg = f"Error creating LinkedIn-only company: {e}"
                    logger.warning(error_msg)
                    processing_stats['processing_errors'].append(error_msg)
                    continue
            
            # Log comprehensive merge results
            duration = time.time() - start_time
            log_performance(logger, "Match and merge data", duration, {
                'total_merged_companies': len(merged_companies),
                'matched_companies': processing_stats['matched_companies'],
                'unmatched_yc_companies': processing_stats['unmatched_yc_companies'],
                'new_linkedin_companies': processing_stats['new_linkedin_companies'],
                'processing_errors': len(processing_stats['processing_errors'])
            })
            
            # Log detailed merge statistics
            logger.info(f"✓ Merge completed successfully: {len(merged_companies)} total companies")
            logger.info(f"  - Matched companies (YC + LinkedIn): {processing_stats['matched_companies']}")
            logger.info(f"  - Unmatched YC companies: {processing_stats['unmatched_yc_companies']}")
            logger.info(f"  - New LinkedIn-only companies: {processing_stats['new_linkedin_companies']}")
            logger.info(f"  - Unmatched LinkedIn results: {len(match_result.unmatched_linkedin)}")
            
            # Log errors if any occurred
            if processing_stats['processing_errors']:
                logger.warning(f"Processing errors encountered: {len(processing_stats['processing_errors'])}")
                for error in processing_stats['processing_errors'][:5]:  # Log first 5 errors
                    logger.warning(f"  - {error}")
                if len(processing_stats['processing_errors']) > 5:
                    logger.warning(f"  ... and {len(processing_stats['processing_errors']) - 5} more errors")
            
            # Log data operation success
            log_data_operation(logger, "Match and merge companies", len(merged_companies), True)
            
            return merged_companies
            
        except Exception as e:
            duration = time.time() - start_time
            log_exception(logger, f"Critical error in match_and_merge_data after {duration:.2f}s")
            log_data_operation(logger, "Match and merge companies", 0, False, [str(e)])
            raise
    
    def _log_discovery_summary(self, results: Dict[str, Any], total_duration: float):
        """
        Log comprehensive summary statistics for the discovery process.
        
        Args:
            results: Discovery results dictionary
            total_duration: Total processing time in seconds
        """
        logger.info("=== DISCOVERY SUMMARY STATISTICS ===")
        logger.info(f"Batch: {results['batch']}")
        logger.info(f"Total processing time: {total_duration:.2f} seconds")
        logger.info(f"YC companies fetched: {results['yc_companies_count']}")
        logger.info(f"LinkedIn profiles found: {results['linkedin_results_count']}")
        logger.info(f"Total merged companies: {results['merged_companies_count']}")
        
        # Calculate efficiency metrics
        if total_duration > 0:
            companies_per_second = results['merged_companies_count'] / total_duration
            logger.info(f"Processing rate: {companies_per_second:.2f} companies/second")
        
        # Log data quality metrics
        if results['yc_companies_count'] > 0:
            coverage_rate = (results['merged_companies_count'] / results['yc_companies_count']) * 100
            logger.info(f"Data coverage: {coverage_rate:.1f}% of YC companies processed")
        
        logger.info("=== END DISCOVERY SUMMARY ===")
    
    def close(self):
        """Close any open connections."""
        if hasattr(self, 'linkedin_client'):
            self.linkedin_client.close()
        logger.debug("LinkedInDiscoveryService closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class CompanyMatcher:
    """
    Company matching service that uses fuzzy string matching to match LinkedIn
    profiles with YC company data.
    
    This class implements fuzzy string matching using difflib.SequenceMatcher
    to handle company name variations and provide confidence scoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the company matcher.
        
        Args:
            config: Configuration dictionary with matching settings
        """
        self.config = config or {}
        
        # Extract matching configuration
        matching_config = self.config.get('company_matching', {})
        
        # Confidence thresholds
        self.min_confidence_threshold = matching_config.get('min_confidence_threshold', 0.6)
        self.high_confidence_threshold = matching_config.get('high_confidence_threshold', 0.8)
        
        # Matching parameters
        self.case_sensitive = matching_config.get('case_sensitive', False)
        self.normalize_whitespace = matching_config.get('normalize_whitespace', True)
        self.remove_common_suffixes = matching_config.get('remove_common_suffixes', True)
        
        # Common company suffixes to handle
        self.common_suffixes = matching_config.get('common_suffixes', [
            'inc', 'inc.', 'incorporated',
            'llc', 'l.l.c.', 'limited liability company',
            'corp', 'corp.', 'corporation',
            'ltd', 'ltd.', 'limited',
            'co', 'co.', 'company',
            'pllc', 'p.l.l.c.',
            'lp', 'l.p.', 'limited partnership',
            'llp', 'l.l.p.', 'limited liability partnership'
        ])
        
        logger.info("CompanyMatcher initialized")
    
    def find_best_match(self, company_name: str, linkedin_results: List[LinkedInResult]) -> Optional[Tuple[LinkedInResult, float]]:
        """
        Find the best matching LinkedIn result for a given company name.
        
        Args:
            company_name: Company name to match against
            linkedin_results: List of LinkedIn results to search through
            
        Returns:
            Tuple of (best_match, confidence_score) if found, None otherwise
        """
        start_time = time.time()
        
        if not company_name or not linkedin_results:
            logger.debug(f"No match possible: company_name='{company_name}', linkedin_results_count={len(linkedin_results) if linkedin_results else 0}")
            return None
        
        logger.debug(f"Finding best match for company: '{company_name}' among {len(linkedin_results)} LinkedIn results")
        
        # Normalize the target company name
        normalized_target = self._normalize_company_name(company_name)
        logger.debug(f"Normalized target name: '{normalized_target}'")
        
        best_match = None
        best_score = 0.0
        scores_calculated = 0
        
        for linkedin_result in linkedin_results:
            # Calculate similarity score
            score = self.calculate_name_similarity(company_name, linkedin_result.company_name)
            scores_calculated += 1
            
            logger.debug(f"Similarity score for '{linkedin_result.company_name}': {score:.3f}")
            
            # Update best match if this score is higher
            if score > best_score and score >= self.min_confidence_threshold:
                best_match = linkedin_result
                best_score = score
                logger.debug(f"New best match: '{linkedin_result.company_name}' with score {score:.3f}")
        
        # Log performance and results
        duration = time.time() - start_time
        log_performance(logger, f"Find best match for '{company_name}'", duration, {
            'linkedin_results_checked': len(linkedin_results),
            'scores_calculated': scores_calculated,
            'best_score': best_score,
            'match_found': best_match is not None
        })
        
        if best_match:
            logger.debug(f"✓ Best match found: '{best_match.company_name}' with score {best_score:.3f}")
            return (best_match, best_score)
        else:
            logger.debug(f"✗ No match found above threshold {self.min_confidence_threshold} for '{company_name}'")
            return None
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity score between two company names using fuzzy matching.
        
        Args:
            name1: First company name
            name2: Second company name
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not name1 or not name2:
            return 0.0
        
        # Normalize both names
        norm_name1 = self._normalize_company_name(name1)
        norm_name2 = self._normalize_company_name(name2)
        
        # Handle exact matches after normalization
        if norm_name1 == norm_name2:
            return 1.0
        
        # Use difflib.SequenceMatcher for fuzzy matching
        matcher = difflib.SequenceMatcher(None, norm_name1, norm_name2)
        base_score = matcher.ratio()
        
        # Apply additional scoring bonuses
        final_score = self._apply_scoring_bonuses(name1, name2, norm_name1, norm_name2, base_score)
        
        # Ensure score is within valid range
        return max(0.0, min(1.0, final_score))
    
    def _normalize_company_name(self, name: str) -> str:
        """
        Normalize a company name for better matching.
        
        Args:
            name: Company name to normalize
            
        Returns:
            Normalized company name
        """
        if not name:
            return ""
        
        normalized = name.strip()
        
        # Convert to lowercase if not case sensitive
        if not self.case_sensitive:
            normalized = normalized.lower()
        
        # Normalize whitespace
        if self.normalize_whitespace:
            # Replace multiple whitespace with single space
            normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove common company suffixes
        if self.remove_common_suffixes:
            normalized = self._remove_company_suffixes(normalized)
        
        return normalized.strip()
    
    def _remove_company_suffixes(self, name: str) -> str:
        """
        Remove common company suffixes from a name.
        
        Args:
            name: Company name to process
            
        Returns:
            Name with suffixes removed
        """
        if not name:
            return ""
        
        # Create a copy to work with
        cleaned_name = name
        
        # Sort suffixes by length (longest first) to avoid partial matches
        sorted_suffixes = sorted(self.common_suffixes, key=len, reverse=True)
        
        for suffix in sorted_suffixes:
            # Create pattern to match suffix at end of string
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(suffix) + r'$'
            
            # Remove the suffix if found
            cleaned_name = re.sub(pattern, '', cleaned_name, flags=re.IGNORECASE)
            cleaned_name = cleaned_name.strip()
        
        # Return original name if cleaning resulted in empty string
        return cleaned_name if cleaned_name else name
    
    def _apply_scoring_bonuses(self, original1: str, original2: str, 
                              norm1: str, norm2: str, base_score: float) -> float:
        """
        Apply additional scoring bonuses based on specific matching criteria.
        
        Args:
            original1: Original first company name
            original2: Original second company name
            norm1: Normalized first company name
            norm2: Normalized second company name
            base_score: Base similarity score from SequenceMatcher
            
        Returns:
            Adjusted similarity score
        """
        score = base_score
        
        # Bonus for exact match after normalization
        if norm1 == norm2:
            score += 0.1
        
        # Bonus for one name being a substring of the other
        if norm1 in norm2 or norm2 in norm1:
            score += 0.05
        
        # Bonus for similar word count
        words1 = norm1.split()
        words2 = norm2.split()
        
        if len(words1) == len(words2):
            score += 0.02
        
        # Bonus for matching first word (often the main company name)
        if words1 and words2 and words1[0] == words2[0]:
            score += 0.05
        
        # Penalty for very different lengths
        len_diff = abs(len(norm1) - len(norm2))
        max_len = max(len(norm1), len(norm2))
        
        if max_len > 0:
            len_ratio = len_diff / max_len
            if len_ratio > 0.5:  # Names are very different in length
                score -= 0.1
        
        return score
    
    def match_companies(self, yc_companies: List[Dict[str, Any]], 
                       linkedin_results: List[LinkedInResult]) -> MatchResult:
        """
        Match YC companies with LinkedIn results and categorize the results.
        
        Args:
            yc_companies: List of YC company dictionaries
            linkedin_results: List of LinkedIn search results
            
        Returns:
            MatchResult containing categorized matching results
        """
        start_time = time.time()
        logger.info(f"=== Starting company matching process ===")
        logger.info(f"Input: {len(yc_companies)} YC companies, {len(linkedin_results)} LinkedIn results")
        
        # Initialize tracking variables
        matched_companies = []
        new_companies = []
        unmatched_yc = []
        unmatched_linkedin = linkedin_results.copy()  # Start with all LinkedIn results
        matched_linkedin_indices = set()
        
        # Processing statistics
        processing_stats = {
            'companies_processed': 0,
            'companies_skipped_existing_linkedin': 0,
            'companies_skipped_no_name': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'match_attempts': 0,
            'high_confidence_matches': 0,
            'medium_confidence_matches': 0,
            'low_confidence_matches': 0
        }
        
        # Process each YC company
        logger.info(f"Processing {len(yc_companies)} YC companies for matching...")
        
        for i, yc_company in enumerate(yc_companies):
            processing_stats['companies_processed'] += 1
            company_name = yc_company.get('name', '')
            
            # Log progress every 25 companies
            if i % 25 == 0 and i > 0:
                logger.info(f"Progress: {i}/{len(yc_companies)} companies processed ({(i/len(yc_companies)*100):.1f}%)")
            
            # Validate company name
            if not company_name:
                logger.warning(f"YC company at index {i} missing name field, skipping")
                processing_stats['companies_skipped_no_name'] += 1
                unmatched_yc.append(yc_company)
                continue
            
            # Skip if company already has LinkedIn URL
            existing_linkedin_url = yc_company.get('linkedin_url', '')
            if existing_linkedin_url and existing_linkedin_url.strip():
                logger.debug(f"Company '{company_name}' already has LinkedIn URL, keeping as-is")
                processing_stats['companies_skipped_existing_linkedin'] += 1
                matched_companies.append(yc_company)  # Keep as-is
                continue
            
            # Attempt to find best match for this YC company
            processing_stats['match_attempts'] += 1
            logger.debug(f"Attempting to match YC company: '{company_name}'")
            
            match_result = self.find_best_match(company_name, linkedin_results)
            
            if match_result:
                linkedin_match, confidence = match_result
                
                # Categorize match by confidence level
                if confidence >= self.high_confidence_threshold:
                    processing_stats['high_confidence_matches'] += 1
                    confidence_level = "HIGH"
                elif confidence >= (self.high_confidence_threshold + self.min_confidence_threshold) / 2:
                    processing_stats['medium_confidence_matches'] += 1
                    confidence_level = "MEDIUM"
                else:
                    processing_stats['low_confidence_matches'] += 1
                    confidence_level = "LOW"
                
                # Find the index of the matched LinkedIn result
                try:
                    linkedin_index = linkedin_results.index(linkedin_match)
                    matched_linkedin_indices.add(linkedin_index)
                    
                    # Create matched company with LinkedIn URL filled
                    matched_company = yc_company.copy()
                    matched_company['linkedin_url'] = linkedin_match.linkedin_url
                    matched_company['linkedin_match_confidence'] = confidence
                    
                    matched_companies.append(matched_company)
                    processing_stats['successful_matches'] += 1
                    
                    logger.debug(f"✓ MATCH: '{company_name}' → '{linkedin_match.company_name}' "
                               f"(confidence: {confidence:.3f}, level: {confidence_level})")
                    
                except ValueError:
                    # LinkedIn result not found in list (shouldn't happen)
                    logger.error(f"LinkedIn match not found in results list for {company_name}")
                    processing_stats['failed_matches'] += 1
                    unmatched_yc.append(yc_company)
            else:
                # No match found for this YC company
                processing_stats['failed_matches'] += 1
                unmatched_yc.append(yc_company)
                logger.debug(f"✗ NO MATCH: No LinkedIn match found for YC company: '{company_name}'")
        
        # Create new companies from unmatched LinkedIn results
        logger.info(f"Creating new companies from {len(linkedin_results) - len(matched_linkedin_indices)} unmatched LinkedIn results...")
        
        for i, linkedin_result in enumerate(linkedin_results):
            if i not in matched_linkedin_indices:
                # This LinkedIn result wasn't matched with any YC company
                new_company = {
                    'name': linkedin_result.company_name,
                    'linkedin_url': linkedin_result.linkedin_url,
                    'description': linkedin_result.description,
                    'yc_page': '',  # Empty since not from YC API
                    'batch': '',    # Will be filled by discovery service
                    'linkedin_only': True,  # Flag to indicate LinkedIn-only discovery
                    'linkedin_match_confidence': linkedin_result.confidence_score
                }
                new_companies.append(new_company)
                logger.debug(f"Created new LinkedIn-only company: '{linkedin_result.company_name}'")
        
        # Update unmatched LinkedIn results (remove matched ones)
        unmatched_linkedin = [
            linkedin_results[i] for i in range(len(linkedin_results))
            if i not in matched_linkedin_indices
        ]
        
        # Create final result
        result = MatchResult(
            matched_companies=matched_companies,
            new_companies=new_companies,
            unmatched_yc=unmatched_yc,
            unmatched_linkedin=unmatched_linkedin
        )
        
        # Log comprehensive matching results
        duration = time.time() - start_time
        log_performance(logger, "Company matching process", duration, {
            'yc_companies_input': len(yc_companies),
            'linkedin_results_input': len(linkedin_results),
            'successful_matches': processing_stats['successful_matches'],
            'failed_matches': processing_stats['failed_matches'],
            'new_companies_created': len(new_companies)
        })
        
        # Log detailed statistics
        logger.info(f"=== MATCHING RESULTS SUMMARY ===")
        logger.info(f"Processing time: {duration:.2f} seconds")
        logger.info(f"Companies processed: {processing_stats['companies_processed']}")
        logger.info(f"  - Skipped (existing LinkedIn): {processing_stats['companies_skipped_existing_linkedin']}")
        logger.info(f"  - Skipped (no name): {processing_stats['companies_skipped_no_name']}")
        logger.info(f"  - Match attempts: {processing_stats['match_attempts']}")
        logger.info(f"Matching results:")
        logger.info(f"  - Successful matches: {processing_stats['successful_matches']}")
        logger.info(f"    * High confidence: {processing_stats['high_confidence_matches']}")
        logger.info(f"    * Medium confidence: {processing_stats['medium_confidence_matches']}")
        logger.info(f"    * Low confidence: {processing_stats['low_confidence_matches']}")
        logger.info(f"  - Failed matches: {processing_stats['failed_matches']}")
        logger.info(f"Final counts:")
        logger.info(f"  - Matched companies: {len(matched_companies)}")
        logger.info(f"  - New LinkedIn companies: {len(new_companies)}")
        logger.info(f"  - Unmatched YC companies: {len(unmatched_yc)}")
        logger.info(f"  - Unmatched LinkedIn results: {len(unmatched_linkedin)}")
        
        # Calculate and log match rates
        if processing_stats['match_attempts'] > 0:
            match_rate = (processing_stats['successful_matches'] / processing_stats['match_attempts']) * 100
            logger.info(f"Overall match rate: {match_rate:.1f}%")
        
        logger.info(f"=== END MATCHING SUMMARY ===")
        
        # Log data operation success
        log_data_operation(logger, "Company matching", len(matched_companies) + len(new_companies), True)
        
        return result
    
    def get_match_statistics(self, match_result: MatchResult) -> Dict[str, Any]:
        """
        Generate statistics about matching results.
        
        Args:
            match_result: MatchResult to analyze
            
        Returns:
            Dictionary containing matching statistics
        """
        logger.debug("Generating match statistics")
        
        total_yc = len(match_result.matched_companies) + len(match_result.unmatched_yc)
        total_linkedin = len(match_result.matched_companies) + len(match_result.unmatched_linkedin)
        
        stats = {
            'total_yc_companies': total_yc,
            'total_linkedin_results': total_linkedin,
            'matched_companies': len(match_result.matched_companies),
            'new_linkedin_companies': len(match_result.new_companies),
            'unmatched_yc_companies': len(match_result.unmatched_yc),
            'unmatched_linkedin_results': len(match_result.unmatched_linkedin),
            'yc_match_rate': len(match_result.matched_companies) / total_yc if total_yc > 0 else 0.0,
            'linkedin_match_rate': len(match_result.matched_companies) / total_linkedin if total_linkedin > 0 else 0.0
        }
        
        # Calculate confidence score statistics for matched companies
        if match_result.matched_companies:
            confidence_scores = [
                company.get('linkedin_match_confidence', 0.0)
                for company in match_result.matched_companies
                if 'linkedin_match_confidence' in company
            ]
            
            if confidence_scores:
                stats['avg_confidence_score'] = sum(confidence_scores) / len(confidence_scores)
                stats['min_confidence_score'] = min(confidence_scores)
                stats['max_confidence_score'] = max(confidence_scores)
                stats['high_confidence_matches'] = sum(1 for score in confidence_scores if score >= self.high_confidence_threshold)
                
                logger.debug(f"Confidence score statistics: avg={stats['avg_confidence_score']:.3f}, "
                           f"min={stats['min_confidence_score']:.3f}, max={stats['max_confidence_score']:.3f}, "
                           f"high_confidence_count={stats['high_confidence_matches']}")
        
        logger.debug(f"Match statistics generated: {stats}")
        return stats