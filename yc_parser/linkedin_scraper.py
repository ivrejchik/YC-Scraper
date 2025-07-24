"""
LinkedIn Scraper Module

Checks LinkedIn company pages for "YC S25" mentions.
"""
import time
import random
import requests
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class LinkedInScraper:
    """
    Scraper for LinkedIn company pages to check for YC S25 mentions.
    
    Implements rate limiting, retry logic, and proper HTTP headers
    to respectfully access LinkedIn pages.
    """
    
    def __init__(self, delay_range: tuple = (1, 2), max_retries: int = 3):
        """
        Initialize the LinkedIn scraper.
        
        Args:
            delay_range: Tuple of (min, max) seconds to wait between requests
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set up proper headers to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _make_request(self, url: str) -> Optional[str]:
        """
        Make HTTP request to LinkedIn URL with comprehensive retry logic and rate limiting.
        
        Args:
            url: LinkedIn company page URL
            
        Returns:
            HTML content as string, or None if request failed
        """
        if not url or not url.strip():
            logger.warning("Empty or invalid URL provided")
            return None
            
        # Validate and normalize LinkedIn URL
        try:
            if not url.startswith('http'):
                url = 'https://' + url
            
            # Basic URL validation
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.netloc:
                logger.warning(f"Invalid URL format: {url}")
                return None
                
        except Exception as e:
            logger.warning(f"URL validation failed for {url}: {e}")
            return None
            
        last_error = None
        retry_delay = 2
        
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting with jitter
                base_delay = random.uniform(*self.delay_range)
                jitter = random.uniform(0.1, 0.5)  # Add small random jitter
                delay = base_delay + jitter
                time.sleep(delay)
                
                logger.debug(f"Making request to {url} (attempt {attempt + 1}/{self.max_retries})")
                
                # Make request with timeout
                response = self.session.get(url, timeout=15)
                
                # Handle different response codes
                if response.status_code == 200:
                    # Validate response content
                    if not response.content:
                        logger.warning(f"Empty response content from {url}")
                        last_error = "Empty response content"
                        if attempt < self.max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 1.5
                            continue
                        return None
                    
                    # Check if we got a valid HTML response
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type and 'text/plain' not in content_type:
                        logger.warning(f"Unexpected content type from {url}: {content_type}")
                        last_error = f"Unexpected content type: {content_type}"
                        if attempt < self.max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        return None
                    
                    logger.debug(f"Successfully retrieved content from {url} ({len(response.content)} bytes)")
                    return response.text
                
                elif response.status_code == 429:
                    # Rate limiting - respect Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                        except ValueError:
                            wait_time = 60  # Default fallback
                    else:
                        wait_time = min(60, retry_delay * (attempt + 1))  # Exponential backoff with cap
                    
                    logger.warning(f"Rate limited by {url}. Waiting {wait_time} seconds before retry")
                    time.sleep(wait_time)
                    last_error = f"Rate limited (429), waited {wait_time}s"
                    continue
                
                elif response.status_code in [403, 401]:
                    logger.warning(f"Access denied for {url} (status: {response.status_code})")
                    last_error = f"Access denied ({response.status_code})"
                    # Don't retry access denied errors
                    return None
                
                elif response.status_code == 404:
                    logger.warning(f"Page not found for {url} (status: 404)")
                    last_error = "Page not found (404)"
                    # Don't retry 404 errors
                    return None
                
                elif response.status_code in [500, 502, 503, 504]:
                    # Server errors - retry with backoff
                    logger.warning(f"Server error {response.status_code} for {url} (attempt {attempt + 1})")
                    last_error = f"Server error ({response.status_code})"
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                
                elif response.status_code == 302 or response.status_code == 301:
                    # Redirect - let requests handle it, but log it
                    logger.debug(f"Redirect {response.status_code} for {url}")
                    last_error = f"Redirect ({response.status_code})"
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                
                else:
                    logger.warning(f"Unexpected HTTP {response.status_code} for {url}")
                    last_error = f"HTTP {response.status_code}"
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1}): {e}")
                last_error = f"Timeout: {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error for {url} (attempt {attempt + 1}): {e}")
                last_error = f"Connection error: {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                
            except requests.exceptions.TooManyRedirects as e:
                logger.warning(f"Too many redirects for {url}: {e}")
                last_error = f"Too many redirects: {e}"
                # Don't retry redirect loops
                return None
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {url} (attempt {attempt + 1}): {e}")
                last_error = f"Request error: {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
            
            except Exception as e:
                logger.error(f"Unexpected error for {url} (attempt {attempt + 1}): {e}")
                last_error = f"Unexpected error: {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
        
        # All attempts failed
        logger.error(f"Failed to retrieve content from {url} after {self.max_retries} attempts. Last error: {last_error}")
        return None
    
    def check_yc_mention(self, linkedin_url: str) -> bool:
        """
        Check if a LinkedIn company page mentions "YC S25".
        
        Handles various HTML structures and text encodings by:
        - Searching for multiple variations of the YC S25 phrase
        - Handling HTML entities and special characters
        - Case-insensitive matching
        
        Args:
            linkedin_url: LinkedIn company page URL
            
        Returns:
            True if "YC S25" is found, False otherwise
        """
        if not linkedin_url or not linkedin_url.strip():
            logger.debug("No LinkedIn URL provided")
            return False
            
        html_content = self._make_request(linkedin_url)
        
        if html_content is None:
            logger.debug(f"Could not retrieve content from {linkedin_url}")
            return False
        
        # Convert to lowercase for case-insensitive search
        content_lower = html_content.lower()
        
        # Define various patterns to search for YC S25 mentions
        yc_patterns = [
            "yc s25",           # Standard format
            "yc&nbsp;s25",      # With HTML non-breaking space
            "yc s25",           # With regular space
            "yc  s25",          # With multiple spaces
            "yc\ns25",          # With newline
            "yc\ts25",          # With tab
            "y combinator s25", # Full name
            "y&nbsp;combinator&nbsp;s25",  # Full name with HTML spaces
            "ycombinator s25",  # No space in YC
            "yc summer 2025",   # Alternative format
            "yc summer '25",    # Alternative format with quote
            "yc summer 25",     # Alternative format without quote
            "yc batch s25",     # With batch mention
            "yc class s25",     # With class mention
        ]
        
        # Search for any of the patterns
        for pattern in yc_patterns:
            if pattern in content_lower:
                logger.info(f"Found YC S25 mention (pattern: '{pattern}') on {linkedin_url}")
                return True
        
        # Additional check for HTML-encoded content
        import html
        try:
            decoded_content = html.unescape(content_lower)
            for pattern in yc_patterns[:5]:  # Check main patterns on decoded content
                if pattern in decoded_content:
                    logger.info(f"Found YC S25 mention in decoded HTML (pattern: '{pattern}') on {linkedin_url}")
                    return True
        except Exception as e:
            logger.debug(f"Could not decode HTML entities: {e}")
        
        logger.debug(f"No YC S25 mention found on {linkedin_url}")
        return False
    
    def batch_check_mentions(self, companies: List[Dict[str, Any]], max_workers: int = 3, 
                           progress_callback=None) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Check multiple companies for YC S25 mentions using parallel processing.
        
        Features:
        - Parallel execution with configurable worker count
        - Progress tracking with optional callback
        - Individual error handling without stopping batch
        - Comprehensive error collection and reporting
        
        Args:
            companies: List of company dictionaries with 'linkedin_url' field
            max_workers: Maximum number of concurrent workers
            progress_callback: Optional function called with (completed, total) for progress tracking
            
        Returns:
            Tuple of (processed_companies, error_messages)
            - processed_companies: List of company dictionaries with 'yc_s25_on_linkedin' field added
            - error_messages: List of error messages encountered during processing
        """
        if not companies:
            logger.info("No companies to process")
            return [], []
            
        results = []
        errors = []
        completed_count = 0
        total_count = len(companies)
        
        def process_company(company: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single company and add LinkedIn flag."""
            nonlocal completed_count
            
            try:
                company_name = company.get('name', 'Unknown')
                linkedin_url = company.get('linkedin_url', '')
                
                logger.debug(f"Processing company: {company_name}")
                
                # Check for YC mention
                yc_mention = self.check_yc_mention(linkedin_url)
                
                # Create a copy of the company data with the new field
                updated_company = company.copy()
                updated_company['yc_s25_on_linkedin'] = yc_mention
                
                # Update progress
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_count)
                    except Exception as callback_error:
                        logger.warning(f"Progress callback error: {callback_error}")
                
                logger.debug(f"Completed processing {company_name}: YC mention = {yc_mention}")
                return updated_company
                
            except Exception as e:
                error_msg = f"Error processing company {company.get('name', 'Unknown')}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                # Update progress even on error
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_count)
                    except Exception as callback_error:
                        logger.warning(f"Progress callback error: {callback_error}")
                
                # Return company with False flag on error
                updated_company = company.copy()
                updated_company['yc_s25_on_linkedin'] = False
                return updated_company
        
        logger.info(f"Starting batch processing of {total_count} companies with {max_workers} workers")
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_company = {
                    executor.submit(process_company, company): company 
                    for company in companies
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_company):
                    try:
                        result = future.result()
                        results.append(result)
                        
                    except Exception as e:
                        company = future_to_company[future]
                        company_name = company.get('name', 'Unknown')
                        error_msg = f"Unexpected error processing {company_name}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        
                        # Add company with False flag
                        updated_company = company.copy()
                        updated_company['yc_s25_on_linkedin'] = False
                        results.append(updated_company)
                        
        except Exception as e:
            error_msg = f"Critical error during batch processing: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            # If we have a critical failure, ensure all companies are included with False flags
            processed_names = {result.get('name') for result in results}
            for company in companies:
                if company.get('name') not in processed_names:
                    updated_company = company.copy()
                    updated_company['yc_s25_on_linkedin'] = False
                    results.append(updated_company)
        
        # Log summary
        success_count = len([r for r in results if r.get('yc_s25_on_linkedin', False)])
        error_count = len(errors)
        
        logger.info(f"Batch processing completed:")
        logger.info(f"  - Total companies: {total_count}")
        logger.info(f"  - Successfully processed: {len(results)}")
        logger.info(f"  - YC S25 mentions found: {success_count}")
        logger.info(f"  - Errors encountered: {error_count}")
        
        if errors:
            logger.warning(f"Error summary: {error_count} errors during processing")
            for error in errors[:5]:  # Log first 5 errors
                logger.warning(f"  - {error}")
            if len(errors) > 5:
                logger.warning(f"  - ... and {len(errors) - 5} more errors")
        
        return results, errors
    
    def close(self):
        """Close the session and clean up resources."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()