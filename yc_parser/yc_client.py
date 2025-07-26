"""
YC Data Client Module

Handles communication with YC's unofficial API and individual company page parsing.
"""
import requests
import time
import logging
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class YcClient:
    """Client for fetching YC company data from API and profile pages."""
    
    def __init__(self, season: str = "Summer", year: int = 2025):
        self.base_url = "https://www.ycombinator.com"
        self.api_base = "https://yc-oss.github.io/api/batches"
        self.season = season
        self.year = year
        self.session = self._create_session()
    
    def _build_api_url(self, season: str = None, year: int = None) -> str:
        """
        Build API URL dynamically based on season and year.
        
        Args:
            season: "Summer" or "Winter" (defaults to instance season)
            year: Year (defaults to instance year)
            
        Returns:
            API URL for the specified batch
            
        Examples:
            Summer 2025 -> https://yc-oss.github.io/api/batches/summer-2025.json
            Winter 2024 -> https://yc-oss.github.io/api/batches/winter-2024.json
        """
        season = season or self.season
        year = year or self.year
        
        season_name = season.lower()  # "summer" or "winter"
        return f"{self.api_base}/{season_name}-{year}.json"
    
    def set_batch(self, season: str, year: int):
        """
        Set the batch season and year for API requests.
        
        Args:
            season: "Summer" or "Winter"
            year: Year (e.g., 2025, 2024, etc.)
        """
        self.season = season
        self.year = year
        logger.info(f"Batch set to {season} {year}")
    
    def get_current_batch_code(self) -> str:
        """Get the current batch code (e.g., S25, W24)."""
        season_code = "S" if self.season == "Summer" else "W"
        year_code = str(self.year)[2:]  # Last 2 digits
        return f"{season_code}{year_code}"
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        return session
    
    def fetch_yc_batch_data(self, season: str = None, year: int = None) -> List[Dict]:
        """
        Fetch YC companies from the official API endpoint with comprehensive validation.
        
        This method dynamically fetches data for any YC batch:
        - Builds API URL based on season and year
        - Validates JSON response structure integrity
        - Extracts company information including name, website, YC profile URL, and metadata
        - Implements comprehensive error handling for network failures and malformed responses
        - Validates that all required fields are present
        
        Args:
            season: "Summer" or "Winter" (defaults to instance season)
            year: Year (defaults to instance year)
        
        Returns:
            List of validated company dictionaries with required fields
            
        Raises:
            requests.RequestException: If API request fails after all retries
            ValueError: If response format is invalid or validation fails
        """
        api_url = self._build_api_url(season, year)
        batch_code = f"{'S' if (season or self.season) == 'Summer' else 'W'}{str(year or self.year)[2:]}"
        
        logger.info(f"Fetching YC {batch_code} batch data from official API: {api_url}")
        
        try:
            # Make request to official YC API endpoint
            response = self.session.get(api_url, timeout=10)
            response.raise_for_status()
            
            # Validate response is not empty
            if not response.content:
                raise ValueError("Empty response from YC S25 API")
            
            # Parse and validate JSON structure
            try:
                companies_data = response.json()
            except ValueError as json_error:
                logger.error(f"Invalid JSON response from YC S25 API: {json_error}")
                raise ValueError(f"Malformed JSON response: {json_error}")
            
            # Validate response structure integrity
            if not isinstance(companies_data, list):
                raise ValueError(f"Expected list response, got {type(companies_data)}")
            
            if not companies_data:
                raise ValueError("API returned empty companies list")
            
            logger.info(f"Successfully fetched {len(companies_data)} companies from YC S25 API")
            
            # Validate and extract company information
            validated_companies = self._validate_batch_company_data(companies_data, season, year)
            
            if not validated_companies:
                raise ValueError("No valid companies found after validation")
            
            logger.info(f"Successfully validated {len(validated_companies)} companies")
            return validated_companies
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout fetching YC S25 data: {e}")
            raise requests.RequestException(f"API request timeout: {e}")
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching YC S25 data: {e}")
            raise requests.RequestException(f"API connection failed: {e}")
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching YC S25 data: {e.response.status_code}")
            raise requests.RequestException(f"API HTTP error {e.response.status_code}: {e}")
            
        except requests.RequestException as e:
            logger.error(f"Request error fetching YC S25 data: {e}")
            raise
            
        except ValueError as e:
            logger.error(f"Data validation error: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error fetching YC S25 data: {e}")
            raise ValueError(f"Unexpected error: {e}")
    
    def _validate_batch_company_data(self, companies_data: List[Dict], season: str = None, year: int = None) -> List[Dict]:
        """
        Validate YC company data structure and ensure all required fields are present.
        
        Based on the actual YC API format (like airbnb.json example):
        - name: company name
        - website: company website URL
        - one_liner: short description
        - long_description: detailed description
        - slug: URL slug for YC profile
        - batch: YC batch (varies by season/year)
        - team_size: number of employees
        - all_locations: company locations
        
        Args:
            companies_data: Raw company data from API
            season: Season for batch validation (defaults to instance season)
            year: Year for batch validation (defaults to instance year)
            
        Returns:
            List of validated company dictionaries
            
        Raises:
            ValueError: If critical validation fails
        """
        validated_companies = []
        validation_errors = []
        
        # Get expected batch formats
        season = season or self.season
        year = year or self.year
        batch_code = f"{'S' if season == 'Summer' else 'W'}{str(year)[2:]}"
        expected_batch_formats = [
            batch_code,  # e.g., "S25", "W24"
            f"{season} {year}",  # e.g., "Summer 2025", "Winter 2024"
        ]
        
        required_fields = ['name']  # Minimum required field
        
        for i, company in enumerate(companies_data):
            try:
                # Validate company is a dictionary
                if not isinstance(company, dict):
                    validation_errors.append(f"Company {i}: Expected dict, got {type(company)}")
                    continue
                
                # Check for required fields
                missing_required = [field for field in required_fields if not company.get(field)]
                if missing_required:
                    validation_errors.append(f"Company {i}: Missing required fields: {missing_required}")
                    continue
                
                # Extract and validate company information using actual YC API format
                validated_company = {
                    'name': str(company.get('name', '')).strip(),
                    'website': str(company.get('website', '')).strip(),
                    'description': str(company.get('one_liner', company.get('long_description', ''))).strip(),
                    'yc_page': self._build_yc_url_from_slug(company.get('slug', '')),
                    'linkedin_url': '',  # Not provided in YC API, will be enriched later
                    'batch': str(company.get('batch', batch_code)).strip(),
                    'metadata': {
                        'id': company.get('id'),
                        'slug': str(company.get('slug', '')).strip(),
                        'one_liner': str(company.get('one_liner', '')).strip(),
                        'long_description': str(company.get('long_description', '')).strip(),
                        'team_size': company.get('team_size'),
                        'all_locations': str(company.get('all_locations', '')).strip(),
                        'industry': str(company.get('industry', '')).strip(),
                        'subindustry': str(company.get('subindustry', '')).strip(),
                        'launched_at': company.get('launched_at'),
                        'tags': company.get('tags', []),
                        'status': str(company.get('status', '')).strip(),
                        'stage': str(company.get('stage', '')).strip(),
                        'top_company': company.get('top_company', False),
                        'small_logo_thumb_url': str(company.get('small_logo_thumb_url', '')).strip(),
                    }
                }
                
                # Validate name is not empty after stripping
                if not validated_company['name']:
                    validation_errors.append(f"Company {i}: Name is empty after validation")
                    continue
                
                # Normalize batch to standard format (e.g., S25, W24)
                validated_company['batch'] = batch_code
                
                # Clean metadata - remove empty values
                validated_company['metadata'] = {
                    k: v for k, v in validated_company['metadata'].items() 
                    if v is not None and (str(v).strip() if isinstance(v, str) else True)
                }
                
                validated_companies.append(validated_company)
                
            except Exception as e:
                validation_errors.append(f"Company {i}: Validation error: {e}")
                continue
        
        # Log validation summary
        if validation_errors:
            logger.warning(f"Validation errors encountered: {len(validation_errors)} companies failed validation")
            for error in validation_errors[:5]:  # Log first 5 errors
                logger.warning(f"Validation error: {error}")
            if len(validation_errors) > 5:
                logger.warning(f"... and {len(validation_errors) - 5} more validation errors")
        
        # Ensure we have some valid companies
        if not validated_companies:
            raise ValueError(f"All companies failed validation. Errors: {validation_errors[:3]}")
        
        success_rate = len(validated_companies) / len(companies_data) * 100
        logger.info(f"Validation complete: {len(validated_companies)}/{len(companies_data)} companies valid ({success_rate:.1f}%)")
        
        return validated_companies
    
    def _build_yc_url_from_slug(self, slug: str) -> str:
        """Build YC profile URL from company slug."""
        if not slug:
            return ''
        
        return f"{self.base_url}/companies/{slug}"

    def fetch_s25_companies(self) -> List[Dict]:
        """
        Fetch all S25 companies from YC API endpoint (DEPRECATED).
        
        This method is deprecated. Use fetch_yc_batch_data() instead for any batch.
        
        Returns:
            List of company dictionaries with basic information
        """
        logger.warning("fetch_s25_companies() is deprecated. Use fetch_yc_batch_data() instead.")
        return self.fetch_yc_batch_data("Summer", 2025)

    
    def _normalize_api_data(self, companies: List[Dict]) -> List[Dict]:
        """
        Normalize API response data to consistent format.
        
        Args:
            companies: Raw company data from API
            
        Returns:
            Normalized company data
        """
        normalized = []
        
        for company in companies:
            try:
                # Handle different possible API response formats
                normalized_company = {
                    'name': company.get('name', '').strip(),
                    'website': company.get('website', '').strip(),
                    'description': company.get('description', company.get('tagline', '')).strip(),
                    'yc_page': self._build_yc_url(company.get('slug', company.get('name', ''))),
                    'linkedin_url': company.get('linkedin_url', '').strip(),
                    'batch': company.get('batch', 'S25')
                }
                
                # Only include companies from S25 batch
                if normalized_company['batch'] == 'S25' and normalized_company['name']:
                    normalized.append(normalized_company)
                    
            except Exception as e:
                logger.warning(f"Failed to normalize company data: {company}, error: {e}")
                continue
        
        return normalized
    
    def _build_yc_url(self, slug: str) -> str:
        """Build YC profile URL from company slug or name."""
        if not slug:
            return ''
        
        # Clean slug for URL
        clean_slug = slug.lower().replace(' ', '-').replace('_', '-')
        return f"{self.base_url}/companies/{clean_slug}"
    
    def _fallback_scrape_companies(self) -> List[Dict]:
        """
        Fallback method to scrape companies page if API fails.
        
        Returns:
            List of company dictionaries
        """
        try:
            logger.info("Attempting fallback scraping of companies page")
            
            # Try to scrape the main companies page filtered for S25
            companies_url = f"{self.api_base}?batch=S25"
            response = self.session.get(companies_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # This is a simplified fallback - in practice, you'd need to 
            # analyze the actual HTML structure of the YC companies page
            companies = []
            
            # Look for company cards or similar elements
            company_elements = soup.find_all('div', class_=['company-card', 'company-item'])
            
            for element in company_elements:
                try:
                    name_elem = element.find(['h3', 'h4', 'a'])
                    name = name_elem.get_text().strip() if name_elem else ''
                    
                    link_elem = element.find('a', href=True)
                    yc_page = f"{self.base_url}{link_elem['href']}" if link_elem else ''
                    
                    if name:
                        companies.append({
                            'name': name,
                            'website': '',
                            'description': '',
                            'yc_page': yc_page,
                            'linkedin_url': '',
                            'batch': 'S25'
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to parse company element: {e}")
                    continue
            
            logger.info(f"Fallback scraping found {len(companies)} companies")
            return companies
            
        except Exception as e:
            logger.error(f"Fallback scraping failed: {e}")
            return []
    
    def parse_company_profile(self, yc_url: str) -> Dict:
        """
        Parse individual YC company profile page to extract additional details.
        
        Args:
            yc_url: URL to YC company profile page
            
        Returns:
            Dictionary with extracted company details
            
        Raises:
            requests.RequestException: If page request fails
        """
        if not yc_url:
            return {}
        
        try:
            logger.debug(f"Parsing YC profile: {yc_url}")
            
            response = self.session.get(yc_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract company details from profile page
            profile_data = {
                'tagline': self._extract_tagline(soup),
                'website': self._extract_website(soup),
                'linkedin_url': self._extract_linkedin_url(soup),
                'description': self._extract_description(soup)
            }
            
            # Remove empty values
            profile_data = {k: v for k, v in profile_data.items() if v}
            
            logger.debug(f"Extracted profile data: {profile_data}")
            return profile_data
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch YC profile {yc_url}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Failed to parse YC profile {yc_url}: {e}")
            return {}
    
    def _extract_tagline(self, soup: BeautifulSoup) -> str:
        """Extract company tagline from YC profile page."""
        # Try multiple selectors for tagline
        selectors = [
            'h2.tagline',
            '.tagline',
            'h2[class*="tagline"]',
            '.company-tagline',
            'p.tagline'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        # Fallback: look for text patterns
        # Sometimes tagline appears in meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()
        
        return ''
    
    def _extract_website(self, soup: BeautifulSoup) -> str:
        """Extract company website URL from YC profile page."""
        # Try multiple selectors for website link
        selectors = [
            'a.website-link',
            '.company-website a',
            'a[class*="website"]'
        ]
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element and element.get('href'):
                    href = element['href']
                    # Skip YC internal links
                    if not href.startswith('http://www.ycombinator.com') and not href.startswith('https://www.ycombinator.com'):
                        return href.strip()
            except:
                continue
        
        # Look for links with "Website" text
        website_links = soup.find_all('a', string=lambda text: text and 'website' in text.lower())
        for link in website_links:
            if link.get('href'):
                href = link['href']
                if not href.startswith('http://www.ycombinator.com') and not href.startswith('https://www.ycombinator.com'):
                    return href.strip()
        
        # Fallback: look for any external links that might be the website
        external_links = soup.find_all('a', href=True)
        for link in external_links:
            href = link['href']
            if (href.startswith('http') and 
                'ycombinator.com' not in href and 
                'linkedin.com' not in href and
                'twitter.com' not in href and
                'facebook.com' not in href):
                # This might be the company website
                return href.strip()
        
        return ''
    
    def _extract_linkedin_url(self, soup: BeautifulSoup) -> str:
        """Extract LinkedIn company URL from YC profile page."""
        # Look for LinkedIn links
        linkedin_links = soup.find_all('a', href=True)
        
        for link in linkedin_links:
            href = link['href']
            if 'linkedin.com/company/' in href:
                return href.strip()
        
        return ''
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract company description from YC profile page."""
        # Try multiple selectors for description
        selectors = [
            '.company-description',
            '.description',
            'p.description',
            '.company-summary',
            '.about'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text().strip()
                if len(text) > 20:  # Ensure it's substantial content
                    return text
        
        # Fallback: look for paragraphs that might contain description
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text().strip()
            # Look for substantial paragraphs that aren't navigation or footer text
            if (len(text) > 50 and 
                'ycombinator' not in text.lower() and
                'sign up' not in text.lower() and
                'login' not in text.lower()):
                return text
        
        return ''
    
    def enrich_company_data(self, companies: List[Dict]) -> List[Dict]:
        """
        Enrich company data by parsing individual YC profile pages.
        
        Args:
            companies: List of company dictionaries from API
            
        Returns:
            List of enriched company dictionaries
        """
        enriched_companies = []
        
        for i, company in enumerate(companies):
            try:
                logger.info(f"Enriching company {i+1}/{len(companies)}: {company.get('name', 'Unknown')}")
                
                # Start with existing data
                enriched_company = company.copy()
                
                # Parse profile page if we have a YC URL
                if company.get('yc_page'):
                    profile_data = self.parse_company_profile(company['yc_page'])
                    
                    # Fill in missing fields with profile data
                    for field in ['tagline', 'website', 'linkedin_url', 'description']:
                        if not enriched_company.get(field) and profile_data.get(field):
                            enriched_company[field] = profile_data[field]
                
                # Ensure all required fields exist with defaults
                enriched_company.setdefault('website', '')
                enriched_company.setdefault('description', '')
                enriched_company.setdefault('linkedin_url', '')
                
                enriched_companies.append(enriched_company)
                
                # Add small delay to be respectful
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to enrich company {company.get('name', 'Unknown')}: {e}")
                # Add company with existing data even if enrichment fails
                enriched_companies.append(company)
                continue
        
        logger.info(f"Successfully enriched {len(enriched_companies)} companies")
        return enriched_companies
    
    def get_complete_batch_data(self, season: str = None, year: int = None) -> List[Dict]:
        """
        Get complete YC batch data from API (fast version without profile enrichment).
        
        This method provides fast access to YC company data by:
        1. Fetching companies from API with comprehensive validation
        2. Skipping slow profile page enrichment for better performance
        3. Returning clean, validated data ready for use
        
        Args:
            season: "Summer" or "Winter" (defaults to instance season)
            year: Year (defaults to instance year)
        
        Returns:
            List of complete company dictionaries
            
        Raises:
            Exception: Only if API fetch fails completely
        """
        season = season or self.season
        year = year or self.year
        batch_code = f"{'S' if season == 'Summer' else 'W'}{str(year)[2:]}"
        
        try:
            logger.info(f"Starting fast {batch_code} data collection")
            
            # Fetch companies from API with validation (fast)
            companies = self.fetch_yc_batch_data(season, year)
            
            if not companies:
                logger.warning(f"No companies found from {batch_code} API")
                return []
            
            logger.info(f"Successfully fetched {len(companies)} companies from {batch_code} API")
            
            # The fetch_yc_batch_data method already provides clean, validated data
            # No need for additional enrichment or cleaning for fast access
            return companies
            
        except Exception as e:
            logger.error(f"Failed to collect {batch_code} data: {e}")
            raise
    
    def get_complete_batch_data_with_enrichment(self, season: str = None, year: int = None) -> List[Dict]:
        """
        Get complete YC batch data with profile page enrichment (slow but comprehensive).
        
        This method provides comprehensive data by:
        1. Fetching companies from API
        2. Enriching with profile page data (slow)
        3. Final validation and cleanup
        
        Use this method when you need the most complete data and can wait for enrichment.
        
        Args:
            season: "Summer" or "Winter" (defaults to instance season)
            year: Year (defaults to instance year)
        
        Returns:
            List of enriched company dictionaries
            
        Raises:
            Exception: Only if both API and fallback methods fail completely
        """
        season = season or self.season
        year = year or self.year
        batch_code = f"{'S' if season == 'Summer' else 'W'}{str(year)[2:]}"
        
        try:
            logger.info(f"Starting comprehensive {batch_code} data collection with enrichment")
            
            # Step 1: Fetch companies from API
            companies = self.fetch_yc_batch_data(season, year)
            
            if not companies:
                logger.warning(f"No companies found from {batch_code} API")
                return []
            
            logger.info(f"Found {len(companies)} companies from {batch_code} API")
            
            # Step 2: Enrich with profile page data (slow)
            enriched_companies = self.enrich_company_data(companies)
            
            # Step 3: Final validation and cleanup
            complete_companies = self._validate_and_clean_data(enriched_companies)
            
            logger.info(f"Successfully collected enriched data for {len(complete_companies)} companies")
            return complete_companies
            
        except Exception as e:
            logger.error(f"Failed to collect enriched {batch_code} data: {e}")
            raise
    
    def _validate_and_clean_data(self, companies: List[Dict]) -> List[Dict]:
        """
        Validate and clean company data before returning.
        
        Args:
            companies: List of company dictionaries
            
        Returns:
            List of validated and cleaned company dictionaries
        """
        cleaned_companies = []
        
        for company in companies:
            try:
                # Ensure required fields exist
                if not company.get('name'):
                    logger.warning(f"Skipping company with no name: {company}")
                    continue
                
                # Clean and validate data
                cleaned_company = {
                    'name': company.get('name', '').strip(),
                    'website': self._clean_url(company.get('website', '')),
                    'description': company.get('description', '').strip(),
                    'yc_page': self._clean_url(company.get('yc_page', '')),
                    'linkedin_url': self._clean_url(company.get('linkedin_url', '')),
                    'batch': company.get('batch', 'S25')
                }
                
                # Ensure description is not empty - use tagline as fallback
                if not cleaned_company['description'] and company.get('tagline'):
                    cleaned_company['description'] = company['tagline'].strip()
                
                cleaned_companies.append(cleaned_company)
                
            except Exception as e:
                logger.warning(f"Failed to clean company data {company.get('name', 'Unknown')}: {e}")
                continue
        
        return cleaned_companies
    
    def _clean_url(self, url: str) -> str:
        """Clean and validate URL."""
        if not url:
            return ''
        
        url = url.strip()
        
        # Add protocol if missing
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        return url