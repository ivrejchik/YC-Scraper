"""
Data models and validation functions for the YC S25 Company Parser.

This module contains the core data structures used throughout the application,
including Company and ProcessingResult dataclasses with serialization methods
and data validation functions.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import re
from urllib.parse import urlparse


@dataclass
class Company:
    """
    Represents a Y Combinator company with all relevant information.
    
    Attributes:
        name: Company name
        website: Company website URL
        description: One-line company description
        yc_page: YC profile URL
        linkedin_url: LinkedIn company page URL
        yc_s25_on_linkedin: Whether "YC S25" appears on LinkedIn
        last_updated: ISO timestamp of last check
    """
    name: str
    website: str
    description: str
    yc_page: str
    linkedin_url: str
    yc_s25_on_linkedin: bool
    last_updated: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Company instance to dictionary for CSV storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Company':
        """Create Company instance from dictionary."""
        return cls(
            name=str(data.get('name', '')),
            website=str(data.get('website', '')),
            description=str(data.get('description', '')),
            yc_page=str(data.get('yc_page', '')),
            linkedin_url=str(data.get('linkedin_url', '')),
            yc_s25_on_linkedin=bool(data.get('yc_s25_on_linkedin', False)),
            last_updated=str(data.get('last_updated', datetime.now().isoformat()))
        )
    
    def __post_init__(self):
        """Validate data after initialization."""
        validation_errors = validate_company_data(self.to_dict())
        if validation_errors:
            raise ValueError(f"Invalid company data: {', '.join(validation_errors)}")


@dataclass
class ProcessingResult:
    """
    Tracks the outcome of a company processing operation.
    
    Attributes:
        new_companies_count: Number of new companies processed
        total_companies_count: Total number of companies in dataset
        errors: List of error messages encountered
        processing_time: Time taken for processing in seconds
        success: Whether the operation completed successfully
    """
    new_companies_count: int
    total_companies_count: int
    errors: List[str]
    processing_time: float
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ProcessingResult instance to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingResult':
        """Create ProcessingResult instance from dictionary."""
        return cls(
            new_companies_count=int(data.get('new_companies_count', 0)),
            total_companies_count=int(data.get('total_companies_count', 0)),
            errors=list(data.get('errors', [])),
            processing_time=float(data.get('processing_time', 0.0)),
            success=bool(data.get('success', False))
        )


def validate_company_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate company data integrity and return list of validation errors.
    
    Args:
        data: Dictionary containing company data
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate required fields
    required_fields = ['name', 'website', 'description', 'yc_page', 'linkedin_url', 'yc_s25_on_linkedin', 'last_updated']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate company name
    name = data.get('name', '')
    if not name or not isinstance(name, str) or len(name.strip()) == 0:
        errors.append("Company name must be a non-empty string")
    elif len(name) > 200:
        errors.append("Company name must be 200 characters or less")
    
    # Validate URLs
    url_fields = ['website', 'yc_page', 'linkedin_url']
    for field in url_fields:
        url = data.get(field, '')
        if url and not _is_valid_url(url):
            errors.append(f"Invalid URL format for {field}: {url}")
    
    # Validate YC page URL format
    yc_page = data.get('yc_page', '')
    if yc_page and not yc_page.startswith('https://www.ycombinator.com/companies/'):
        errors.append("YC page URL must start with 'https://www.ycombinator.com/companies/'")
    
    # Validate LinkedIn URL format (more flexible)
    linkedin_url = data.get('linkedin_url', '')
    if linkedin_url and linkedin_url.strip():  # Only validate non-empty URLs
        linkedin_url = linkedin_url.strip()
        
        # Check if it's a LinkedIn URL (very flexible validation)
        is_linkedin_url = (
            'linkedin.com' in linkedin_url.lower() and 
            (linkedin_url.startswith('http://') or linkedin_url.startswith('https://'))
        )
        
        # Additional check for common LinkedIn patterns
        if not is_linkedin_url:
            # Accept various LinkedIn URL formats (both HTTP and HTTPS, with or without www)
            valid_linkedin_patterns = [
                'https://www.linkedin.com/',
                'https://linkedin.com/',
                'http://www.linkedin.com/',
                'http://linkedin.com/'
            ]
            is_linkedin_url = any(linkedin_url.startswith(pattern) for pattern in valid_linkedin_patterns)
        
        if not is_linkedin_url:
            errors.append("LinkedIn URL must be a valid LinkedIn profile or company page")
    
    # Validate description
    description = data.get('description', '')
    if description and len(description) > 500:
        errors.append("Description must be 500 characters or less")
    
    # Validate boolean field (support both old and new field names)
    yc_batch_flag = data.get('yc_batch_on_linkedin')
    yc_s25_flag = data.get('yc_s25_on_linkedin')  # Backward compatibility
    
    if yc_batch_flag is not None and not isinstance(yc_batch_flag, bool):
        errors.append("yc_batch_on_linkedin must be a boolean value")
    if yc_s25_flag is not None and not isinstance(yc_s25_flag, bool):
        errors.append("yc_s25_on_linkedin must be a boolean value")
    
    # Validate timestamp format
    last_updated = data.get('last_updated', '')
    if last_updated and not _is_valid_iso_timestamp(last_updated):
        errors.append("last_updated must be a valid ISO timestamp")
    
    return errors


def validate_processing_result_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate processing result data integrity.
    
    Args:
        data: Dictionary containing processing result data
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate required fields
    required_fields = ['new_companies_count', 'total_companies_count', 'errors', 'processing_time', 'success']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate counts
    for count_field in ['new_companies_count', 'total_companies_count']:
        count = data.get(count_field)
        if count is not None and (not isinstance(count, int) or count < 0):
            errors.append(f"{count_field} must be a non-negative integer")
    
    # Validate processing time
    processing_time = data.get('processing_time')
    if processing_time is not None and (not isinstance(processing_time, (int, float)) or processing_time < 0):
        errors.append("processing_time must be a non-negative number")
    
    # Validate errors list
    errors_list = data.get('errors')
    if errors_list is not None:
        if not isinstance(errors_list, list):
            errors.append("errors must be a list")
        elif not all(isinstance(error, str) for error in errors_list):
            errors.append("all items in errors list must be strings")
    
    # Validate success flag
    success = data.get('success')
    if success is not None and not isinstance(success, bool):
        errors.append("success must be a boolean value")
    
    return errors


def _is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception:
        return False


def _is_valid_iso_timestamp(timestamp: str) -> bool:
    """Check if a string is a valid ISO timestamp."""
    try:
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


def create_company_from_yc_data(yc_data: Dict[str, Any]) -> Company:
    """
    Create a Company instance from YC API data.
    
    Args:
        yc_data: Raw data from YC API
        
    Returns:
        Company instance with default values for missing fields
    """
    return Company(
        name=yc_data.get('name', ''),
        website=yc_data.get('website', ''),
        description=yc_data.get('one_liner', ''),
        yc_page=yc_data.get('yc_page', ''),
        linkedin_url=yc_data.get('linkedin_url', ''),
        yc_s25_on_linkedin=False,  # Default to False, will be updated by LinkedIn scraper
        last_updated=datetime.now().isoformat()
    )


def create_empty_processing_result() -> ProcessingResult:
    """Create an empty ProcessingResult for initialization."""
    return ProcessingResult(
        new_companies_count=0,
        total_companies_count=0,
        errors=[],
        processing_time=0.0,
        success=False
    )