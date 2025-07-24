"""
Test script to verify the data models implementation.
"""

from yc_parser.models import (
    Company, ProcessingResult, 
    validate_company_data, validate_processing_result_data,
    create_company_from_yc_data, create_empty_processing_result
)
from datetime import datetime


def test_company_model():
    """Test Company dataclass functionality."""
    print("Testing Company model...")
    
    # Test valid company creation
    company = Company(
        name="Test Company",
        website="https://example.com",
        description="A test company",
        yc_page="https://www.ycombinator.com/companies/test-company",
        linkedin_url="https://www.linkedin.com/company/test-company",
        yc_s25_on_linkedin=True,
        last_updated=datetime.now().isoformat()
    )
    
    # Test serialization
    company_dict = company.to_dict()
    print(f"Company dict: {company_dict}")
    
    # Test deserialization
    company_from_dict = Company.from_dict(company_dict)
    print(f"Company from dict: {company_from_dict}")
    
    # Test validation with valid data
    errors = validate_company_data(company_dict)
    print(f"Validation errors for valid data: {errors}")
    assert len(errors) == 0, f"Expected no errors, got: {errors}"
    
    print("✓ Company model tests passed")


def test_processing_result_model():
    """Test ProcessingResult dataclass functionality."""
    print("\nTesting ProcessingResult model...")
    
    # Test valid processing result creation
    result = ProcessingResult(
        new_companies_count=5,
        total_companies_count=100,
        errors=["Test error"],
        processing_time=30.5,
        success=True
    )
    
    # Test serialization
    result_dict = result.to_dict()
    print(f"ProcessingResult dict: {result_dict}")
    
    # Test deserialization
    result_from_dict = ProcessingResult.from_dict(result_dict)
    print(f"ProcessingResult from dict: {result_from_dict}")
    
    # Test validation with valid data
    errors = validate_processing_result_data(result_dict)
    print(f"Validation errors for valid data: {errors}")
    assert len(errors) == 0, f"Expected no errors, got: {errors}"
    
    print("✓ ProcessingResult model tests passed")


def test_validation_functions():
    """Test data validation functions."""
    print("\nTesting validation functions...")
    
    # Test invalid company data
    invalid_company_data = {
        "name": "",  # Empty name
        "website": "invalid-url",  # Invalid URL
        "description": "A" * 600,  # Too long description
        "yc_page": "https://wrong-domain.com/companies/test",  # Wrong domain
        "linkedin_url": "https://wrong-domain.com/company/test",  # Wrong domain
        "yc_s25_on_linkedin": "not-a-boolean",  # Wrong type
        "last_updated": "invalid-timestamp"  # Invalid timestamp
    }
    
    errors = validate_company_data(invalid_company_data)
    print(f"Validation errors for invalid company data: {errors}")
    assert len(errors) > 0, "Expected validation errors for invalid data"
    
    # Test invalid processing result data
    invalid_result_data = {
        "new_companies_count": -1,  # Negative count
        "total_companies_count": "not-a-number",  # Wrong type
        "errors": "not-a-list",  # Wrong type
        "processing_time": -5.0,  # Negative time
        "success": "not-a-boolean"  # Wrong type
    }
    
    errors = validate_processing_result_data(invalid_result_data)
    print(f"Validation errors for invalid result data: {errors}")
    assert len(errors) > 0, "Expected validation errors for invalid data"
    
    print("✓ Validation function tests passed")


def test_helper_functions():
    """Test helper functions."""
    print("\nTesting helper functions...")
    
    # Test create_company_from_yc_data
    yc_data = {
        "name": "YC Test Company",
        "website": "https://yctest.com",
        "one_liner": "Testing YC integration",
        "yc_page": "https://www.ycombinator.com/companies/yc-test",
        "linkedin_url": "https://www.linkedin.com/company/yc-test"
    }
    
    company = create_company_from_yc_data(yc_data)
    print(f"Company from YC data: {company}")
    assert company.name == "YC Test Company"
    assert company.yc_s25_on_linkedin == False  # Default value
    
    # Test create_empty_processing_result
    empty_result = create_empty_processing_result()
    print(f"Empty processing result: {empty_result}")
    assert empty_result.new_companies_count == 0
    assert empty_result.success == False
    
    print("✓ Helper function tests passed")


if __name__ == "__main__":
    test_company_model()
    test_processing_result_model()
    test_validation_functions()
    test_helper_functions()
    print("\n🎉 All tests passed! Data models implementation is working correctly.")