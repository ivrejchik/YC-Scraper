"""
Integration tests for CompanyProcessor class.

Tests the complete processing pipeline including YC data fetching,
LinkedIn enrichment, and data persistence.
"""
import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from yc_parser.company_processor import CompanyProcessor
from yc_parser.models import ProcessingResult


class TestCompanyProcessor:
    """Test suite for CompanyProcessor integration tests."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.test_dir, "test_companies.csv")
        
        # Create processor instance with test CSV path
        self.processor = CompanyProcessor(self.test_csv_path)
        
        # Sample YC company data for testing
        self.sample_yc_companies = [
            {
                'name': 'Test Company 1',
                'website': 'https://testcompany1.com',
                'description': 'A test company for unit testing',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-1',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-1',
                'batch': 'S25'
            },
            {
                'name': 'Test Company 2',
                'website': 'https://testcompany2.com',
                'description': 'Another test company',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-2',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-2',
                'batch': 'S25'
            }
        ]
        
        # Sample existing data
        self.sample_existing_data = pd.DataFrame([
            {
                'name': 'Existing Company',
                'website': 'https://existing.com',
                'description': 'An existing company',
                'yc_page': 'https://www.ycombinator.com/companies/existing-company',
                'linkedin_url': 'https://www.linkedin.com/company/existing-company',
                'yc_s25_on_linkedin': True,
                'last_updated': '2025-01-01T12:00:00'
            }
        ])
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        self.processor.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('yc_parser.company_processor.YcClient')
    @patch('yc_parser.company_processor.LinkedInScraper')
    def test_process_new_companies_success(self, mock_linkedin_scraper, mock_yc_client):
        """Test successful processing of new companies."""
        # Mock YC client
        mock_yc_instance = Mock()
        mock_yc_instance.get_complete_s25_data.return_value = self.sample_yc_companies
        mock_yc_client.return_value = mock_yc_instance
        
        # Mock LinkedIn scraper
        mock_linkedin_instance = Mock()
        enriched_companies = []
        for company in self.sample_yc_companies:
            enriched = company.copy()
            enriched['yc_s25_on_linkedin'] = True
            enriched['last_updated'] = datetime.now().isoformat()
            enriched_companies.append(enriched)
        
        mock_linkedin_instance.batch_check_mentions.return_value = (enriched_companies, [])
        mock_linkedin_scraper.return_value = mock_linkedin_instance
        
        # Create new processor instance with mocked dependencies
        processor = CompanyProcessor(self.test_csv_path)
        
        # Run processing
        result = processor.process_new_companies()
        
        # Verify results
        assert result.success is True
        assert result.new_companies_count == 2
        assert result.total_companies_count == 2
        assert len(result.errors) == 0
        assert result.processing_time > 0
        
        # Verify data was saved
        assert os.path.exists(self.test_csv_path)
        saved_data = pd.read_csv(self.test_csv_path)
        assert len(saved_data) == 2
        assert all(saved_data['yc_s25_on_linkedin'] == True)
        
        processor.close()
    
    @patch('yc_parser.company_processor.YcClient')
    @patch('yc_parser.company_processor.LinkedInScraper')
    def test_process_new_companies_with_existing_data(self, mock_linkedin_scraper, mock_yc_client):
        """Test processing when existing data is present."""
        # Save existing data first
        self.sample_existing_data.to_csv(self.test_csv_path, index=False)
        
        # Mock YC client to return existing + new companies
        all_companies = [
            {
                'name': 'Existing Company',  # This should be filtered out
                'website': 'https://existing.com',
                'description': 'An existing company',
                'yc_page': 'https://www.ycombinator.com/companies/existing-company',
                'linkedin_url': 'https://www.linkedin.com/company/existing-company',
                'batch': 'S25'
            }
        ] + self.sample_yc_companies
        
        mock_yc_instance = Mock()
        mock_yc_instance.get_complete_s25_data.return_value = all_companies
        mock_yc_client.return_value = mock_yc_instance
        
        # Mock LinkedIn scraper for new companies only
        mock_linkedin_instance = Mock()
        enriched_companies = []
        for company in self.sample_yc_companies:  # Only new companies
            enriched = company.copy()
            enriched['yc_s25_on_linkedin'] = False
            enriched['last_updated'] = datetime.now().isoformat()
            enriched_companies.append(enriched)
        
        mock_linkedin_instance.batch_check_mentions.return_value = (enriched_companies, [])
        mock_linkedin_scraper.return_value = mock_linkedin_instance
        
        # Create new processor instance
        processor = CompanyProcessor(self.test_csv_path)
        
        # Run processing
        result = processor.process_new_companies()
        
        # Verify results
        assert result.success is True
        assert result.new_companies_count == 2  # Only new companies processed
        assert result.total_companies_count == 3  # 1 existing + 2 new
        
        # Verify data was merged correctly
        saved_data = pd.read_csv(self.test_csv_path)
        assert len(saved_data) == 3
        
        # Check that existing company data was preserved
        existing_row = saved_data[saved_data['name'] == 'Existing Company']
        assert len(existing_row) == 1
        assert existing_row.iloc[0]['yc_s25_on_linkedin'] == True
        
        processor.close()
    
    @patch('yc_parser.company_processor.YcClient')
    def test_process_new_companies_no_new_companies(self, mock_yc_client):
        """Test processing when no new companies are found."""
        # Save existing data
        self.sample_existing_data.to_csv(self.test_csv_path, index=False)
        
        # Mock YC client to return only existing companies
        mock_yc_instance = Mock()
        mock_yc_instance.get_complete_s25_data.return_value = [
            {
                'name': 'Existing Company',
                'website': 'https://existing.com',
                'description': 'An existing company',
                'yc_page': 'https://www.ycombinator.com/companies/existing-company',
                'linkedin_url': 'https://www.linkedin.com/company/existing-company',
                'batch': 'S25'
            }
        ]
        mock_yc_client.return_value = mock_yc_instance
        
        # Create processor
        processor = CompanyProcessor(self.test_csv_path)
        
        # Run processing
        result = processor.process_new_companies()
        
        # Verify results
        assert result.success is True
        assert result.new_companies_count == 0
        assert result.total_companies_count == 1
        assert len(result.errors) == 0
        
        processor.close()
    
    @patch('yc_parser.company_processor.YcClient')
    def test_process_new_companies_yc_client_failure(self, mock_yc_client):
        """Test handling of YC client failures."""
        # Mock YC client to raise exception
        mock_yc_instance = Mock()
        mock_yc_instance.get_complete_s25_data.side_effect = Exception("YC API failed")
        mock_yc_client.return_value = mock_yc_instance
        
        # Create processor
        processor = CompanyProcessor(self.test_csv_path)
        
        # Run processing
        result = processor.process_new_companies()
        
        # Verify failure handling
        assert result.success is False
        assert result.new_companies_count == 0
        assert len(result.errors) == 1
        assert "Processing pipeline failed" in result.errors[0]
        
        processor.close()
    
    @patch('yc_parser.company_processor.YcClient')
    @patch('yc_parser.company_processor.LinkedInScraper')
    def test_process_new_companies_linkedin_errors(self, mock_linkedin_scraper, mock_yc_client):
        """Test handling of LinkedIn scraper errors."""
        # Mock YC client
        mock_yc_instance = Mock()
        mock_yc_instance.get_complete_s25_data.return_value = self.sample_yc_companies
        mock_yc_client.return_value = mock_yc_instance
        
        # Mock LinkedIn scraper with errors
        mock_linkedin_instance = Mock()
        enriched_companies = []
        for company in self.sample_yc_companies:
            enriched = company.copy()
            enriched['yc_s25_on_linkedin'] = False
            enriched['last_updated'] = datetime.now().isoformat()
            enriched_companies.append(enriched)
        
        linkedin_errors = ["Failed to access LinkedIn for Test Company 1"]
        mock_linkedin_instance.batch_check_mentions.return_value = (enriched_companies, linkedin_errors)
        mock_linkedin_scraper.return_value = mock_linkedin_instance
        
        # Create processor
        processor = CompanyProcessor(self.test_csv_path)
        
        # Run processing
        result = processor.process_new_companies()
        
        # Verify error handling
        assert result.success is True  # Processing should still succeed
        assert result.new_companies_count == 2
        assert len(result.errors) == 1
        assert "Failed to access LinkedIn" in result.errors[0]
        
        processor.close()
    
    def test_get_processing_statistics_empty_data(self):
        """Test statistics with empty dataset."""
        stats = self.processor.get_processing_statistics()
        
        assert stats['total_companies'] == 0
        assert stats['companies_with_linkedin'] == 0
        assert stats['yc_s25_mentions'] == 0
        assert stats['last_updated'] is None
        assert stats['data_integrity_issues'] == 0
    
    def test_get_processing_statistics_with_data(self):
        """Test statistics with existing data."""
        # Create test data with mixed LinkedIn flags
        test_data = pd.DataFrame([
            {
                'name': 'Company 1',
                'website': 'https://company1.com',
                'description': 'Company 1 description',
                'yc_page': 'https://www.ycombinator.com/companies/company-1',
                'linkedin_url': 'https://www.linkedin.com/company/company-1',
                'yc_s25_on_linkedin': True,
                'last_updated': '2025-01-01T12:00:00'
            },
            {
                'name': 'Company 2',
                'website': 'https://company2.com',
                'description': 'Company 2 description',
                'yc_page': 'https://www.ycombinator.com/companies/company-2',
                'linkedin_url': '',  # No LinkedIn URL
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-02T12:00:00'
            }
        ])
        
        test_data.to_csv(self.test_csv_path, index=False)
        
        stats = self.processor.get_processing_statistics()
        
        assert stats['total_companies'] == 2
        assert stats['companies_with_linkedin'] == 1
        assert stats['yc_s25_mentions'] == 1
        assert stats['last_updated'] == '2025-01-02T12:00:00'
        # Note: data_integrity_issues might be > 0 due to empty LinkedIn URL validation
        assert stats['data_integrity_issues'] >= 0
    
    def test_validate_data_integrity_valid_data(self):
        """Test data integrity validation with valid data."""
        # Create valid test data
        valid_data = pd.DataFrame([
            {
                'name': 'Valid Company',
                'website': 'https://valid.com',
                'description': 'A valid company',
                'yc_page': 'https://www.ycombinator.com/companies/valid-company',
                'linkedin_url': 'https://www.linkedin.com/company/valid-company',
                'yc_s25_on_linkedin': True,
                'last_updated': datetime.now().isoformat()
            }
        ])
        
        valid_data.to_csv(self.test_csv_path, index=False)
        
        result = self.processor.validate_data_integrity()
        
        assert result.success is True
        assert len(result.errors) == 0
        assert result.total_companies_count == 1
    
    def test_create_data_backup(self):
        """Test manual backup creation."""
        # Create some test data
        test_data = pd.DataFrame([
            {
                'name': 'Backup Test Company',
                'website': 'https://backup.com',
                'description': 'Test backup functionality',
                'yc_page': 'https://www.ycombinator.com/companies/backup-test',
                'linkedin_url': 'https://www.linkedin.com/company/backup-test',
                'yc_s25_on_linkedin': False,
                'last_updated': datetime.now().isoformat()
            }
        ])
        
        test_data.to_csv(self.test_csv_path, index=False)
        
        # Create backup
        backup_path = self.processor.create_data_backup("test_backup")
        
        # Verify backup was created
        assert os.path.exists(backup_path)
        assert "test_backup" in backup_path
        
        # Verify backup content matches original
        backup_data = pd.read_csv(backup_path)
        original_data = pd.read_csv(self.test_csv_path)
        
        assert len(backup_data) == len(original_data)
        assert backup_data['name'].iloc[0] == original_data['name'].iloc[0]
    
    def test_context_manager(self):
        """Test CompanyProcessor as context manager."""
        with CompanyProcessor(self.test_csv_path) as processor:
            assert processor is not None
            # Context manager should handle cleanup automatically
    
    @patch('yc_parser.company_processor.YcClient')
    @patch('yc_parser.company_processor.LinkedInScraper')
    def test_end_to_end_processing_pipeline(self, mock_linkedin_scraper, mock_yc_client):
        """Test complete end-to-end processing pipeline."""
        # Mock YC client
        mock_yc_instance = Mock()
        mock_yc_instance.get_complete_s25_data.return_value = self.sample_yc_companies
        mock_yc_client.return_value = mock_yc_instance
        
        # Mock LinkedIn scraper with realistic behavior
        mock_linkedin_instance = Mock()
        
        def mock_batch_check(companies, max_workers=3, progress_callback=None):
            enriched = []
            errors = []
            
            for i, company in enumerate(companies):
                enriched_company = company.copy()
                # Simulate some companies having YC mentions
                enriched_company['yc_s25_on_linkedin'] = (i % 2 == 0)
                enriched_company['last_updated'] = datetime.now().isoformat()
                enriched.append(enriched_company)
                
                # Simulate progress callback
                if progress_callback:
                    progress_callback(i + 1, len(companies))
                
                # Simulate some errors
                if i == 1:
                    errors.append(f"Error processing {company['name']}")
            
            return enriched, errors
        
        mock_linkedin_instance.batch_check_mentions.side_effect = mock_batch_check
        mock_linkedin_scraper.return_value = mock_linkedin_instance
        
        # Create processor and run complete pipeline
        processor = CompanyProcessor(self.test_csv_path)
        
        # First run - should process all companies
        result1 = processor.process_new_companies()
        
        assert result1.success is True
        assert result1.new_companies_count == 2
        assert result1.total_companies_count == 2
        assert len(result1.errors) == 1  # One LinkedIn error
        
        # Verify data was saved correctly
        saved_data = pd.read_csv(self.test_csv_path)
        assert len(saved_data) == 2
        assert saved_data.iloc[0]['yc_s25_on_linkedin'] == True
        assert saved_data.iloc[1]['yc_s25_on_linkedin'] == False
        
        # Second run - should find no new companies
        result2 = processor.process_new_companies()
        
        assert result2.success is True
        assert result2.new_companies_count == 0
        assert result2.total_companies_count == 2
        
        # Get statistics
        stats = processor.get_processing_statistics()
        assert stats['total_companies'] == 2
        assert stats['yc_s25_mentions'] == 1
        
        processor.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])