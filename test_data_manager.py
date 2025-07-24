"""
Unit tests for the DataManager class.

Tests CSV data operations, deduplication logic, validation,
and error handling scenarios.
"""

import unittest
import pandas as pd
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock

from yc_parser.data_manager import DataManager
from yc_parser.models import Company


class TestDataManager(unittest.TestCase):
    """Test cases for DataManager class."""
    
    def setUp(self):
        """Set up test environment with temporary files."""
        self.test_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.test_dir, "test_companies.csv")
        self.data_manager = DataManager(csv_file_path=self.csv_file)
        self.data_manager.backup_dir = os.path.join(self.test_dir, "backups")
        
        # Sample company data for testing
        self.sample_companies = [
            {
                'name': 'Test Company 1',
                'website': 'https://testcompany1.com',
                'description': 'A test company',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-1',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-1',
                'yc_s25_on_linkedin': True,
                'last_updated': '2025-01-01T10:00:00'
            },
            {
                'name': 'Test Company 2',
                'website': 'https://testcompany2.com',
                'description': 'Another test company',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-2',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-2',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T11:00:00'
            }
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_load_existing_data_file_not_exists(self):
        """Test loading data when CSV file doesn't exist."""
        df = self.data_manager.load_existing_data()
        
        self.assertTrue(df.empty)
        expected_columns = {
            'name', 'website', 'description', 'yc_page', 
            'linkedin_url', 'yc_s25_on_linkedin', 'last_updated'
        }
        self.assertEqual(set(df.columns), expected_columns)
    
    def test_save_and_load_data(self):
        """Test saving and loading data roundtrip."""
        # Create DataFrame from sample data
        df = pd.DataFrame(self.sample_companies)
        
        # Save data
        self.data_manager.save_data(df)
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.csv_file))
        
        # Load data back
        loaded_df = self.data_manager.load_existing_data()
        
        # Verify data integrity
        self.assertEqual(len(loaded_df), 2)
        self.assertEqual(loaded_df.iloc[0]['name'], 'Test Company 1')
        self.assertEqual(loaded_df.iloc[1]['name'], 'Test Company 2')
        self.assertTrue(loaded_df.iloc[0]['yc_s25_on_linkedin'])
        self.assertFalse(loaded_df.iloc[1]['yc_s25_on_linkedin'])
    
    def test_save_empty_dataframe(self):
        """Test saving empty DataFrame creates file with headers."""
        empty_df = pd.DataFrame()
        self.data_manager.save_data(empty_df)
        
        self.assertTrue(os.path.exists(self.csv_file))
        
        loaded_df = self.data_manager.load_existing_data()
        self.assertTrue(loaded_df.empty)
        expected_columns = {
            'name', 'website', 'description', 'yc_page', 
            'linkedin_url', 'yc_s25_on_linkedin', 'last_updated'
        }
        self.assertEqual(set(loaded_df.columns), expected_columns)
    
    def test_identify_new_companies_empty_existing(self):
        """Test identifying new companies when no existing data."""
        existing_df = pd.DataFrame()
        new_companies = self.data_manager.identify_new_companies(
            self.sample_companies, existing_df
        )
        
        self.assertEqual(len(new_companies), 2)
        self.assertEqual(new_companies, self.sample_companies)
    
    def test_identify_new_companies_with_existing(self):
        """Test identifying new companies with existing data."""
        # Create existing data with one company
        existing_df = pd.DataFrame([self.sample_companies[0]])
        
        # Add a new company to current list
        current_companies = self.sample_companies + [{
            'name': 'New Company',
            'website': 'https://newcompany.com',
            'description': 'A new company',
            'yc_page': 'https://www.ycombinator.com/companies/new-company',
            'linkedin_url': 'https://www.linkedin.com/company/new-company',
            'yc_s25_on_linkedin': False,
            'last_updated': '2025-01-01T12:00:00'
        }]
        
        new_companies = self.data_manager.identify_new_companies(
            current_companies, existing_df
        )
        
        self.assertEqual(len(new_companies), 2)  # Test Company 2 and New Company
        company_names = [c['name'] for c in new_companies]
        self.assertIn('Test Company 2', company_names)
        self.assertIn('New Company', company_names)
        self.assertNotIn('Test Company 1', company_names)
    
    def test_identify_new_companies_case_insensitive(self):
        """Test that company name comparison is case-insensitive."""
        existing_df = pd.DataFrame([{
            'name': 'test company 1',  # lowercase
            'website': 'https://testcompany1.com',
            'description': 'A test company',
            'yc_page': 'https://www.ycombinator.com/companies/test-company-1',
            'linkedin_url': 'https://www.linkedin.com/company/test-company-1',
            'yc_s25_on_linkedin': True,
            'last_updated': '2025-01-01T10:00:00'
        }])
        
        current_companies = [{
            'name': 'Test Company 1',  # different case
            'website': 'https://testcompany1.com',
            'description': 'A test company',
            'yc_page': 'https://www.ycombinator.com/companies/test-company-1',
            'linkedin_url': 'https://www.linkedin.com/company/test-company-1',
            'yc_s25_on_linkedin': True,
            'last_updated': '2025-01-01T10:00:00'
        }]
        
        new_companies = self.data_manager.identify_new_companies(
            current_companies, existing_df
        )
        
        self.assertEqual(len(new_companies), 0)  # Should not be considered new
    
    def test_merge_and_deduplicate_no_duplicates(self):
        """Test merging data with no duplicates."""
        existing_df = pd.DataFrame([self.sample_companies[0]])
        new_data = [self.sample_companies[1]]
        
        merged_df = self.data_manager.merge_and_deduplicate(new_data, existing_df)
        
        self.assertEqual(len(merged_df), 2)
        company_names = merged_df['name'].tolist()
        self.assertIn('Test Company 1', company_names)
        self.assertIn('Test Company 2', company_names)
    
    def test_merge_and_deduplicate_with_duplicates(self):
        """Test merging data with duplicates keeps most recent."""
        existing_df = pd.DataFrame([self.sample_companies[0]])
        
        # Create updated version of same company
        updated_company = self.sample_companies[0].copy()
        updated_company['description'] = 'Updated description'
        updated_company['last_updated'] = '2025-01-02T10:00:00'  # More recent
        
        new_data = [updated_company]
        
        merged_df = self.data_manager.merge_and_deduplicate(new_data, existing_df)
        
        self.assertEqual(len(merged_df), 1)  # Should have only one company
        self.assertEqual(merged_df.iloc[0]['description'], 'Updated description')
        self.assertEqual(merged_df.iloc[0]['last_updated'], '2025-01-02T10:00:00')
    
    def test_merge_and_deduplicate_empty_new_data(self):
        """Test merging with empty new data returns existing data."""
        existing_df = pd.DataFrame(self.sample_companies)
        new_data = []
        
        merged_df = self.data_manager.merge_and_deduplicate(new_data, existing_df)
        
        self.assertEqual(len(merged_df), 2)
        pd.testing.assert_frame_equal(merged_df, existing_df)
    
    def test_validate_data_integrity_valid_data(self):
        """Test data validation with valid data."""
        df = pd.DataFrame(self.sample_companies)
        
        is_valid, errors = self.data_manager.validate_data_integrity(df)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_data_integrity_invalid_structure(self):
        """Test data validation with invalid DataFrame structure."""
        df = pd.DataFrame({'invalid_column': ['value']})
        
        is_valid, errors = self.data_manager.validate_data_integrity(df)
        
        self.assertFalse(is_valid)
        self.assertIn('Invalid DataFrame structure', errors)
    
    def test_validate_data_integrity_invalid_data(self):
        """Test data validation with invalid data values."""
        invalid_data = self.sample_companies[0].copy()
        invalid_data['name'] = ''  # Invalid empty name
        invalid_data['website'] = 'invalid-url'  # Invalid URL
        
        df = pd.DataFrame([invalid_data])
        
        is_valid, errors = self.data_manager.validate_data_integrity(df)
        
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
    
    def test_backup_creation(self):
        """Test that backups are created when saving data."""
        # Create initial data
        df = pd.DataFrame([self.sample_companies[0]])
        self.data_manager.save_data(df)
        
        # Save updated data (should create backup)
        updated_df = pd.DataFrame(self.sample_companies)
        self.data_manager.save_data(updated_df)
        
        # Check backup directory exists and has files
        self.assertTrue(os.path.exists(self.data_manager.backup_dir))
        backup_files = [f for f in os.listdir(self.data_manager.backup_dir) 
                       if f.startswith('yc_s25_companies_backup_')]
        self.assertTrue(len(backup_files) > 0)
    
    @patch('pandas.read_csv')
    def test_load_data_with_corruption_recovery(self, mock_read_csv):
        """Test recovery from backup when main file is corrupted."""
        # Mock corrupted main file
        mock_read_csv.side_effect = [
            pd.errors.EmptyDataError("Corrupted file"),  # Main file fails
            pd.DataFrame(self.sample_companies)  # Backup succeeds
        ]
        
        # Create a backup file
        os.makedirs(self.data_manager.backup_dir, exist_ok=True)
        backup_file = os.path.join(self.data_manager.backup_dir, 
                                  'yc_s25_companies_backup_20250101_120000.csv')
        pd.DataFrame(self.sample_companies).to_csv(backup_file, index=False)
        
        # Create main file (will be mocked as corrupted)
        with open(self.csv_file, 'w') as f:
            f.write("corrupted data")
        
        df = self.data_manager.load_existing_data()
        
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['name'], 'Test Company 1')
    
    @patch('pandas.read_csv')
    def test_load_data_no_backup_available(self, mock_read_csv):
        """Test handling when main file is corrupted and no backup exists."""
        mock_read_csv.side_effect = pd.errors.EmptyDataError("Corrupted file")
        
        # Create main file (will be mocked as corrupted)
        with open(self.csv_file, 'w') as f:
            f.write("corrupted data")
        
        df = self.data_manager.load_existing_data()
        
        # Should return empty DataFrame with proper structure
        self.assertTrue(df.empty)
        expected_columns = {
            'name', 'website', 'description', 'yc_page', 
            'linkedin_url', 'yc_s25_on_linkedin', 'last_updated'
        }
        self.assertEqual(set(df.columns), expected_columns)
    
    def test_save_invalid_data_raises_error(self):
        """Test that saving invalid data raises an error."""
        invalid_data = [{
            'name': '',  # Invalid empty name
            'website': 'invalid-url',  # Invalid URL
            'description': 'Test',
            'yc_page': 'https://www.ycombinator.com/companies/test',
            'linkedin_url': 'https://www.linkedin.com/company/test',
            'yc_s25_on_linkedin': True,
            'last_updated': '2025-01-01T10:00:00'
        }]
        
        df = pd.DataFrame(invalid_data)
        
        with self.assertRaises(ValueError) as context:
            self.data_manager.save_data(df)
        
        self.assertIn('Cannot save invalid data', str(context.exception))
    
    def test_cleanup_old_backups(self):
        """Test that old backups are cleaned up, keeping only 5 most recent."""
        # First create an initial CSV file so backup will be triggered
        initial_df = pd.DataFrame([self.sample_companies[0]])
        self.data_manager.save_data(initial_df)
        
        # Create 7 backup files with different timestamps
        os.makedirs(self.data_manager.backup_dir, exist_ok=True)
        
        backup_files = []
        for i in range(7):
            timestamp = f"2025010{i+1}_120000"
            backup_file = os.path.join(self.data_manager.backup_dir, 
                                      f'yc_s25_companies_backup_{timestamp}.csv')
            pd.DataFrame([self.sample_companies[0]]).to_csv(backup_file, index=False)
            backup_files.append(backup_file)
        
        # Verify we have more than 5 files before cleanup
        initial_backups = [f for f in os.listdir(self.data_manager.backup_dir) 
                          if f.startswith('yc_s25_companies_backup_')]
        self.assertGreater(len(initial_backups), 5)
        
        # Trigger cleanup by saving data (this creates a backup first, then cleans up)
        df = pd.DataFrame(self.sample_companies)
        self.data_manager.save_data(df)
        
        # Check that cleanup occurred - should have at most 5 files now
        remaining_backups = [f for f in os.listdir(self.data_manager.backup_dir) 
                           if f.startswith('yc_s25_companies_backup_')]
        self.assertLessEqual(len(remaining_backups), 5)


    def test_deduplicate_with_conflict_resolution(self):
        """Test advanced deduplication with conflict resolution."""
        # Create duplicate entries with different data completeness
        duplicate_data = [
            {
                'name': 'Test Company',
                'website': 'https://testcompany.com',
                'description': '',  # Empty description
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': '',  # Empty LinkedIn
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            },
            {
                'name': 'Test Company',  # Same name
                'website': '',  # Empty website
                'description': 'A test company description',  # Has description
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://www.linkedin.com/company/test-company',  # Has LinkedIn
                'yc_s25_on_linkedin': True,
                'last_updated': '2025-01-02T10:00:00'  # More recent
            }
        ]
        
        df = pd.DataFrame(duplicate_data)
        deduplicated_df = self.data_manager._deduplicate_with_conflict_resolution(df)
        
        # Should have only one entry
        self.assertEqual(len(deduplicated_df), 1)
        
        # Should combine the best data from both entries
        result = deduplicated_df.iloc[0]
        self.assertEqual(result['name'], 'Test Company')
        self.assertEqual(result['website'], 'https://testcompany.com')  # From first entry
        self.assertEqual(result['description'], 'A test company description')  # From second entry
        self.assertEqual(result['linkedin_url'], 'https://www.linkedin.com/company/test-company')  # From second entry
        self.assertTrue(result['yc_s25_on_linkedin'])  # True from more recent entry
        self.assertEqual(result['last_updated'], '2025-01-02T10:00:00')  # Most recent timestamp
    
    def test_resolve_duplicate_conflicts_linkedin_flag_preference(self):
        """Test that LinkedIn flag prefers True from recent entries."""
        duplicate_data = [
            {
                'name': 'Test Company',
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://www.linkedin.com/company/test-company',
                'yc_s25_on_linkedin': True,
                'last_updated': '2025-01-01T10:00:00'  # Older but has True
            },
            {
                'name': 'Test Company',
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://www.linkedin.com/company/test-company',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-02T10:00:00'  # More recent but has False
            }
        ]
        
        df = pd.DataFrame(duplicate_data)
        deduplicated_df = self.data_manager._deduplicate_with_conflict_resolution(df)
        
        # Should prefer True even if it's from an older entry (within 30 days)
        result = deduplicated_df.iloc[0]
        self.assertTrue(result['yc_s25_on_linkedin'])
    
    def test_find_potential_duplicates(self):
        """Test finding potential duplicates based on name similarity."""
        similar_companies = [
            {
                'name': 'Test Company Inc',
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://www.linkedin.com/company/test-company',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            },
            {
                'name': 'Test Company LLC',  # Similar name
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-llc',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-llc',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            },
            {
                'name': 'Different Company',  # Different name
                'website': 'https://different.com',
                'description': 'Different description',
                'yc_page': 'https://www.ycombinator.com/companies/different-company',
                'linkedin_url': 'https://www.linkedin.com/company/different-company',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            }
        ]
        
        df = pd.DataFrame(similar_companies)
        duplicates = self.data_manager.find_potential_duplicates(df, similarity_threshold=0.7)
        
        # Should find one pair of similar companies
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0][0], 0)  # First company index
        self.assertEqual(duplicates[0][1], 1)  # Second company index
        self.assertGreater(duplicates[0][2], 0.7)  # Similarity score
    
    def test_remove_duplicates_by_similarity(self):
        """Test removing duplicates based on name similarity."""
        similar_companies = [
            {
                'name': 'Test Company Inc',
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': '',  # Less complete
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            },
            {
                'name': 'Test Company LLC',  # Similar name
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-llc',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-llc',  # More complete
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            }
        ]
        
        df = pd.DataFrame(similar_companies)
        deduplicated_df = self.data_manager.remove_duplicates_by_similarity(df, similarity_threshold=0.7)
        
        # Should keep only one company (the more complete one)
        self.assertEqual(len(deduplicated_df), 1)
        self.assertEqual(deduplicated_df.iloc[0]['name'], 'Test Company LLC')
        self.assertEqual(deduplicated_df.iloc[0]['linkedin_url'], 'https://www.linkedin.com/company/test-company-llc')
    
    def test_remove_duplicates_by_similarity_prefer_recent(self):
        """Test that similarity-based deduplication prefers more recent entries when completeness is equal."""
        similar_companies = [
            {
                'name': 'Test Company Inc',
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://www.linkedin.com/company/test-company',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'  # Older
            },
            {
                'name': 'Test Company LLC',  # Similar name
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-llc',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-llc',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-02T10:00:00'  # More recent
            }
        ]
        
        df = pd.DataFrame(similar_companies)
        deduplicated_df = self.data_manager.remove_duplicates_by_similarity(df, similarity_threshold=0.7)
        
        # Should keep the more recent entry
        self.assertEqual(len(deduplicated_df), 1)
        self.assertEqual(deduplicated_df.iloc[0]['name'], 'Test Company LLC')
        self.assertEqual(deduplicated_df.iloc[0]['last_updated'], '2025-01-02T10:00:00')
    
    def test_merge_and_deduplicate_with_advanced_logic(self):
        """Test that merge_and_deduplicate uses the advanced conflict resolution."""
        existing_data = [{
            'name': 'Test Company',
            'website': '',  # Empty
            'description': 'Original description',
            'yc_page': 'https://www.ycombinator.com/companies/test-company',
            'linkedin_url': '',  # Empty
            'yc_s25_on_linkedin': False,
            'last_updated': '2025-01-01T10:00:00'
        }]
        
        new_data = [{
            'name': 'Test Company',  # Same name
            'website': 'https://testcompany.com',  # Has website
            'description': '',  # Empty description
            'yc_page': 'https://www.ycombinator.com/companies/test-company',
            'linkedin_url': 'https://www.linkedin.com/company/test-company',  # Has LinkedIn
            'yc_s25_on_linkedin': True,
            'last_updated': '2025-01-02T10:00:00'  # More recent
        }]
        
        existing_df = pd.DataFrame(existing_data)
        merged_df = self.data_manager.merge_and_deduplicate(new_data, existing_df)
        
        # Should have only one entry with combined data
        self.assertEqual(len(merged_df), 1)
        result = merged_df.iloc[0]
        
        self.assertEqual(result['name'], 'Test Company')
        self.assertEqual(result['website'], 'https://testcompany.com')  # From new data
        self.assertEqual(result['description'], 'Original description')  # From existing data
        self.assertEqual(result['linkedin_url'], 'https://www.linkedin.com/company/test-company')  # From new data
        self.assertTrue(result['yc_s25_on_linkedin'])  # True from more recent entry
        self.assertEqual(result['last_updated'], '2025-01-02T10:00:00')  # Most recent
    
    def test_validate_and_clean_data_empty_names(self):
        """Test that validate_and_clean_data removes companies with empty names."""
        dirty_data = [
            {
                'name': '',  # Empty name
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://www.linkedin.com/company/test-company',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            },
            {
                'name': 'Valid Company',
                'website': 'https://validcompany.com',
                'description': 'Valid description',
                'yc_page': 'https://www.ycombinator.com/companies/valid-company',
                'linkedin_url': 'https://www.linkedin.com/company/valid-company',
                'yc_s25_on_linkedin': True,
                'last_updated': '2025-01-01T10:00:00'
            }
        ]
        
        df = pd.DataFrame(dirty_data)
        cleaned_df, warnings = self.data_manager.validate_and_clean_data(df)
        
        # Should remove the company with empty name
        self.assertEqual(len(cleaned_df), 1)
        self.assertEqual(cleaned_df.iloc[0]['name'], 'Valid Company')
        self.assertTrue(any('empty names' in warning for warning in warnings))
    
    def test_validate_and_clean_data_invalid_urls(self):
        """Test that validate_and_clean_data fixes invalid URLs."""
        dirty_data = [{
            'name': 'Test Company',
            'website': 'invalid-url',  # Invalid URL
            'description': 'Test description',
            'yc_page': 'https://www.ycombinator.com/companies/test-company',
            'linkedin_url': 'not-a-url',  # Invalid URL
            'yc_s25_on_linkedin': False,
            'last_updated': '2025-01-01T10:00:00'
        }]
        
        df = pd.DataFrame(dirty_data)
        cleaned_df, warnings = self.data_manager.validate_and_clean_data(df)
        
        # Should clean invalid URLs
        self.assertEqual(cleaned_df.iloc[0]['website'], '')
        self.assertEqual(cleaned_df.iloc[0]['linkedin_url'], '')
        self.assertTrue(any('invalid URLs' in warning for warning in warnings))
    
    def test_validate_and_clean_data_long_descriptions(self):
        """Test that validate_and_clean_data truncates long descriptions."""
        long_description = 'A' * 600  # 600 characters, over the 500 limit
        
        dirty_data = [{
            'name': 'Test Company',
            'website': 'https://testcompany.com',
            'description': long_description,
            'yc_page': 'https://www.ycombinator.com/companies/test-company',
            'linkedin_url': 'https://www.linkedin.com/company/test-company',
            'yc_s25_on_linkedin': False,
            'last_updated': '2025-01-01T10:00:00'
        }]
        
        df = pd.DataFrame(dirty_data)
        cleaned_df, warnings = self.data_manager.validate_and_clean_data(df)
        
        # Should truncate description to 500 characters
        self.assertEqual(len(cleaned_df.iloc[0]['description']), 500)
        self.assertTrue(any('over 500 characters' in warning for warning in warnings))
    
    def test_validate_and_clean_data_boolean_conversion(self):
        """Test that validate_and_clean_data properly converts boolean values."""
        dirty_data = [
            {
                'name': 'Test Company 1',
                'website': 'https://testcompany1.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-1',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-1',
                'yc_s25_on_linkedin': 'true',  # String representation
                'last_updated': '2025-01-01T10:00:00'
            },
            {
                'name': 'Test Company 2',
                'website': 'https://testcompany2.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company-2',
                'linkedin_url': 'https://www.linkedin.com/company/test-company-2',
                'yc_s25_on_linkedin': '0',  # String representation
                'last_updated': '2025-01-01T10:00:00'
            }
        ]
        
        df = pd.DataFrame(dirty_data)
        cleaned_df, warnings = self.data_manager.validate_and_clean_data(df)
        
        # Should convert string representations to proper booleans
        self.assertTrue(cleaned_df.iloc[0]['yc_s25_on_linkedin'])
        self.assertFalse(cleaned_df.iloc[1]['yc_s25_on_linkedin'])
    
    def test_perform_data_integrity_check_empty_data(self):
        """Test integrity check with empty data."""
        result = self.data_manager.perform_data_integrity_check()
        
        self.assertTrue(result.success)
        self.assertEqual(result.total_companies_count, 0)
        self.assertEqual(len(result.errors), 0)
    
    def test_perform_data_integrity_check_with_issues(self):
        """Test integrity check that finds issues."""
        # Create data with issues
        problematic_data = [
            {
                'name': 'Test Company 1',
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://www.linkedin.com/company/test-company',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            },
            {
                'name': 'Test Company 1',  # Duplicate name
                'website': 'https://testcompany.com',
                'description': 'Test description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://www.linkedin.com/company/test-company',
                'yc_s25_on_linkedin': False,
                'last_updated': '2025-01-01T10:00:00'
            }
        ]
        
        df = pd.DataFrame(problematic_data)
        self.data_manager.save_data(df)
        
        result = self.data_manager.perform_data_integrity_check()
        
        self.assertFalse(result.success)
        self.assertEqual(result.total_companies_count, 2)
        self.assertGreater(len(result.errors), 0)
    
    def test_repair_corrupted_data_success(self):
        """Test successful data repair."""
        # Create data that needs cleaning and save it directly to CSV (bypassing validation)
        dirty_data = [{
            'name': 'Test Company',
            'website': 'invalid-url',  # Will be cleaned
            'description': 'Test description',
            'yc_page': 'https://www.ycombinator.com/companies/test-company',
            'linkedin_url': 'https://www.linkedin.com/company/test-company',
            'yc_s25_on_linkedin': 'true',  # Will be converted
            'last_updated': '2025-01-01T10:00:00'
        }]
        
        # Save dirty data directly to CSV file (bypassing validation)
        df = pd.DataFrame(dirty_data)
        df.to_csv(self.data_manager.csv_file_path, index=False)
        
        success, actions = self.data_manager.repair_corrupted_data()
        
        self.assertTrue(success)
        self.assertGreater(len(actions), 0)
    
    def test_create_manual_backup(self):
        """Test creating manual backup."""
        # Create some data first
        df = pd.DataFrame(self.sample_companies)
        self.data_manager.save_data(df)
        
        backup_path = self.data_manager.create_manual_backup("test_backup")
        
        self.assertTrue(os.path.exists(backup_path))
        self.assertIn("manual_test_backup", backup_path)
        
        # Verify backup contains the data
        backup_df = pd.read_csv(backup_path)
        self.assertEqual(len(backup_df), 2)
    
    def test_create_manual_backup_no_data_file(self):
        """Test creating manual backup when no data file exists."""
        with self.assertRaises(FileNotFoundError):
            self.data_manager.create_manual_backup()
    
    def test_is_valid_url_format(self):
        """Test URL format validation."""
        # Valid URLs
        self.assertTrue(self.data_manager._is_valid_url_format('https://example.com'))
        self.assertTrue(self.data_manager._is_valid_url_format('http://example.com'))
        self.assertTrue(self.data_manager._is_valid_url_format('https://www.example.com/path'))
        
        # Invalid URLs
        self.assertFalse(self.data_manager._is_valid_url_format('invalid-url'))
        self.assertFalse(self.data_manager._is_valid_url_format('ftp://example.com'))
        self.assertFalse(self.data_manager._is_valid_url_format(''))
        self.assertFalse(self.data_manager._is_valid_url_format('example.com'))
    
    def test_is_valid_timestamp_format(self):
        """Test timestamp format validation."""
        # Valid timestamps
        self.assertTrue(self.data_manager._is_valid_timestamp_format('2025-01-01T10:00:00'))
        self.assertTrue(self.data_manager._is_valid_timestamp_format('2025-01-01T10:00:00.123456'))
        self.assertTrue(self.data_manager._is_valid_timestamp_format('2025-01-01T10:00:00Z'))
        
        # Invalid timestamps
        self.assertFalse(self.data_manager._is_valid_timestamp_format('invalid-timestamp'))
        self.assertFalse(self.data_manager._is_valid_timestamp_format('2025-13-01T10:00:00'))
        self.assertFalse(self.data_manager._is_valid_timestamp_format(''))
        self.assertFalse(self.data_manager._is_valid_timestamp_format('2025-01-01'))


if __name__ == '__main__':
    unittest.main()