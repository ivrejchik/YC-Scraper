"""
Unit tests for YC Client functionality.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import requests
import json
from bs4 import BeautifulSoup
from yc_parser.yc_client import YcClient


class TestYcClient(unittest.TestCase):
    """Test cases for YcClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = YcClient()
    
    def test_init(self):
        """Test YcClient initialization."""
        self.assertEqual(self.client.base_url, "https://www.ycombinator.com")
        self.assertEqual(self.client.api_base, "https://yc-oss.github.io/api/batches")
        self.assertEqual(self.client.s25_api_url, "https://yc-oss.github.io/api/batches/summer-2025.json")
        self.assertIsNotNone(self.client.session)
    
    def test_build_yc_url(self):
        """Test YC URL building from slug."""
        # Test normal slug
        url = self.client._build_yc_url("test-company")
        self.assertEqual(url, "https://www.ycombinator.com/companies/test-company")
        
        # Test slug with spaces
        url = self.client._build_yc_url("Test Company")
        self.assertEqual(url, "https://www.ycombinator.com/companies/test-company")
        
        # Test empty slug
        url = self.client._build_yc_url("")
        self.assertEqual(url, "")
    
    def test_normalize_api_data(self):
        """Test API data normalization."""
        raw_companies = [
            {
                'name': 'Test Company 1',
                'website': 'https://test1.com',
                'description': 'Test description 1',
                'slug': 'test-company-1',
                'linkedin_url': 'https://linkedin.com/company/test1',
                'batch': 'S25'
            },
            {
                'name': 'Test Company 2',
                'website': 'https://test2.com',
                'tagline': 'Test tagline 2',  # Using tagline instead of description
                'slug': 'test-company-2',
                'batch': 'S25'
            },
            {
                'name': 'Wrong Batch Company',
                'batch': 'W25'  # Wrong batch, should be filtered out
            }
        ]
        
        normalized = self.client._normalize_api_data(raw_companies)
        
        # Should only have 2 companies (S25 batch only)
        self.assertEqual(len(normalized), 2)
        
        # Check first company
        company1 = normalized[0]
        self.assertEqual(company1['name'], 'Test Company 1')
        self.assertEqual(company1['website'], 'https://test1.com')
        self.assertEqual(company1['description'], 'Test description 1')
        self.assertEqual(company1['yc_page'], 'https://www.ycombinator.com/companies/test-company-1')
        self.assertEqual(company1['linkedin_url'], 'https://linkedin.com/company/test1')
        
        # Check second company (with tagline)
        company2 = normalized[1]
        self.assertEqual(company2['name'], 'Test Company 2')
        self.assertEqual(company2['description'], 'Test tagline 2')
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_s25_companies_success_list_format(self, mock_get):
        """Test successful API fetch with list format response."""
        # Mock successful API response (list format)
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                'name': 'API Company 1',
                'website': 'https://api1.com',
                'description': 'API description 1',
                'slug': 'api-company-1',
                'batch': 'S25'
            }
        ]
        mock_get.return_value = mock_response
        
        companies = self.client.fetch_s25_companies()
        
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0]['name'], 'API Company 1')
        mock_get.assert_called_once()
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_s25_companies_success_dict_format(self, mock_get):
        """Test successful API fetch with dict format response."""
        # Mock successful API response (dict format)
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'companies': [
                {
                    'name': 'API Company 2',
                    'website': 'https://api2.com',
                    'description': 'API description 2',
                    'slug': 'api-company-2',
                    'batch': 'S25'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        companies = self.client.fetch_s25_companies()
        
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0]['name'], 'API Company 2')
    
    @patch('yc_parser.yc_client.YcClient._fallback_scrape_companies')
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_s25_companies_api_failure_fallback(self, mock_get, mock_fallback):
        """Test API failure triggers fallback scraping."""
        # Mock API failure
        mock_get.side_effect = requests.RequestException("API failed")
        
        # Mock fallback response
        mock_fallback.return_value = [
            {
                'name': 'Fallback Company',
                'website': '',
                'description': '',
                'yc_page': 'https://www.ycombinator.com/companies/fallback-company',
                'linkedin_url': '',
                'batch': 'S25'
            }
        ]
        
        companies = self.client.fetch_s25_companies()
        
        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0]['name'], 'Fallback Company')
        mock_fallback.assert_called_once()
    
    @patch('yc_parser.yc_client.YcClient._fallback_scrape_companies')
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_s25_companies_invalid_json_fallback(self, mock_get, mock_fallback):
        """Test invalid JSON response triggers fallback."""
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = "invalid format"  # Not list or dict with companies
        mock_get.return_value = mock_response
        
        mock_fallback.return_value = []
        
        companies = self.client.fetch_s25_companies()
        
        mock_fallback.assert_called_once()
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fallback_scrape_companies(self, mock_get):
        """Test fallback scraping functionality."""
        # Mock HTML response
        html_content = """
        <html>
            <div class="company-card">
                <h3>Scraped Company 1</h3>
                <a href="/companies/scraped-company-1">View Profile</a>
            </div>
            <div class="company-item">
                <h4>Scraped Company 2</h4>
                <a href="/companies/scraped-company-2">View Profile</a>
            </div>
        </html>
        """
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = html_content.encode('utf-8')
        mock_get.return_value = mock_response
        
        companies = self.client._fallback_scrape_companies()
        
        self.assertEqual(len(companies), 2)
        self.assertEqual(companies[0]['name'], 'Scraped Company 1')
        self.assertEqual(companies[0]['yc_page'], 'https://www.ycombinator.com/companies/scraped-company-1')
        self.assertEqual(companies[1]['name'], 'Scraped Company 2')
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fallback_scrape_companies_failure(self, mock_get):
        """Test fallback scraping handles failures gracefully."""
        # Mock scraping failure
        mock_get.side_effect = requests.RequestException("Scraping failed")
        
        companies = self.client._fallback_scrape_companies()
        
        self.assertEqual(companies, [])
    
    def test_create_session(self):
        """Test session creation with proper configuration."""
        session = self.client._create_session()
        
        # Check headers
        self.assertIn('User-Agent', session.headers)
        self.assertTrue(session.headers['User-Agent'].startswith('Mozilla'))
        
        # Check adapters are mounted
        self.assertIn('http://', session.adapters)
        self.assertIn('https://', session.adapters)


    @patch('yc_parser.yc_client.requests.Session.get')
    def test_parse_company_profile_success(self, mock_get):
        """Test successful company profile parsing."""
        html_content = """
        <html>
            <head>
                <meta name="description" content="Test company tagline from meta">
            </head>
            <body>
                <h2 class="tagline">Test Company Tagline</h2>
                <a href="https://testcompany.com" class="website-link">Website</a>
                <a href="https://linkedin.com/company/test-company">LinkedIn</a>
                <div class="company-description">This is a detailed description of the test company.</div>
            </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = html_content.encode('utf-8')
        mock_get.return_value = mock_response
        
        profile_data = self.client.parse_company_profile("https://www.ycombinator.com/companies/test")
        
        self.assertEqual(profile_data['tagline'], 'Test Company Tagline')
        self.assertEqual(profile_data['website'], 'https://testcompany.com')
        self.assertEqual(profile_data['linkedin_url'], 'https://linkedin.com/company/test-company')
        self.assertEqual(profile_data['description'], 'This is a detailed description of the test company.')
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_parse_company_profile_empty_url(self, mock_get):
        """Test profile parsing with empty URL."""
        profile_data = self.client.parse_company_profile("")
        self.assertEqual(profile_data, {})
        mock_get.assert_not_called()
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_parse_company_profile_request_failure(self, mock_get):
        """Test profile parsing handles request failures."""
        mock_get.side_effect = requests.RequestException("Request failed")
        
        profile_data = self.client.parse_company_profile("https://www.ycombinator.com/companies/test")
        self.assertEqual(profile_data, {})
    
    def test_extract_tagline(self):
        """Test tagline extraction from various HTML structures."""
        # Test with h2.tagline
        html1 = '<h2 class="tagline">Main Tagline</h2>'
        soup1 = BeautifulSoup(html1, 'html.parser')
        tagline1 = self.client._extract_tagline(soup1)
        self.assertEqual(tagline1, 'Main Tagline')
        
        # Test with meta description fallback
        html2 = '<meta name="description" content="Meta tagline">'
        soup2 = BeautifulSoup(html2, 'html.parser')
        tagline2 = self.client._extract_tagline(soup2)
        self.assertEqual(tagline2, 'Meta tagline')
        
        # Test with no tagline
        html3 = '<div>No tagline here</div>'
        soup3 = BeautifulSoup(html3, 'html.parser')
        tagline3 = self.client._extract_tagline(soup3)
        self.assertEqual(tagline3, '')
    
    def test_extract_website(self):
        """Test website URL extraction."""
        # Test with website link
        html1 = '<a href="https://example.com" class="website-link">Website</a>'
        soup1 = BeautifulSoup(html1, 'html.parser')
        website1 = self.client._extract_website(soup1)
        self.assertEqual(website1, 'https://example.com')
        
        # Test filtering out YC links
        html2 = '<a href="https://www.ycombinator.com/companies/test">YC Link</a><a href="https://realsite.com">Real Site</a>'
        soup2 = BeautifulSoup(html2, 'html.parser')
        website2 = self.client._extract_website(soup2)
        self.assertEqual(website2, 'https://realsite.com')
        
        # Test with no website
        html3 = '<a href="https://linkedin.com/company/test">LinkedIn</a>'
        soup3 = BeautifulSoup(html3, 'html.parser')
        website3 = self.client._extract_website(soup3)
        self.assertEqual(website3, '')
    
    def test_extract_linkedin_url(self):
        """Test LinkedIn URL extraction."""
        # Test with LinkedIn company URL
        html1 = '<a href="https://linkedin.com/company/test-company">LinkedIn</a>'
        soup1 = BeautifulSoup(html1, 'html.parser')
        linkedin1 = self.client._extract_linkedin_url(soup1)
        self.assertEqual(linkedin1, 'https://linkedin.com/company/test-company')
        
        # Test with no LinkedIn URL
        html2 = '<a href="https://twitter.com/test">Twitter</a>'
        soup2 = BeautifulSoup(html2, 'html.parser')
        linkedin2 = self.client._extract_linkedin_url(soup2)
        self.assertEqual(linkedin2, '')
    
    def test_extract_description(self):
        """Test description extraction."""
        # Test with company description class
        html1 = '<div class="company-description">This is the company description.</div>'
        soup1 = BeautifulSoup(html1, 'html.parser')
        desc1 = self.client._extract_description(soup1)
        self.assertEqual(desc1, 'This is the company description.')
        
        # Test with paragraph fallback
        html2 = '<p>This is a substantial paragraph that describes the company and what it does in detail.</p>'
        soup2 = BeautifulSoup(html2, 'html.parser')
        desc2 = self.client._extract_description(soup2)
        self.assertEqual(desc2, 'This is a substantial paragraph that describes the company and what it does in detail.')
        
        # Test filtering out short or navigation text
        html3 = '<p>Login</p><p>Sign up</p><p>Short</p>'
        soup3 = BeautifulSoup(html3, 'html.parser')
        desc3 = self.client._extract_description(soup3)
        self.assertEqual(desc3, '')
    
    @patch('yc_parser.yc_client.YcClient.parse_company_profile')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_enrich_company_data(self, mock_sleep, mock_parse):
        """Test company data enrichment."""
        # Mock profile parsing responses
        mock_parse.side_effect = [
            {'tagline': 'Enriched tagline 1', 'website': 'https://enriched1.com'},
            {'linkedin_url': 'https://linkedin.com/company/enriched2'}
        ]
        
        companies = [
            {
                'name': 'Company 1',
                'yc_page': 'https://www.ycombinator.com/companies/company1',
                'website': '',  # Will be enriched
                'description': 'Existing description'
            },
            {
                'name': 'Company 2',
                'yc_page': 'https://www.ycombinator.com/companies/company2',
                'linkedin_url': '',  # Will be enriched
                'description': ''
            }
        ]
        
        enriched = self.client.enrich_company_data(companies)
        
        self.assertEqual(len(enriched), 2)
        
        # Check first company enrichment
        self.assertEqual(enriched[0]['name'], 'Company 1')
        self.assertEqual(enriched[0]['website'], 'https://enriched1.com')  # Enriched
        self.assertEqual(enriched[0]['description'], 'Existing description')  # Preserved
        
        # Check second company enrichment
        self.assertEqual(enriched[1]['name'], 'Company 2')
        self.assertEqual(enriched[1]['linkedin_url'], 'https://linkedin.com/company/enriched2')  # Enriched
        
        # Verify parse_company_profile was called for each company
        self.assertEqual(mock_parse.call_count, 2)
    
    @patch('yc_parser.yc_client.YcClient.parse_company_profile')
    @patch('time.sleep')
    def test_enrich_company_data_handles_errors(self, mock_sleep, mock_parse):
        """Test enrichment handles individual company errors gracefully."""
        # Mock one successful parse and one failure
        mock_parse.side_effect = [
            {'website': 'https://success.com'},
            Exception("Parse failed")
        ]
        
        companies = [
            {
                'name': 'Success Company',
                'yc_page': 'https://www.ycombinator.com/companies/success',
                'website': ''
            },
            {
                'name': 'Fail Company',
                'yc_page': 'https://www.ycombinator.com/companies/fail',
                'website': 'https://existing.com'
            }
        ]
        
        enriched = self.client.enrich_company_data(companies)
        
        # Should still return both companies
        self.assertEqual(len(enriched), 2)
        self.assertEqual(enriched[0]['website'], 'https://success.com')  # Enriched
        self.assertEqual(enriched[1]['website'], 'https://existing.com')  # Original data preserved
    
    @patch('yc_parser.yc_client.YcClient.enrich_company_data')
    @patch('yc_parser.yc_client.YcClient.fetch_s25_companies')
    def test_get_complete_s25_data_success(self, mock_fetch, mock_enrich):
        """Test successful complete data collection."""
        # Mock API response
        mock_fetch.return_value = [
            {
                'name': 'Test Company',
                'website': 'testcompany.com',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'batch': 'S25'
            }
        ]
        
        # Mock enrichment response
        mock_enrich.return_value = [
            {
                'name': 'Test Company',
                'website': 'testcompany.com',
                'description': 'Enriched description',
                'yc_page': 'https://www.ycombinator.com/companies/test-company',
                'linkedin_url': 'https://linkedin.com/company/test-company',
                'batch': 'S25'
            }
        ]
        
        complete_data = self.client.get_complete_s25_data()
        
        self.assertEqual(len(complete_data), 1)
        company = complete_data[0]
        self.assertEqual(company['name'], 'Test Company')
        self.assertEqual(company['website'], 'https://testcompany.com')  # URL cleaned
        self.assertEqual(company['description'], 'Enriched description')
        
        mock_fetch.assert_called_once()
        mock_enrich.assert_called_once()
    
    @patch('yc_parser.yc_client.YcClient.fetch_s25_companies')
    def test_get_complete_s25_data_no_companies(self, mock_fetch):
        """Test complete data collection when no companies found."""
        mock_fetch.return_value = []
        
        complete_data = self.client.get_complete_s25_data()
        
        self.assertEqual(complete_data, [])
    
    @patch('yc_parser.yc_client.YcClient.fetch_s25_companies')
    def test_get_complete_s25_data_fetch_failure(self, mock_fetch):
        """Test complete data collection handles fetch failures."""
        mock_fetch.side_effect = Exception("Fetch failed")
        
        with self.assertRaises(Exception):
            self.client.get_complete_s25_data()
    
    def test_validate_and_clean_data(self):
        """Test data validation and cleaning."""
        raw_companies = [
            {
                'name': 'Valid Company',
                'website': 'validcompany.com',  # Missing protocol
                'description': 'Valid description',
                'yc_page': 'https://www.ycombinator.com/companies/valid',
                'linkedin_url': 'https://linkedin.com/company/valid'
            },
            {
                'name': '',  # Invalid - no name
                'website': 'https://invalid.com'
            },
            {
                'name': 'Company with Tagline',
                'website': '',
                'description': '',  # Empty description
                'tagline': 'This is the tagline',  # Should be used as description
                'yc_page': 'https://www.ycombinator.com/companies/tagline'
            }
        ]
        
        cleaned = self.client._validate_and_clean_data(raw_companies)
        
        # Should have 2 valid companies (invalid one filtered out)
        self.assertEqual(len(cleaned), 2)
        
        # Check first company
        company1 = cleaned[0]
        self.assertEqual(company1['name'], 'Valid Company')
        self.assertEqual(company1['website'], 'https://validcompany.com')  # Protocol added
        self.assertEqual(company1['description'], 'Valid description')
        
        # Check second company (tagline used as description)
        company2 = cleaned[1]
        self.assertEqual(company2['name'], 'Company with Tagline')
        self.assertEqual(company2['description'], 'This is the tagline')
    
    def test_clean_url(self):
        """Test URL cleaning functionality."""
        # Test adding protocol
        self.assertEqual(self.client._clean_url('example.com'), 'https://example.com')
        
        # Test preserving existing protocol
        self.assertEqual(self.client._clean_url('http://example.com'), 'http://example.com')
        self.assertEqual(self.client._clean_url('https://example.com'), 'https://example.com')
        
        # Test empty URL
        self.assertEqual(self.client._clean_url(''), '')
        self.assertEqual(self.client._clean_url(None), '')
        
        # Test whitespace handling
        self.assertEqual(self.client._clean_url('  example.com  '), 'https://example.com')

    # Task 11: YC S25 Batch Data Retrieval and Validation Tests
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_success(self, mock_get):
        """Test successful YC S25 batch data retrieval from official API endpoint."""
        # Mock successful API response with valid S25 companies using actual YC API format
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'[{"name": "Test Company"}]'  # Non-empty content
        mock_response.json.return_value = [
            {
                'id': 1001,
                'name': 'YC S25 Company 1',
                'slug': 'yc-s25-company-1',
                'website': 'https://ycs25company1.com',
                'one_liner': 'First YC S25 company one-liner',
                'long_description': 'Detailed description of the first YC S25 company',
                'batch': 'S25',
                'team_size': 5,
                'all_locations': 'San Francisco, CA, USA',
                'industry': 'B2B',
                'subindustry': 'B2B -> Engineering, Product and Design',
                'launched_at': 1640995200,
                'tags': ['SaaS', 'Enterprise'],
                'status': 'Active',
                'stage': 'Early',
                'top_company': False,
                'small_logo_thumb_url': 'https://example.com/logo1.png'
            },
            {
                'id': 1002,
                'name': 'YC S25 Company 2',
                'slug': 'yc-s25-company-2',
                'website': 'ycs25company2.com',  # Missing protocol
                'one_liner': 'Second YC S25 company tagline',
                'batch': 'S25',
                'team_size': 3,
                'all_locations': 'New York, NY, USA',
                'industry': 'Consumer',
                'tags': ['Mobile', 'Social'],
                'status': 'Active'
            }
        ]
        mock_get.return_value = mock_response
        
        companies = self.client.fetch_yc_s25_batch_data()
        
        # Verify API was called with correct endpoint
        mock_get.assert_called_once_with(self.client.s25_api_url, timeout=10)
        
        # Verify response structure
        self.assertEqual(len(companies), 2)
        
        # Check first company validation and extraction
        company1 = companies[0]
        self.assertEqual(company1['name'], 'YC S25 Company 1')
        self.assertEqual(company1['website'], 'https://ycs25company1.com')
        self.assertEqual(company1['description'], 'First YC S25 company one-liner')  # one_liner used as description
        self.assertEqual(company1['yc_page'], 'https://www.ycombinator.com/companies/yc-s25-company-1')
        self.assertEqual(company1['linkedin_url'], '')  # Not provided in YC API
        self.assertEqual(company1['batch'], 'S25')
        
        # Check metadata extraction
        self.assertIn('metadata', company1)
        self.assertEqual(company1['metadata']['id'], 1001)
        self.assertEqual(company1['metadata']['slug'], 'yc-s25-company-1')
        self.assertEqual(company1['metadata']['one_liner'], 'First YC S25 company one-liner')
        self.assertEqual(company1['metadata']['long_description'], 'Detailed description of the first YC S25 company')
        self.assertEqual(company1['metadata']['team_size'], 5)
        self.assertEqual(company1['metadata']['all_locations'], 'San Francisco, CA, USA')
        self.assertEqual(company1['metadata']['industry'], 'B2B')
        self.assertEqual(company1['metadata']['tags'], ['SaaS', 'Enterprise'])
        self.assertEqual(company1['metadata']['status'], 'Active')
        self.assertEqual(company1['metadata']['stage'], 'Early')
        self.assertEqual(company1['metadata']['top_company'], False)
        
        # Check second company
        company2 = companies[1]
        self.assertEqual(company2['name'], 'YC S25 Company 2')
        self.assertEqual(company2['description'], 'Second YC S25 company tagline')
        self.assertEqual(company2['website'], 'ycs25company2.com')
        self.assertEqual(company2['batch'], 'S25')
        
        # Check metadata for second company
        self.assertEqual(company2['metadata']['id'], 1002)
        self.assertEqual(company2['metadata']['slug'], 'yc-s25-company-2')
        self.assertEqual(company2['metadata']['team_size'], 3)
        self.assertEqual(company2['metadata']['all_locations'], 'New York, NY, USA')
        self.assertEqual(company2['metadata']['tags'], ['Mobile', 'Social'])
        # long_description not present, should not be in metadata
        self.assertNotIn('long_description', company2['metadata'])
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_empty_response(self, mock_get):
        """Test handling of empty response from YC S25 API."""
        # Mock empty response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b''  # Empty content
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            self.client.fetch_yc_s25_batch_data()
        
        self.assertIn("Empty response from YC S25 API", str(context.exception))
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_invalid_json(self, mock_get):
        """Test handling of malformed JSON response."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'{"invalid": json syntax'
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            self.client.fetch_yc_s25_batch_data()
        
        self.assertIn("Malformed JSON response", str(context.exception))
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_non_list_response(self, mock_get):
        """Test handling of non-list JSON response structure."""
        # Mock response with dict instead of list
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'{"companies": []}'
        mock_response.json.return_value = {"companies": []}
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            self.client.fetch_yc_s25_batch_data()
        
        self.assertIn("Expected list response, got", str(context.exception))
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_empty_list(self, mock_get):
        """Test handling of empty companies list."""
        # Mock response with empty list
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'[]'
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            self.client.fetch_yc_s25_batch_data()
        
        self.assertIn("API returned empty companies list", str(context.exception))
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_timeout_error(self, mock_get):
        """Test handling of network timeout errors."""
        # Mock timeout error
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with self.assertRaises(requests.RequestException) as context:
            self.client.fetch_yc_s25_batch_data()
        
        self.assertIn("API request timeout", str(context.exception))
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_connection_error(self, mock_get):
        """Test handling of network connection errors."""
        # Mock connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with self.assertRaises(requests.RequestException) as context:
            self.client.fetch_yc_s25_batch_data()
        
        self.assertIn("API connection failed", str(context.exception))
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError("Not found")
        http_error.response = mock_response
        mock_get.side_effect = http_error
        
        with self.assertRaises(requests.RequestException) as context:
            self.client.fetch_yc_s25_batch_data()
        
        self.assertIn("API HTTP error 404", str(context.exception))
    
    def test_validate_s25_company_data_success(self):
        """Test successful validation of S25 company data."""
        raw_companies = [
            {
                'id': 1001,
                'name': 'Valid Company 1',
                'slug': 'valid-company-1',
                'website': 'https://valid1.com',
                'one_liner': 'Valid one-liner 1',
                'long_description': 'Valid long description 1',
                'batch': 'S25',
                'team_size': 10,
                'all_locations': 'New York, NY, USA',
                'industry': 'B2B',
                'tags': ['SaaS'],
                'status': 'Active'
            },
            {
                'id': 1002,
                'name': 'Valid Company 2',
                'slug': 'valid-company-2',
                'website': 'valid2.com',
                'one_liner': 'Valid one-liner 2',  # Should be used as description
                'batch': 'S25',
                'team_size': 5,
                'all_locations': 'San Francisco, CA, USA'
            }
        ]
        
        validated = self.client._validate_s25_company_data(raw_companies)
        
        self.assertEqual(len(validated), 2)
        
        # Check first company
        company1 = validated[0]
        self.assertEqual(company1['name'], 'Valid Company 1')
        self.assertEqual(company1['website'], 'https://valid1.com')
        self.assertEqual(company1['description'], 'Valid one-liner 1')  # one_liner used as description
        self.assertEqual(company1['yc_page'], 'https://www.ycombinator.com/companies/valid-company-1')
        self.assertEqual(company1['batch'], 'S25')
        
        # Check metadata
        self.assertEqual(company1['metadata']['id'], 1001)
        self.assertEqual(company1['metadata']['slug'], 'valid-company-1')
        self.assertEqual(company1['metadata']['one_liner'], 'Valid one-liner 1')
        self.assertEqual(company1['metadata']['long_description'], 'Valid long description 1')
        self.assertEqual(company1['metadata']['team_size'], 10)
        self.assertEqual(company1['metadata']['all_locations'], 'New York, NY, USA')
        self.assertEqual(company1['metadata']['industry'], 'B2B')
        self.assertEqual(company1['metadata']['tags'], ['SaaS'])
        self.assertEqual(company1['metadata']['status'], 'Active')
        
        # Check second company (one_liner as description)
        company2 = validated[1]
        self.assertEqual(company2['name'], 'Valid Company 2')
        self.assertEqual(company2['description'], 'Valid one-liner 2')
        self.assertEqual(company2['website'], 'valid2.com')
        self.assertEqual(company2['batch'], 'S25')
        self.assertEqual(company2['metadata']['id'], 1002)
        self.assertEqual(company2['metadata']['team_size'], 5)
        self.assertEqual(company2['metadata']['all_locations'], 'San Francisco, CA, USA')
        # long_description not present, should not be in metadata
        self.assertNotIn('long_description', company2['metadata'])
    
    def test_validate_s25_company_data_missing_name(self):
        """Test validation handles companies with missing names."""
        raw_companies = [
            {
                'id': 1001,
                'name': 'Valid Company',
                'slug': 'valid-company',
                'website': 'https://valid.com',
                'batch': 'S25'
            },
            {
                'id': 1002,
                'name': '',  # Empty name
                'website': 'https://invalid.com',
                'batch': 'S25'
            },
            {
                'id': 1003,
                'website': 'https://noname.com',  # No name field
                'batch': 'S25'
            }
        ]
        
        validated = self.client._validate_s25_company_data(raw_companies)
        
        # Should only have 1 valid company
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]['name'], 'Valid Company')
    
    def test_validate_s25_company_data_non_dict_entries(self):
        """Test validation handles non-dictionary entries."""
        raw_companies = [
            {
                'id': 1001,
                'name': 'Valid Company',
                'slug': 'valid-company',
                'website': 'https://valid.com',
                'batch': 'S25'
            },
            "invalid string entry",
            123,  # Invalid number entry
            None  # Invalid None entry
        ]
        
        validated = self.client._validate_s25_company_data(raw_companies)
        
        # Should only have 1 valid company
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]['name'], 'Valid Company')
    
    def test_validate_s25_company_data_all_invalid(self):
        """Test validation when all companies are invalid."""
        raw_companies = [
            {'name': '', 'batch': 'S25'},  # Empty name
            {'website': 'https://noname.com', 'batch': 'S25'},  # No name
            "invalid entry"  # Not a dict
        ]
        
        with self.assertRaises(ValueError) as context:
            self.client._validate_s25_company_data(raw_companies)
        
        self.assertIn("All companies failed validation", str(context.exception))
    
    def test_validate_s25_company_data_batch_filtering(self):
        """Test that only S25 batch companies are included."""
        raw_companies = [
            {
                'id': 1001,
                'name': 'S25 Company',
                'slug': 's25-company',
                'batch': 'S25',  # Should be included
                'website': 'https://s25.com'
            },
            {
                'id': 1002,
                'name': 'Summer 2025 Company',
                'slug': 'summer-2025-company',
                'batch': 'Summer 2025',  # Should be included (actual API format)
                'website': 'https://summer2025.com'
            },
            {
                'id': 1003,
                'name': 'W25 Company',
                'slug': 'w25-company',
                'batch': 'W25',  # Should be filtered out
                'website': 'https://w25.com'
            },
            {
                'id': 1004,
                'name': 'F24 Company',
                'slug': 'f24-company',
                'batch': 'F24',  # Should be filtered out
                'website': 'https://f24.com'
            }
        ]
        
        validated = self.client._validate_s25_company_data(raw_companies)
        
        # Should have 2 companies (S25 and Summer 2025 batches)
        self.assertEqual(len(validated), 2)
        self.assertEqual(validated[0]['name'], 'S25 Company')
        self.assertEqual(validated[0]['batch'], 'S25')  # Normalized to S25
        self.assertEqual(validated[1]['name'], 'Summer 2025 Company')
        self.assertEqual(validated[1]['batch'], 'S25')  # Normalized to S25
    
    def test_validate_s25_company_data_metadata_cleaning(self):
        """Test that metadata is properly cleaned of empty values."""
        raw_companies = [
            {
                'id': 1001,
                'name': 'Test Company',
                'slug': 'test-company',
                'one_liner': 'Test one-liner',
                'batch': 'S25',
                'launched_at': 1640995200,
                'team_size': None,  # Should be filtered out
                'all_locations': '',  # Should be filtered out
                'extra_field': 'should be ignored'  # Not in expected metadata fields
            }
        ]
        
        validated = self.client._validate_s25_company_data(raw_companies)
        
        company = validated[0]
        metadata = company['metadata']
        
        # Should have non-empty values
        self.assertEqual(metadata['id'], 1001)
        self.assertEqual(metadata['slug'], 'test-company')
        self.assertEqual(metadata['one_liner'], 'Test one-liner')
        self.assertEqual(metadata['launched_at'], 1640995200)
        
        # Should not have empty/None values
        self.assertNotIn('team_size', metadata)
        self.assertNotIn('all_locations', metadata)
        # extra_field should not be included as it's not in the expected metadata fields
    
    @patch('yc_parser.yc_client.requests.Session.get')
    def test_fetch_yc_s25_batch_data_validation_failure(self, mock_get):
        """Test handling when validation fails for all companies."""
        # Mock response with invalid company data
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'[{"invalid": "data"}]'
        mock_response.json.return_value = [
            {'invalid': 'data'},  # No name field
            {'name': ''}  # Empty name
        ]
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            self.client.fetch_yc_s25_batch_data()
        
        self.assertIn("All companies failed validation", str(context.exception))


class TestYcClientIntegration(unittest.TestCase):
    """Integration tests for YcClient complete workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.client = YcClient()
    
    @patch('yc_parser.yc_client.requests.Session.get')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_complete_workflow_integration(self, mock_sleep, mock_get):
        """Test complete workflow from API to enriched data."""
        # Mock API response
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = [
            {
                'name': 'Integration Test Company',
                'website': 'integration.com',
                'slug': 'integration-test-company',
                'batch': 'S25'
            }
        ]
        
        # Mock profile page response
        profile_html = """
        <html>
            <h2 class="tagline">Integration test tagline</h2>
            <a href="https://integration.com" class="website-link">Website</a>
            <a href="https://linkedin.com/company/integration-test">LinkedIn</a>
            <div class="company-description">Integration test description</div>
        </html>
        """
        profile_response = Mock()
        profile_response.raise_for_status.return_value = None
        profile_response.content = profile_html.encode('utf-8')
        
        # Configure mock to return different responses for different URLs
        def mock_get_side_effect(url, **kwargs):
            if 'summer-2025.json' in url:
                return api_response
            else:
                return profile_response
        
        mock_get.side_effect = mock_get_side_effect
        
        # Run complete workflow
        complete_data = self.client.get_complete_s25_data()
        
        # Verify results
        self.assertEqual(len(complete_data), 1)
        company = complete_data[0]
        
        self.assertEqual(company['name'], 'Integration Test Company')
        self.assertEqual(company['website'], 'https://integration.com')
        self.assertEqual(company['description'], 'Integration test description')
        self.assertEqual(company['linkedin_url'], 'https://linkedin.com/company/integration-test')
        self.assertEqual(company['yc_page'], 'https://www.ycombinator.com/companies/integration-test-company')
        
        # Verify both API and profile requests were made
        self.assertEqual(mock_get.call_count, 2)


if __name__ == '__main__':
    unittest.main()