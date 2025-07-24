"""
Unit tests for LinkedIn scraper module.
"""
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from yc_parser.linkedin_scraper import LinkedInScraper


class TestLinkedInScraper:
    """Test cases for LinkedInScraper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scraper = LinkedInScraper(delay_range=(0.1, 0.2), max_retries=2)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.scraper.close()
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    @patch('requests.Session.get')
    def test_make_request_success(self, mock_get, mock_sleep):
        """Test successful HTTP request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html>Test content</html>"
        mock_get.return_value = mock_response
        
        result = self.scraper._make_request("https://linkedin.com/company/test")
        
        assert result == "<html>Test content</html>"
        mock_get.assert_called_once()
        assert mock_sleep.call_count >= 1  # Rate limiting sleep
    
    @patch('time.sleep')
    @patch('requests.Session.get')
    def test_make_request_empty_url(self, mock_get, mock_sleep):
        """Test handling of empty URL."""
        result = self.scraper._make_request("")
        assert result is None
        mock_get.assert_not_called()
        
        result = self.scraper._make_request(None)
        assert result is None
        mock_get.assert_not_called()
    
    @patch('time.sleep')
    @patch('requests.Session.get')
    def test_make_request_url_normalization(self, mock_get, mock_sleep):
        """Test URL normalization (adding https://)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "content"
        mock_get.return_value = mock_response
        
        self.scraper._make_request("linkedin.com/company/test")
        
        # Should add https:// prefix
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://linkedin.com/company/test"
    
    @patch('time.sleep')
    @patch('requests.Session.get')
    def test_make_request_rate_limit_429(self, mock_get, mock_sleep):
        """Test handling of 429 rate limit response."""
        # First call returns 429, second call succeeds
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {'Retry-After': '1'}
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.text = "success content"
        
        mock_get.side_effect = [mock_response_429, mock_response_success]
        
        result = self.scraper._make_request("https://linkedin.com/company/test")
        
        assert result == "success content"
        assert mock_get.call_count == 2
        # Should sleep for rate limit + random delays
        assert mock_sleep.call_count >= 2
    
    @patch('time.sleep')
    @patch('requests.Session.get')
    def test_make_request_403_404_errors(self, mock_get, mock_sleep):
        """Test handling of 403/404 errors (should not retry)."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        
        result = self.scraper._make_request("https://linkedin.com/company/test")
        
        assert result is None
        mock_get.assert_called_once()  # Should not retry for 403/404
    
    @patch('time.sleep')
    @patch('requests.Session.get')
    def test_make_request_timeout_retry(self, mock_get, mock_sleep):
        """Test retry logic for timeout errors."""
        # First call times out, second call succeeds
        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            Mock(status_code=200, text="success")
        ]
        
        result = self.scraper._make_request("https://linkedin.com/company/test")
        
        assert result == "success"
        assert mock_get.call_count == 2
    
    @patch('time.sleep')
    @patch('requests.Session.get')
    def test_make_request_max_retries_exceeded(self, mock_get, mock_sleep):
        """Test behavior when max retries are exceeded."""
        # All calls fail
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = self.scraper._make_request("https://linkedin.com/company/test")
        
        assert result is None
        assert mock_get.call_count == 2  # max_retries = 2
    
    @patch('time.sleep')
    @patch('requests.Session.get')
    def test_make_request_connection_error(self, mock_get, mock_sleep):
        """Test handling of connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        result = self.scraper._make_request("https://linkedin.com/company/test")
        
        assert result is None
        assert mock_get.call_count == 2  # Should retry
    
    def test_check_yc_mention_empty_url(self):
        """Test YC mention check with empty URL."""
        assert self.scraper.check_yc_mention("") is False
        assert self.scraper.check_yc_mention(None) is False
        assert self.scraper.check_yc_mention("   ") is False
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_found(self, mock_make_request):
        """Test YC mention detection when phrase is found."""
        mock_make_request.return_value = "<html><body>We are part of YC S25 batch!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
        mock_make_request.assert_called_once_with("https://linkedin.com/company/test")
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_found_case_insensitive(self, mock_make_request):
        """Test YC mention detection is case insensitive."""
        mock_make_request.return_value = "<html><body>We are part of yc s25 batch!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_html_entities(self, mock_make_request):
        """Test YC mention detection with HTML entities."""
        mock_make_request.return_value = "<html><body>We are part of YC&nbsp;S25 batch!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_multiple_spaces(self, mock_make_request):
        """Test YC mention detection with multiple spaces."""
        mock_make_request.return_value = "<html><body>We are part of YC  S25 batch!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_with_newlines(self, mock_make_request):
        """Test YC mention detection with newlines."""
        mock_make_request.return_value = "<html><body>We are part of YC\nS25 batch!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_full_name(self, mock_make_request):
        """Test YC mention detection with full Y Combinator name."""
        mock_make_request.return_value = "<html><body>We are part of Y Combinator S25 batch!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_summer_format(self, mock_make_request):
        """Test YC mention detection with summer format."""
        mock_make_request.return_value = "<html><body>We are part of YC Summer 2025!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_batch_format(self, mock_make_request):
        """Test YC mention detection with batch format."""
        mock_make_request.return_value = "<html><body>We are part of YC batch S25!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_complex_html(self, mock_make_request):
        """Test YC mention detection in complex HTML structure."""
        html_content = """
        <html>
        <head><title>Company Page</title></head>
        <body>
            <div class="header">
                <h1>Our Company</h1>
            </div>
            <div class="content">
                <p>We are excited to announce that we are part of the 
                <strong>YC S25</strong> cohort at Y Combinator!</p>
                <p>This is a great opportunity for our startup.</p>
            </div>
        </body>
        </html>
        """
        mock_make_request.return_value = html_content
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_html_decoded(self, mock_make_request):
        """Test YC mention detection with HTML entities that need decoding."""
        mock_make_request.return_value = "<html><body>We are part of Y&nbsp;Combinator&nbsp;S25!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is True
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_false_positives(self, mock_make_request):
        """Test that similar but different phrases don't trigger false positives."""
        false_positive_cases = [
            "<html><body>We are part of YC S24 batch!</body></html>",  # Wrong batch
            "<html><body>We are part of YC W25 batch!</body></html>",  # Wrong season
            "<html><body>We love YC but not in S25!</body></html>",    # YC mentioned but not S25
            "<html><body>S25 is a great highway in YC area!</body></html>",  # Different context
        ]
        
        for html_content in false_positive_cases:
            mock_make_request.return_value = html_content
            result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
            assert result is False, f"False positive for: {html_content}"
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_not_found(self, mock_make_request):
        """Test YC mention detection when phrase is not found."""
        mock_make_request.return_value = "<html><body>We are a great startup!</body></html>"
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is False
    
    @patch.object(LinkedInScraper, '_make_request')
    def test_check_yc_mention_request_failed(self, mock_make_request):
        """Test YC mention check when request fails."""
        mock_make_request.return_value = None
        
        result = self.scraper.check_yc_mention("https://linkedin.com/company/test")
        
        assert result is False
    
    @patch.object(LinkedInScraper, 'check_yc_mention')
    def test_batch_check_mentions_success(self, mock_check_yc):
        """Test batch processing of companies."""
        # Mock the check_yc_mention method
        mock_check_yc.side_effect = [True, False, True]
        
        companies = [
            {'name': 'Company A', 'linkedin_url': 'https://linkedin.com/company/a'},
            {'name': 'Company B', 'linkedin_url': 'https://linkedin.com/company/b'},
            {'name': 'Company C', 'linkedin_url': 'https://linkedin.com/company/c'},
        ]
        
        results, errors = self.scraper.batch_check_mentions(companies, max_workers=1)
        
        assert len(results) == 3
        assert len(errors) == 0
        assert results[0]['yc_s25_on_linkedin'] is True
        assert results[1]['yc_s25_on_linkedin'] is False
        assert results[2]['yc_s25_on_linkedin'] is True
        
        # Original company data should be preserved
        assert results[0]['name'] == 'Company A'
        assert results[1]['name'] == 'Company B'
        assert results[2]['name'] == 'Company C'
    
    @patch.object(LinkedInScraper, 'check_yc_mention')
    def test_batch_check_mentions_with_errors(self, mock_check_yc):
        """Test batch processing handles individual errors gracefully."""
        # Second company will raise an exception
        mock_check_yc.side_effect = [True, Exception("Network error"), False]
        
        companies = [
            {'name': 'Company A', 'linkedin_url': 'https://linkedin.com/company/a'},
            {'name': 'Company B', 'linkedin_url': 'https://linkedin.com/company/b'},
            {'name': 'Company C', 'linkedin_url': 'https://linkedin.com/company/c'},
        ]
        
        results, errors = self.scraper.batch_check_mentions(companies, max_workers=1)
        
        assert len(results) == 3
        assert len(errors) == 1  # One error should be recorded
        assert results[0]['yc_s25_on_linkedin'] is True
        assert results[1]['yc_s25_on_linkedin'] is False  # Error case defaults to False
        assert results[2]['yc_s25_on_linkedin'] is False
        assert "Network error" in errors[0]
    
    def test_batch_check_mentions_empty_list(self):
        """Test batch processing with empty company list."""
        results, errors = self.scraper.batch_check_mentions([])
        assert results == []
        assert errors == []
    
    @patch.object(LinkedInScraper, 'check_yc_mention')
    def test_batch_check_mentions_progress_callback(self, mock_check_yc):
        """Test batch processing with progress callback."""
        mock_check_yc.side_effect = [True, False, True]
        
        progress_calls = []
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        companies = [
            {'name': 'Company A', 'linkedin_url': 'https://linkedin.com/company/a'},
            {'name': 'Company B', 'linkedin_url': 'https://linkedin.com/company/b'},
            {'name': 'Company C', 'linkedin_url': 'https://linkedin.com/company/c'},
        ]
        
        results, errors = self.scraper.batch_check_mentions(
            companies, max_workers=1, progress_callback=progress_callback
        )
        
        assert len(results) == 3
        assert len(errors) == 0
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)  # Final progress should be (3, 3)
    
    @patch.object(LinkedInScraper, 'check_yc_mention')
    def test_batch_check_mentions_progress_callback_error(self, mock_check_yc):
        """Test batch processing handles progress callback errors gracefully."""
        mock_check_yc.return_value = True
        
        def failing_callback(completed, total):
            raise Exception("Callback error")
        
        companies = [{'name': 'Company A', 'linkedin_url': 'https://linkedin.com/company/a'}]
        
        # Should not raise exception even if callback fails
        results, errors = self.scraper.batch_check_mentions(
            companies, max_workers=1, progress_callback=failing_callback
        )
        
        assert len(results) == 1
        assert results[0]['yc_s25_on_linkedin'] is True
    
    @patch.object(LinkedInScraper, 'check_yc_mention')
    def test_batch_check_mentions_critical_error(self, mock_check_yc):
        """Test batch processing handles critical errors gracefully."""
        mock_check_yc.return_value = True
        
        companies = [
            {'name': 'Company A', 'linkedin_url': 'https://linkedin.com/company/a'},
            {'name': 'Company B', 'linkedin_url': 'https://linkedin.com/company/b'},
        ]
        
        # Mock ThreadPoolExecutor to raise an exception
        with patch('yc_parser.linkedin_scraper.ThreadPoolExecutor') as mock_executor:
            mock_executor.side_effect = Exception("Critical error")
            
            results, errors = self.scraper.batch_check_mentions(companies, max_workers=1)
            
            # Should still return all companies with False flags
            assert len(results) == 2
            assert len(errors) == 1
            assert all(not result['yc_s25_on_linkedin'] for result in results)
            assert "Critical error" in errors[0]
    
    def test_context_manager(self):
        """Test LinkedInScraper as context manager."""
        with LinkedInScraper() as scraper:
            assert scraper.session is not None
        # Session should be closed after context exit
    
    def test_initialization_with_custom_params(self):
        """Test scraper initialization with custom parameters."""
        scraper = LinkedInScraper(delay_range=(2, 4), max_retries=5)
        
        assert scraper.delay_range == (2, 4)
        assert scraper.max_retries == 5
        assert 'User-Agent' in scraper.session.headers
        
        scraper.close()


if __name__ == "__main__":
    pytest.main([__file__])