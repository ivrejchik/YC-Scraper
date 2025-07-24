"""
Company Processor Module

Orchestrates the data collection and enrichment process.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .yc_client import YcClient
from .linkedin_scraper import LinkedInScraper
from .data_manager import DataManager
from .models import ProcessingResult, create_empty_processing_result

logger = logging.getLogger(__name__)


class CompanyProcessor:
    """
    Orchestrates the complete company data collection and enrichment process.
    
    This class coordinates YC data fetching, LinkedIn enrichment, and data persistence
    to provide incremental processing that only handles new companies.
    """
    
    def __init__(self, csv_file_path: str = "yc_s25_companies.csv", 
                 max_workers: int = 3, batch_size: int = 10, 
                 linkedin_delay_range: tuple = (1, 2),
                 season: str = "Summer", year: int = 2025):
        """
        Initialize the CompanyProcessor with configurable optimization parameters.
        
        Args:
            csv_file_path: Path to CSV file for data persistence
            max_workers: Maximum number of concurrent workers for LinkedIn processing
            batch_size: Number of companies to process in each batch
            linkedin_delay_range: Tuple of (min, max) seconds for LinkedIn rate limiting
            season: YC batch season ("Summer" or "Winter")
            year: YC batch year (e.g., 2025, 2024, etc.)
        """
        self.data_manager = DataManager(csv_file_path)
        self.yc_client = YcClient(season=season, year=year)
        self.linkedin_scraper = LinkedInScraper(delay_range=linkedin_delay_range)
        
        # Batch parameters
        self.season = season
        self.year = year
        
        # Optimization parameters
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.linkedin_delay_range = linkedin_delay_range
        
        # Resume capability state
        self.resume_state_file = csv_file_path.replace('.csv', '_resume_state.json')
        self._processing_interrupted = False
        
    def process_new_companies(self) -> ProcessingResult:
        """
        Main processing pipeline that identifies and processes only new companies.
        
        This method:
        1. Fetches current S25 companies from YC API
        2. Loads existing processed data
        3. Identifies new companies not in existing dataset
        4. Enriches new companies with LinkedIn data
        5. Merges with existing data and saves
        
        Returns:
            ProcessingResult with statistics and error reporting
        """
        start_time = datetime.now()
        result = create_empty_processing_result()
        
        try:
            logger.info("Starting company processing pipeline")
            
            # Step 1: Fetch YC data
            logger.info("Fetching YC S25 company data")
            yc_companies = self._fetch_yc_data()
            
            if not yc_companies:
                logger.warning("No companies found from YC API")
                result.success = True
                result.processing_time = (datetime.now() - start_time).total_seconds()
                return result
            
            logger.info(f"Found {len(yc_companies)} companies from YC API")
            
            # Step 2: Load existing data
            logger.info("Loading existing company data")
            existing_data = self.data_manager.load_existing_data()
            result.total_companies_count = len(existing_data)
            
            # Step 3: Identify new companies
            logger.info("Identifying new companies to process")
            new_companies = self._filter_new_companies(yc_companies, existing_data)
            result.new_companies_count = len(new_companies)
            
            if not new_companies:
                logger.info("No new companies found to process")
                result.success = True
                result.processing_time = (datetime.now() - start_time).total_seconds()
                return result
            
            logger.info(f"Found {len(new_companies)} new companies to process")
            
            # Step 4: Enrich with LinkedIn data
            logger.info("Enriching new companies with LinkedIn data")
            enriched_companies, linkedin_errors = self._enrich_linkedin_data(new_companies)
            result.errors.extend(linkedin_errors)
            
            # Step 5: Merge and save data
            logger.info("Merging and saving updated data")
            merged_data = self.data_manager.merge_and_deduplicate(enriched_companies, existing_data)
            self.data_manager.save_data(merged_data)
            
            result.total_companies_count = len(merged_data)
            result.success = True
            
            logger.info(f"Processing completed successfully:")
            logger.info(f"  - New companies processed: {result.new_companies_count}")
            logger.info(f"  - Total companies in dataset: {result.total_companies_count}")
            logger.info(f"  - Errors encountered: {len(result.errors)}")
            
        except Exception as e:
            error_msg = f"Processing pipeline failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.success = False
        
        finally:
            result.processing_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _fetch_yc_data(self) -> List[Dict[str, Any]]:
        """
        Fetch YC company data using the YC client.
        
        Returns:
            List of company dictionaries from YC API
            
        Raises:
            Exception: If YC data fetching fails completely
        """
        try:
            # Use enriched version to get LinkedIn URLs from YC profile pages
            companies = self.yc_client.get_complete_batch_data_with_enrichment()
            
            # Add timestamp to each company
            current_time = datetime.now().isoformat()
            for company in companies:
                company['last_updated'] = current_time
                # Ensure yc_s25_on_linkedin field exists with default value
                company.setdefault('yc_s25_on_linkedin', False)
            
            return companies
            
        except Exception as e:
            logger.error(f"Failed to fetch YC data: {e}")
            raise
    
    def _filter_new_companies(self, current_companies: List[Dict[str, Any]], 
                            existing_data) -> List[Dict[str, Any]]:
        """
        Filter companies to identify only new ones not in existing dataset.
        
        Args:
            current_companies: List of companies from YC API
            existing_data: DataFrame of existing company data
            
        Returns:
            List of new company dictionaries
        """
        try:
            new_companies = self.data_manager.identify_new_companies(
                current_companies, existing_data
            )
            
            logger.debug(f"Filtered {len(current_companies)} companies to {len(new_companies)} new ones")
            return new_companies
            
        except Exception as e:
            logger.error(f"Failed to filter new companies: {e}")
            # On error, return all companies to ensure processing continues
            return current_companies
    
    def _enrich_linkedin_data(self, companies: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Enrich company data with LinkedIn YC S25 mention flags using optimized batch processing.
        
        Args:
            companies: List of company dictionaries to enrich
            
        Returns:
            Tuple of (enriched_companies, error_messages)
        """
        if not companies:
            return [], []
        
        try:
            logger.info(f"Starting optimized LinkedIn enrichment for {len(companies)} companies")
            logger.info(f"Using {self.max_workers} workers with batch size {self.batch_size}")
            
            # Check for resume state
            resume_data = self._load_resume_state()
            if resume_data:
                logger.info(f"Resuming processing from previous interruption")
                companies = self._filter_companies_for_resume(companies, resume_data)
            
            # Process companies in batches with parallel processing
            enriched_companies, errors = self._process_companies_in_batches(companies)
            
            # Clear resume state on successful completion
            self._clear_resume_state()
            
            logger.info(f"LinkedIn enrichment completed:")
            logger.info(f"  - Companies processed: {len(enriched_companies)}")
            logger.info(f"  - YC S25 mentions found: {sum(1 for c in enriched_companies if c.get('yc_s25_on_linkedin', False))}")
            logger.info(f"  - Errors encountered: {len(errors)}")
            
            return enriched_companies, errors
            
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            self._processing_interrupted = True
            raise
        except Exception as e:
            error_msg = f"LinkedIn enrichment failed: {e}"
            logger.error(error_msg)
            
            # On error, return companies with False flags
            fallback_companies = []
            for company in companies:
                company_copy = company.copy()
                company_copy['yc_s25_on_linkedin'] = False
                fallback_companies.append(company_copy)
            
            return fallback_companies, [error_msg]
    
    def _log_progress(self, completed: int, total: int):
        """
        Progress callback for LinkedIn processing.
        
        Args:
            completed: Number of companies completed
            total: Total number of companies to process
        """
        if completed % 5 == 0 or completed == total:  # Log every 5 companies or at completion
            percentage = (completed / total) * 100
            logger.info(f"LinkedIn processing progress: {completed}/{total} ({percentage:.1f}%)")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        try:
            existing_data = self.data_manager.load_existing_data()
            
            if existing_data.empty:
                return {
                    'total_companies': 0,
                    'companies_with_linkedin': 0,
                    'yc_s25_mentions': 0,
                    'last_updated': None,
                    'data_integrity_issues': 0
                }
            
            # Calculate statistics
            total_companies = len(existing_data)
            companies_with_linkedin = len(existing_data[(existing_data['linkedin_url'] != '') & (existing_data['linkedin_url'].notna())])
            yc_s25_mentions = len(existing_data[existing_data['yc_s25_on_linkedin'] == True])
            
            # Get most recent update time
            last_updated = None
            if 'last_updated' in existing_data.columns:
                try:
                    last_updated = existing_data['last_updated'].max()
                except:
                    pass
            
            # Check data integrity
            integrity_result = self.data_manager.perform_data_integrity_check()
            data_integrity_issues = len(integrity_result.errors)
            
            return {
                'total_companies': total_companies,
                'companies_with_linkedin': companies_with_linkedin,
                'yc_s25_mentions': yc_s25_mentions,
                'last_updated': last_updated,
                'data_integrity_issues': data_integrity_issues
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {
                'total_companies': 0,
                'companies_with_linkedin': 0,
                'yc_s25_mentions': 0,
                'last_updated': None,
                'data_integrity_issues': 1,
                'error': str(e)
            }
    
    def validate_data_integrity(self) -> ProcessingResult:
        """
        Validate the integrity of stored data.
        
        Returns:
            ProcessingResult with validation results
        """
        try:
            logger.info("Starting data integrity validation")
            result = self.data_manager.perform_data_integrity_check()
            
            if result.success:
                logger.info("Data integrity validation passed")
            else:
                logger.warning(f"Data integrity validation found {len(result.errors)} issues")
                for error in result.errors[:5]:  # Log first 5 errors
                    logger.warning(f"  - {error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
            return ProcessingResult(
                new_companies_count=0,
                total_companies_count=0,
                errors=[f"Validation failed: {e}"],
                processing_time=0.0,
                success=False
            )
    
    def repair_data_issues(self) -> Tuple[bool, List[str]]:
        """
        Attempt to repair data integrity issues.
        
        Returns:
            Tuple of (success, list_of_actions_taken)
        """
        try:
            logger.info("Starting data repair process")
            success, actions = self.data_manager.repair_corrupted_data()
            
            if success:
                logger.info("Data repair completed successfully")
                for action in actions:
                    logger.info(f"  - {action}")
            else:
                logger.error("Data repair failed")
                for action in actions:
                    logger.error(f"  - {action}")
            
            return success, actions
            
        except Exception as e:
            error_msg = f"Data repair process failed: {e}"
            logger.error(error_msg)
            return False, [error_msg]
    
    def create_data_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a manual backup of the current data.
        
        Args:
            backup_name: Optional custom name for the backup
            
        Returns:
            Path to the created backup file
            
        Raises:
            Exception: If backup creation fails
        """
        try:
            logger.info(f"Creating data backup{f' with name: {backup_name}' if backup_name else ''}")
            backup_path = self.data_manager.create_manual_backup(backup_name)
            logger.info(f"Backup created successfully: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def _process_companies_in_batches(self, companies: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Process companies in batches with parallel processing and resume capability.
        
        Args:
            companies: List of company dictionaries to process
            
        Returns:
            Tuple of (enriched_companies, error_messages)
        """
        all_enriched_companies = []
        all_errors = []
        
        # Split companies into batches
        batches = [companies[i:i + self.batch_size] for i in range(0, len(companies), self.batch_size)]
        
        logger.info(f"Processing {len(companies)} companies in {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches):
            try:
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} companies)")
                
                # Save resume state before processing batch
                self._save_resume_state(batch_idx, len(batches), all_enriched_companies)
                
                # Process batch with parallel LinkedIn checks
                batch_enriched, batch_errors = self.linkedin_scraper.batch_check_mentions(
                    batch,
                    max_workers=self.max_workers,
                    progress_callback=lambda completed, total: self._log_batch_progress(
                        batch_idx + 1, len(batches), completed, total
                    )
                )
                
                all_enriched_companies.extend(batch_enriched)
                all_errors.extend(batch_errors)
                
                # Log batch completion
                mentions_in_batch = sum(1 for c in batch_enriched if c.get('yc_s25_on_linkedin', False))
                logger.info(f"Batch {batch_idx + 1} completed: {mentions_in_batch}/{len(batch)} companies with YC mentions")
                
            except KeyboardInterrupt:
                logger.warning(f"Processing interrupted during batch {batch_idx + 1}")
                self._processing_interrupted = True
                raise
            except Exception as e:
                error_msg = f"Error processing batch {batch_idx + 1}: {e}"
                logger.error(error_msg)
                all_errors.append(error_msg)
                
                # Add batch companies with False flags on error
                for company in batch:
                    company_copy = company.copy()
                    company_copy['yc_s25_on_linkedin'] = False
                    all_enriched_companies.append(company_copy)
        
        return all_enriched_companies, all_errors
    
    def _log_batch_progress(self, batch_num: int, total_batches: int, completed: int, total: int):
        """
        Enhanced progress callback that includes batch information.
        
        Args:
            batch_num: Current batch number
            total_batches: Total number of batches
            completed: Companies completed in current batch
            total: Total companies in current batch
        """
        if completed % 3 == 0 or completed == total:  # Log every 3 companies or at completion
            batch_percentage = (completed / total) * 100
            overall_percentage = ((batch_num - 1) / total_batches + (completed / total) / total_batches) * 100
            logger.info(f"Batch {batch_num}/{total_batches}: {completed}/{total} ({batch_percentage:.1f}%) | Overall: {overall_percentage:.1f}%")
    
    def _save_resume_state(self, current_batch: int, total_batches: int, processed_companies: List[Dict[str, Any]]):
        """
        Save current processing state for resume capability.
        
        Args:
            current_batch: Current batch index
            total_batches: Total number of batches
            processed_companies: Companies processed so far
        """
        import json
        
        try:
            resume_state = {
                'timestamp': datetime.now().isoformat(),
                'current_batch': current_batch,
                'total_batches': total_batches,
                'processed_count': len(processed_companies),
                'processed_company_names': [c.get('name', '') for c in processed_companies],
                'optimization_params': {
                    'max_workers': self.max_workers,
                    'batch_size': self.batch_size,
                    'linkedin_delay_range': self.linkedin_delay_range
                }
            }
            
            with open(self.resume_state_file, 'w') as f:
                json.dump(resume_state, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")
    
    def _load_resume_state(self) -> Optional[Dict[str, Any]]:
        """
        Load resume state from previous interrupted processing.
        
        Returns:
            Resume state dictionary or None if no valid state exists
        """
        import json
        import os
        
        try:
            if not os.path.exists(self.resume_state_file):
                return None
            
            with open(self.resume_state_file, 'r') as f:
                resume_state = json.load(f)
            
            # Validate resume state
            required_fields = ['timestamp', 'current_batch', 'total_batches', 'processed_company_names']
            if not all(field in resume_state for field in required_fields):
                logger.warning("Invalid resume state file, ignoring")
                return None
            
            # Check if resume state is recent (within last 24 hours)
            from datetime import timedelta
            resume_time = datetime.fromisoformat(resume_state['timestamp'])
            if datetime.now() - resume_time > timedelta(hours=24):
                logger.info("Resume state is too old, starting fresh")
                return None
            
            logger.info(f"Found resume state from {resume_state['timestamp']}")
            logger.info(f"Previous session processed {resume_state['processed_count']} companies")
            
            return resume_state
            
        except Exception as e:
            logger.warning(f"Failed to load resume state: {e}")
            return None
    
    def _filter_companies_for_resume(self, companies: List[Dict[str, Any]], 
                                   resume_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter companies to exclude those already processed in previous session.
        
        Args:
            companies: List of companies to process
            resume_data: Resume state data
            
        Returns:
            Filtered list of companies to process
        """
        processed_names = set(resume_data.get('processed_company_names', []))
        
        remaining_companies = [
            company for company in companies 
            if company.get('name', '') not in processed_names
        ]
        
        logger.info(f"Resume: Skipping {len(companies) - len(remaining_companies)} already processed companies")
        return remaining_companies
    
    def _clear_resume_state(self):
        """Clear resume state file after successful completion."""
        import os
        
        try:
            if os.path.exists(self.resume_state_file):
                os.remove(self.resume_state_file)
                logger.debug("Resume state cleared")
        except Exception as e:
            logger.warning(f"Failed to clear resume state: {e}")
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """
        Get current optimization configuration.
        
        Returns:
            Dictionary with optimization parameters
        """
        return {
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'linkedin_delay_range': self.linkedin_delay_range,
            'resume_state_file': self.resume_state_file
        }
    
    def update_optimization_config(self, max_workers: Optional[int] = None,
                                 batch_size: Optional[int] = None,
                                 linkedin_delay_range: Optional[tuple] = None):
        """
        Update optimization configuration parameters.
        
        Args:
            max_workers: Maximum number of concurrent workers
            batch_size: Number of companies to process in each batch
            linkedin_delay_range: Tuple of (min, max) seconds for rate limiting
        """
        if max_workers is not None:
            self.max_workers = max(1, min(max_workers, 10))  # Limit between 1-10
            logger.info(f"Updated max_workers to {self.max_workers}")
        
        if batch_size is not None:
            self.batch_size = max(1, min(batch_size, 50))  # Limit between 1-50
            logger.info(f"Updated batch_size to {self.batch_size}")
        
        if linkedin_delay_range is not None:
            if len(linkedin_delay_range) == 2 and linkedin_delay_range[0] <= linkedin_delay_range[1]:
                self.linkedin_delay_range = linkedin_delay_range
                # Update LinkedIn scraper with new delay range
                self.linkedin_scraper = LinkedInScraper(delay_range=linkedin_delay_range)
                logger.info(f"Updated linkedin_delay_range to {self.linkedin_delay_range}")
            else:
                logger.warning("Invalid linkedin_delay_range, keeping current value")
    
    def estimate_processing_time(self, num_companies: int) -> Dict[str, float]:
        """
        Estimate processing time based on optimization parameters and company count.
        
        Args:
            num_companies: Number of companies to process
            
        Returns:
            Dictionary with time estimates in seconds
        """
        # Base time estimates (in seconds)
        yc_fetch_time_per_company = 0.5  # YC API is fast
        linkedin_check_time_per_company = sum(self.linkedin_delay_range) / 2 + 2  # Delay + processing
        
        # Calculate parallel processing benefit
        effective_linkedin_time = linkedin_check_time_per_company / self.max_workers
        
        # Calculate batch overhead
        num_batches = (num_companies + self.batch_size - 1) // self.batch_size
        batch_overhead = num_batches * 1  # 1 second overhead per batch
        
        estimates = {
            'yc_data_fetch': num_companies * yc_fetch_time_per_company,
            'linkedin_enrichment': num_companies * effective_linkedin_time + batch_overhead,
            'data_processing': num_companies * 0.1,  # Data merge/save time
            'total_estimated': (num_companies * yc_fetch_time_per_company + 
                              num_companies * effective_linkedin_time + 
                              batch_overhead + 
                              num_companies * 0.1)
        }
        
        return estimates
    
    def get_resume_status(self) -> Dict[str, Any]:
        """
        Get information about any interrupted processing session.
        
        Returns:
            Dictionary with resume status information
        """
        resume_data = self._load_resume_state()
        
        if not resume_data:
            return {
                'has_resume_data': False,
                'can_resume': False
            }
        
        return {
            'has_resume_data': True,
            'can_resume': True,
            'interrupted_at': resume_data['timestamp'],
            'processed_count': resume_data['processed_count'],
            'current_batch': resume_data['current_batch'],
            'total_batches': resume_data['total_batches'],
            'optimization_params': resume_data.get('optimization_params', {})
        }
    
    def clear_resume_data(self):
        """Manually clear resume data (useful for starting fresh)."""
        self._clear_resume_state()
        logger.info("Resume data cleared manually")
    
    def close(self):
        """Clean up resources."""
        try:
            self.linkedin_scraper.close()
        except:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()