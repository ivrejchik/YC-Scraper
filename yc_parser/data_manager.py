"""
Data persistence layer for the YC Company Parser.

This module handles CSV data operations including loading, saving, deduplication,
and data validation for company records.
"""

import pandas as pd
import os
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from .models import Company, validate_company_data, ProcessingResult

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data persistence and operations for company records.
    
    Handles CSV file operations, data deduplication, validation,
    and backup functionality for data safety.
    """
    
    def __init__(self, csv_file_path: str = "yc_s25_companies.csv"):
        """
        Initialize DataManager with CSV file path.
        
        Args:
            csv_file_path: Path to the CSV file for data storage
        """
        self.csv_file_path = csv_file_path
        self.backup_dir = "data_backups"
        self._ensure_backup_directory()
    
    def load_existing_data(self) -> pd.DataFrame:
        """
        Load existing company data from CSV file.
        
        Returns:
            DataFrame containing existing company data, empty if file doesn't exist
            
        Raises:
            Exception: If CSV file is corrupted and cannot be recovered
        """
        if not os.path.exists(self.csv_file_path):
            return self._create_empty_dataframe()
        
        try:
            df = pd.read_csv(self.csv_file_path)
            
            # Validate the loaded data structure
            if not self._validate_dataframe_structure(df):
                raise ValueError("CSV file has invalid structure")
            
            # Convert boolean column if it exists as string
            if 'yc_s25_on_linkedin' in df.columns:
                df['yc_s25_on_linkedin'] = df['yc_s25_on_linkedin'].astype(bool)
            
            return df
            
        except Exception as e:
            # Try to recover from backup
            backup_df = self._try_recover_from_backup()
            if backup_df is not None:
                print(f"Warning: Main CSV file corrupted, recovered from backup. Error: {e}")
                return backup_df
            
            # If no backup available, start fresh but log the error
            print(f"Error loading CSV file and no backup available: {e}")
            print("Starting with empty dataset")
            return self._create_empty_dataframe()
    
    def save_data(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to CSV file with backup creation.
        
        Args:
            df: DataFrame to save
            
        Raises:
            Exception: If save operation fails
        """
        if df.empty:
            # Create empty file with headers
            empty_df = self._create_empty_dataframe()
            empty_df.to_csv(self.csv_file_path, index=False)
            return
        
        # Validate data before saving
        validation_errors = self._validate_dataframe_data(df)
        if validation_errors:
            raise ValueError(f"Cannot save invalid data: {validation_errors}")
        
        # Create backup before saving
        self._create_backup()
        
        try:
            # Ensure proper column order and data types
            df_to_save = self._prepare_dataframe_for_save(df)
            df_to_save.to_csv(self.csv_file_path, index=False)
            
        except Exception as e:
            # Restore from backup if save fails
            self._restore_from_backup()
            raise Exception(f"Failed to save data: {e}")
    
    def identify_new_companies(self, current_companies: List[Dict[str, Any]], 
                             existing_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify companies that are not in the existing dataset.
        
        Args:
            current_companies: List of company dictionaries from YC API
            existing_df: DataFrame of existing company data
            
        Returns:
            List of new company dictionaries not in existing data
        """
        if existing_df.empty:
            return current_companies
        
        existing_names = set(existing_df['name'].str.lower().str.strip())
        new_companies = []
        
        for company in current_companies:
            company_name = company.get('name', '').lower().strip()
            if company_name and company_name not in existing_names:
                new_companies.append(company)
        
        return new_companies
    
    def merge_and_deduplicate(self, new_data: List[Dict[str, Any]], 
                            existing_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge new data with existing data and remove duplicates.
        
        Args:
            new_data: List of new company dictionaries
            existing_df: DataFrame of existing data
            
        Returns:
            DataFrame with merged and deduplicated data
        """
        if not new_data:
            return existing_df
        
        # Convert new data to DataFrame
        new_df = pd.DataFrame(new_data)
        
        # If existing data is empty, return new data
        if existing_df.empty:
            return self._prepare_dataframe_for_save(new_df)
        
        # Combine dataframes
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Apply advanced deduplication with conflict resolution
        deduplicated_df = self._deduplicate_with_conflict_resolution(combined_df)
        
        return self._prepare_dataframe_for_save(deduplicated_df)
    
    def _deduplicate_with_conflict_resolution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates with intelligent conflict resolution.
        
        Args:
            df: DataFrame with potential duplicates
            
        Returns:
            DataFrame with duplicates resolved
        """
        if df.empty:
            return df
        
        # Create normalized name for comparison
        df['name_normalized'] = df['name'].str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)
        
        # Group by normalized name to find duplicates
        grouped = df.groupby('name_normalized')
        
        resolved_rows = []
        
        for name, group in grouped:
            if len(group) == 1:
                # No duplicates, keep as is
                resolved_rows.append(group.iloc[0])
            else:
                # Multiple entries, resolve conflicts
                resolved_row = self._resolve_duplicate_conflicts(group)
                resolved_rows.append(resolved_row)
        
        # Create new DataFrame from resolved rows
        result_df = pd.DataFrame(resolved_rows)
        
        # Remove helper column
        if 'name_normalized' in result_df.columns:
            result_df = result_df.drop('name_normalized', axis=1)
        
        return result_df.reset_index(drop=True)
    
    def _resolve_duplicate_conflicts(self, duplicate_group: pd.DataFrame) -> pd.Series:
        """
        Resolve conflicts between duplicate company entries.
        
        Conflict resolution strategy:
        1. Use most recent last_updated timestamp
        2. Prefer non-empty values over empty ones
        3. For LinkedIn flag, prefer True over False if recent
        
        Args:
            duplicate_group: DataFrame containing duplicate entries
            
        Returns:
            Series representing the resolved company record
        """
        # Sort by last_updated to get most recent first
        duplicate_group['last_updated_dt'] = pd.to_datetime(duplicate_group['last_updated'])
        sorted_group = duplicate_group.sort_values('last_updated_dt', ascending=False)
        
        # Start with the most recent entry
        resolved = sorted_group.iloc[0].copy()
        
        # For each field, prefer non-empty values from any entry
        for column in ['website', 'description', 'yc_page', 'linkedin_url']:
            if pd.isna(resolved[column]) or resolved[column] == '':
                # Look for non-empty value in other entries
                for _, row in sorted_group.iterrows():
                    if not pd.isna(row[column]) and row[column] != '':
                        resolved[column] = row[column]
                        break
        
        # Special handling for LinkedIn flag - prefer True if it's from a recent entry
        recent_threshold = sorted_group['last_updated_dt'].max() - pd.Timedelta(days=30)
        recent_entries = sorted_group[sorted_group['last_updated_dt'] >= recent_threshold]
        
        if any(recent_entries['yc_s25_on_linkedin']):
            resolved['yc_s25_on_linkedin'] = True
        
        # Remove helper column
        if 'last_updated_dt' in resolved.index:
            resolved = resolved.drop('last_updated_dt')
        
        return resolved
    
    def find_potential_duplicates(self, df: pd.DataFrame, 
                                similarity_threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Find potential duplicate companies based on name similarity.
        
        Args:
            df: DataFrame to check for duplicates
            similarity_threshold: Minimum similarity score to consider as duplicate
            
        Returns:
            List of tuples (index1, index2, similarity_score) for potential duplicates
        """
        from difflib import SequenceMatcher
        
        potential_duplicates = []
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                name1 = df.iloc[i]['name'].lower().strip()
                name2 = df.iloc[j]['name'].lower().strip()
                
                # Calculate similarity
                similarity = SequenceMatcher(None, name1, name2).ratio()
                
                if similarity >= similarity_threshold:
                    potential_duplicates.append((i, j, similarity))
        
        return potential_duplicates
    
    def remove_duplicates_by_similarity(self, df: pd.DataFrame, 
                                      similarity_threshold: float = 0.9) -> pd.DataFrame:
        """
        Remove duplicates based on name similarity rather than exact matches.
        
        Args:
            df: DataFrame to deduplicate
            similarity_threshold: Minimum similarity to consider as duplicate
            
        Returns:
            DataFrame with similar duplicates removed
        """
        if df.empty:
            return df
        
        duplicates = self.find_potential_duplicates(df, similarity_threshold)
        indices_to_remove = set()
        
        for i, j, similarity in duplicates:
            if i not in indices_to_remove and j not in indices_to_remove:
                # Keep the entry with more complete data
                row_i = df.iloc[i]
                row_j = df.iloc[j]
                
                # Count non-empty fields
                completeness_i = sum(1 for field in ['website', 'description', 'linkedin_url'] 
                                   if not pd.isna(row_i[field]) and row_i[field] != '')
                completeness_j = sum(1 for field in ['website', 'description', 'linkedin_url'] 
                                   if not pd.isna(row_j[field]) and row_j[field] != '')
                
                # Remove the less complete entry, or the older one if equal
                if completeness_i > completeness_j:
                    indices_to_remove.add(j)
                elif completeness_j > completeness_i:
                    indices_to_remove.add(i)
                else:
                    # Equal completeness, keep more recent
                    date_i = pd.to_datetime(row_i['last_updated'])
                    date_j = pd.to_datetime(row_j['last_updated'])
                    if date_i >= date_j:
                        indices_to_remove.add(j)
                    else:
                        indices_to_remove.add(i)
        
        # Remove identified duplicates
        return df.drop(df.index[list(indices_to_remove)]).reset_index(drop=True)
    
    def validate_data_integrity(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the integrity of data in DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check DataFrame structure
        if not self._validate_dataframe_structure(df):
            errors.append("Invalid DataFrame structure")
            return False, errors
        
        # Validate each row
        validation_errors = self._validate_dataframe_data(df)
        if validation_errors:
            errors.extend(validation_errors)
        
        return len(errors) == 0, errors
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with proper column structure."""
        columns = [
            'name', 'website', 'description', 'yc_page', 
            'linkedin_url', 'yc_s25_on_linkedin', 'linkedin_only', 'last_updated'
        ]
        return pd.DataFrame(columns=columns)
    
    def _validate_dataframe_structure(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame has the expected column structure."""
        expected_columns = {
            'name', 'website', 'description', 'yc_page', 
            'linkedin_url', 'yc_s25_on_linkedin', 'linkedin_only', 'last_updated'
        }
        # Allow for backward compatibility - if linkedin_only is missing, that's okay
        df_columns = set(df.columns)
        required_columns = {
            'name', 'website', 'description', 'yc_page', 
            'linkedin_url', 'yc_s25_on_linkedin', 'last_updated'
        }
        return required_columns.issubset(df_columns)
    
    def _validate_dataframe_data(self, df: pd.DataFrame) -> List[str]:
        """Validate data in each row of DataFrame."""
        all_errors = []
        
        for index, row in df.iterrows():
            row_data = row.to_dict()
            row_errors = validate_company_data(row_data)
            if row_errors:
                all_errors.extend([f"Row {index}: {error}" for error in row_errors])
        
        return all_errors
    
    def _prepare_dataframe_for_save(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for saving with proper column order and types."""
        if df.empty:
            return self._create_empty_dataframe()
        
        # Ensure proper column order
        column_order = [
            'name', 'website', 'description', 'yc_page', 
            'linkedin_url', 'yc_s25_on_linkedin', 'linkedin_only', 'last_updated'
        ]
        
        # Reorder columns and ensure all required columns exist
        df_prepared = df.reindex(columns=column_order)
        
        # Fill missing columns with default values
        df_prepared['name'] = df_prepared['name'].fillna('')
        df_prepared['website'] = df_prepared['website'].fillna('')
        df_prepared['description'] = df_prepared['description'].fillna('')
        df_prepared['yc_page'] = df_prepared['yc_page'].fillna('')
        df_prepared['linkedin_url'] = df_prepared['linkedin_url'].fillna('')
        df_prepared['yc_s25_on_linkedin'] = df_prepared['yc_s25_on_linkedin'].fillna(False)
        df_prepared['linkedin_only'] = df_prepared['linkedin_only'].fillna(False)
        df_prepared['last_updated'] = df_prepared['last_updated'].fillna(datetime.now().isoformat())
        
        return df_prepared
    
    def _ensure_backup_directory(self) -> None:
        """Ensure backup directory exists."""
        Path(self.backup_dir).mkdir(exist_ok=True)
    
    def _create_backup(self) -> None:
        """Create backup of current CSV file if it exists."""
        if os.path.exists(self.csv_file_path):
            self._ensure_backup_directory()  # Ensure directory exists
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"yc_s25_companies_backup_{timestamp}.csv"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            shutil.copy2(self.csv_file_path, backup_path)
            
            # Keep only the 5 most recent backups
            self._cleanup_old_backups()
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files, keeping only the 5 most recent."""
        backup_files = []
        for filename in os.listdir(self.backup_dir):
            if filename.startswith("yc_s25_companies_backup_") and filename.endswith(".csv"):
                filepath = os.path.join(self.backup_dir, filename)
                backup_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove files beyond the 5 most recent
        for filepath, _ in backup_files[5:]:
            try:
                os.remove(filepath)
            except OSError:
                pass  # Ignore errors when removing old backups
    
    def _try_recover_from_backup(self) -> Optional[pd.DataFrame]:
        """Try to recover data from the most recent backup."""
        if not os.path.exists(self.backup_dir):
            return None
        
        backup_files = []
        for filename in os.listdir(self.backup_dir):
            if filename.startswith("yc_s25_companies_backup_") and filename.endswith(".csv"):
                filepath = os.path.join(self.backup_dir, filename)
                backup_files.append((filepath, os.path.getmtime(filepath)))
        
        if not backup_files:
            return None
        
        # Get the most recent backup
        most_recent_backup = max(backup_files, key=lambda x: x[1])[0]
        
        try:
            df = pd.read_csv(most_recent_backup)
            if self._validate_dataframe_structure(df):
                return df
        except Exception:
            pass
        
        return None
    
    def _restore_from_backup(self) -> None:
        """Restore CSV file from the most recent backup."""
        backup_df = self._try_recover_from_backup()
        if backup_df is not None:
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("yc_s25_companies_backup_") and filename.endswith(".csv"):
                    filepath = os.path.join(self.backup_dir, filename)
                    backup_files.append((filepath, os.path.getmtime(filepath)))
            
            if backup_files:
                most_recent_backup = max(backup_files, key=lambda x: x[1])[0]
                shutil.copy2(most_recent_backup, self.csv_file_path)
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate data and attempt to clean/fix common issues.
        
        Args:
            df: DataFrame to validate and clean
            
        Returns:
            Tuple of (cleaned_dataframe, list_of_warnings)
        """
        warnings = []
        cleaned_df = df.copy()
        
        if cleaned_df.empty:
            return cleaned_df, warnings
        
        # Fix common data issues
        try:
            # Clean company names
            cleaned_df['name'] = cleaned_df['name'].astype(str).str.strip()
            empty_names = cleaned_df['name'].isin(['', 'nan', 'None'])
            if empty_names.any():
                warnings.append(f"Found {empty_names.sum()} companies with empty names, removing them")
                cleaned_df = cleaned_df[~empty_names]
            
            # Clean URLs
            url_columns = ['website', 'yc_page', 'linkedin_url']
            for col in url_columns:
                if col in cleaned_df.columns:
                    # Convert to string and clean
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                    # Replace 'nan', 'None', etc. with empty string
                    cleaned_df[col] = cleaned_df[col].replace(['nan', 'None', 'null'], '')
                    
                    # Validate URL format for non-empty values
                    non_empty = cleaned_df[col] != ''
                    if non_empty.any():
                        invalid_urls = []
                        for idx, url in cleaned_df[non_empty][col].items():
                            if not self._is_valid_url_format(url):
                                invalid_urls.append(idx)
                        
                        if invalid_urls:
                            warnings.append(f"Found {len(invalid_urls)} invalid URLs in {col}, setting to empty")
                            cleaned_df.loc[invalid_urls, col] = ''
            
            # Clean descriptions
            if 'description' in cleaned_df.columns:
                cleaned_df['description'] = cleaned_df['description'].astype(str).str.strip()
                cleaned_df['description'] = cleaned_df['description'].replace(['nan', 'None', 'null'], '')
                
                # Truncate overly long descriptions
                long_descriptions = cleaned_df['description'].str.len() > 500
                if long_descriptions.any():
                    warnings.append(f"Found {long_descriptions.sum()} descriptions over 500 characters, truncating")
                    cleaned_df.loc[long_descriptions, 'description'] = cleaned_df.loc[long_descriptions, 'description'].str[:500]
            
            # Clean boolean column
            if 'yc_s25_on_linkedin' in cleaned_df.columns:
                # Convert various representations to boolean
                bool_map = {
                    'true': True, 'True': True, 'TRUE': True, '1': True, 1: True,
                    'false': False, 'False': False, 'FALSE': False, '0': False, 0: False,
                    'nan': False, 'None': False, 'null': False, '': False
                }
                
                cleaned_df['yc_s25_on_linkedin'] = cleaned_df['yc_s25_on_linkedin'].map(
                    lambda x: bool_map.get(str(x), False)
                )
            
            # Clean timestamps
            if 'last_updated' in cleaned_df.columns:
                invalid_timestamps = []
                for idx, timestamp in cleaned_df['last_updated'].items():
                    if not self._is_valid_timestamp_format(str(timestamp)):
                        invalid_timestamps.append(idx)
                
                if invalid_timestamps:
                    warnings.append(f"Found {len(invalid_timestamps)} invalid timestamps, setting to current time")
                    cleaned_df.loc[invalid_timestamps, 'last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            warnings.append(f"Error during data cleaning: {e}")
        
        return cleaned_df, warnings
    
    def perform_data_integrity_check(self) -> ProcessingResult:
        """
        Perform comprehensive data integrity check on stored data.
        
        Returns:
            ProcessingResult with integrity check results
        """
        start_time = datetime.now()
        errors = []
        
        try:
            # Load existing data
            df = self.load_existing_data()
            
            if df.empty:
                return ProcessingResult(
                    new_companies_count=0,
                    total_companies_count=0,
                    errors=[],
                    processing_time=0.0,
                    success=True
                )
            
            # Check for structural issues
            is_valid, validation_errors = self.validate_data_integrity(df)
            if not is_valid:
                errors.extend(validation_errors)
            
            # Check for duplicates
            duplicates = self.find_potential_duplicates(df, similarity_threshold=0.9)
            if duplicates:
                errors.append(f"Found {len(duplicates)} potential duplicate pairs")
            
            # Check data completeness
            completeness_issues = self._check_data_completeness(df)
            if completeness_issues:
                errors.extend(completeness_issues)
            
            # Check for data anomalies
            anomalies = self._detect_data_anomalies(df)
            if anomalies:
                errors.extend(anomalies)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                new_companies_count=0,
                total_companies_count=len(df),
                errors=errors,
                processing_time=processing_time,
                success=len(errors) == 0
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                new_companies_count=0,
                total_companies_count=0,
                errors=[f"Integrity check failed: {e}"],
                processing_time=processing_time,
                success=False
            )
    
    def repair_corrupted_data(self) -> Tuple[bool, List[str]]:
        """
        Attempt to repair corrupted data using backups and data cleaning.
        
        Returns:
            Tuple of (success, list_of_actions_taken)
        """
        actions = []
        
        try:
            # Try to load current data
            df = self.load_existing_data()
            
            # Validate and clean the data
            cleaned_df, warnings = self.validate_and_clean_data(df)
            actions.extend(warnings)
            
            # If data was cleaned, save it
            if warnings:
                self.save_data(cleaned_df)
                actions.append("Saved cleaned data")
            
            # Perform integrity check
            integrity_result = self.perform_data_integrity_check()
            if not integrity_result.success:
                actions.append(f"Integrity check found {len(integrity_result.errors)} issues")
                
                # Try to recover from backup if integrity is poor
                backup_df = self._try_recover_from_backup()
                if backup_df is not None:
                    backup_cleaned, backup_warnings = self.validate_and_clean_data(backup_df)
                    backup_integrity = self.validate_data_integrity(backup_cleaned)
                    
                    if backup_integrity[0]:  # If backup data is valid
                        self.save_data(backup_cleaned)
                        actions.append("Restored data from backup")
                        actions.extend(backup_warnings)
                    else:
                        actions.append("Backup data also corrupted, manual intervention required")
                        return False, actions
            
            return True, actions
            
        except Exception as e:
            actions.append(f"Repair failed: {e}")
            return False, actions
    
    def create_manual_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a manual backup with custom name.
        
        Args:
            backup_name: Optional custom name for backup
            
        Returns:
            Path to created backup file
        """
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError("No data file exists to backup")
        
        self._ensure_backup_directory()
        
        if backup_name:
            backup_filename = f"yc_s25_companies_manual_{backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            backup_filename = f"yc_s25_companies_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        backup_path = os.path.join(self.backup_dir, backup_filename)
        shutil.copy2(self.csv_file_path, backup_path)
        
        return backup_path
    
    def merge_discovery_results(self, discovery_companies: List[Any], 
                              existing_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge LinkedIn discovery results with existing data.
        
        This method handles the integration of discovered LinkedIn companies
        with existing YC data, performing deduplication and data validation.
        
        Args:
            discovery_companies: List of Company objects from discovery process
            existing_df: Optional existing DataFrame (loads from file if None)
            
        Returns:
            DataFrame with merged and deduplicated data
        """
        logger.info(f"Merging {len(discovery_companies)} discovery results with existing data")
        
        # Load existing data if not provided
        if existing_df is None:
            existing_df = self.load_existing_data()
        
        # Convert discovery companies to dictionaries
        discovery_data = []
        for company in discovery_companies:
            if hasattr(company, 'to_dict'):
                discovery_data.append(company.to_dict())
            elif isinstance(company, dict):
                discovery_data.append(company)
            else:
                logger.warning(f"Skipping invalid company object: {type(company)}")
                continue
        
        if not discovery_data:
            logger.info("No valid discovery data to merge")
            return existing_df
        
        # Create DataFrame from discovery data
        discovery_df = pd.DataFrame(discovery_data)
        
        # Ensure proper column structure
        discovery_df = self._prepare_dataframe_for_save(discovery_df)
        
        # If existing data is empty, return discovery data
        if existing_df.empty:
            logger.info("No existing data, returning discovery results")
            return discovery_df
        
        # Perform advanced deduplication with discovery-specific logic
        merged_df = self._merge_with_discovery_deduplication(existing_df, discovery_df)
        
        logger.info(f"Merge completed: {len(merged_df)} total companies")
        return merged_df
    
    def _merge_with_discovery_deduplication(self, existing_df: pd.DataFrame, 
                                          discovery_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge existing data with discovery results using advanced deduplication.
        
        This method implements discovery-specific deduplication logic:
        1. Match by exact LinkedIn URL first
        2. Match by company name similarity
        3. Resolve conflicts by preferring more complete data
        4. Add truly new companies
        
        Args:
            existing_df: DataFrame with existing company data
            discovery_df: DataFrame with discovery results
            
        Returns:
            DataFrame with merged and deduplicated data
        """
        logger.info("Performing discovery-specific deduplication")
        
        # Create working copies
        existing_copy = existing_df.copy()
        discovery_copy = discovery_df.copy()
        
        # Track which discovery companies have been matched
        matched_discovery_indices = set()
        updated_existing_indices = set()
        
        # Phase 1: Match by LinkedIn URL (exact matches)
        logger.debug("Phase 1: Matching by LinkedIn URL")
        for disc_idx, disc_row in discovery_copy.iterrows():
            disc_linkedin = disc_row.get('linkedin_url', '').strip()
            if not disc_linkedin:
                continue
                
            # Find existing companies with same LinkedIn URL
            # Convert to string and handle NaN values
            existing_linkedin_urls = existing_copy['linkedin_url'].astype(str).str.strip()
            existing_matches = existing_copy[existing_linkedin_urls == disc_linkedin]
            
            if not existing_matches.empty:
                # Update existing company with discovery data
                existing_idx = existing_matches.index[0]
                existing_copy.loc[existing_idx] = self._merge_company_records(
                    existing_copy.loc[existing_idx], disc_row
                )
                matched_discovery_indices.add(disc_idx)
                updated_existing_indices.add(existing_idx)
                logger.debug(f"LinkedIn URL match: {disc_row.get('name', 'Unknown')}")
        
        # Phase 2: Match by company name similarity
        logger.debug("Phase 2: Matching by company name similarity")
        unmatched_discovery = discovery_copy.drop(matched_discovery_indices)
        
        for disc_idx, disc_row in unmatched_discovery.iterrows():
            disc_name = disc_row.get('name', '').strip().lower()
            if not disc_name:
                continue
            
            best_match_idx = None
            best_similarity = 0.0
            
            # Find best name match in existing data (excluding already updated)
            for exist_idx, exist_row in existing_copy.iterrows():
                if exist_idx in updated_existing_indices:
                    continue
                    
                exist_name = exist_row.get('name', '').strip().lower()
                if not exist_name:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_name_similarity(disc_name, exist_name)
                
                if similarity > best_similarity and similarity >= 0.85:  # High threshold for name matching
                    best_similarity = similarity
                    best_match_idx = exist_idx
            
            # If good match found, merge the records
            if best_match_idx is not None:
                existing_copy.loc[best_match_idx] = self._merge_company_records(
                    existing_copy.loc[best_match_idx], disc_row
                )
                matched_discovery_indices.add(disc_idx)
                updated_existing_indices.add(best_match_idx)
                logger.debug(f"Name match ({best_similarity:.2f}): {disc_row.get('name', 'Unknown')}")
        
        # Phase 3: Add truly new companies
        logger.debug("Phase 3: Adding new companies")
        new_companies = discovery_copy.drop(matched_discovery_indices)
        
        if not new_companies.empty:
            # Combine existing (possibly updated) with new companies
            merged_df = pd.concat([existing_copy, new_companies], ignore_index=True)
            logger.info(f"Added {len(new_companies)} new companies from discovery")
        else:
            merged_df = existing_copy
            logger.info("No new companies to add from discovery")
        
        # Final cleanup and validation
        merged_df = self._prepare_dataframe_for_save(merged_df)
        
        logger.info(f"Discovery deduplication complete: "
                   f"{len(updated_existing_indices)} updated, "
                   f"{len(new_companies)} new, "
                   f"{len(merged_df)} total")
        
        return merged_df
    
    def _merge_company_records(self, existing_record: pd.Series, 
                             discovery_record: pd.Series) -> pd.Series:
        """
        Merge two company records, preferring more complete data.
        
        Args:
            existing_record: Existing company record
            discovery_record: Discovery company record
            
        Returns:
            Merged company record
        """
        merged = existing_record.copy()
        
        # Fields to merge (prefer non-empty values from discovery)
        mergeable_fields = ['website', 'description', 'linkedin_url']
        
        for field in mergeable_fields:
            # Handle NaN values by converting to string first
            existing_raw = existing_record.get(field, '')
            discovery_raw = discovery_record.get(field, '')
            
            existing_value = str(existing_raw).strip() if pd.notna(existing_raw) else ''
            discovery_value = str(discovery_raw).strip() if pd.notna(discovery_raw) else ''
            
            # Prefer discovery value if existing is empty or discovery is more complete
            if not existing_value and discovery_value:
                merged[field] = discovery_value
            elif discovery_value and len(discovery_value) > len(existing_value):
                # Prefer longer descriptions/more complete data
                if field == 'description':
                    merged[field] = discovery_value
        
        # Always update timestamp to reflect the merge
        merged['last_updated'] = datetime.now().isoformat()
        
        # Special handling for YC page - preserve existing if present
        existing_yc_raw = existing_record.get('yc_page', '')
        existing_yc_page = str(existing_yc_raw).strip() if pd.notna(existing_yc_raw) else ''
        
        if not existing_yc_page:
            discovery_yc_raw = discovery_record.get('yc_page', '')
            discovery_yc_page = str(discovery_yc_raw).strip() if pd.notna(discovery_yc_raw) else ''
            if discovery_yc_page:
                merged['yc_page'] = discovery_yc_page
        
        return merged
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two company names.
        
        Uses multiple similarity metrics and normalization for better matching.
        
        Args:
            name1: First company name
            name2: Second company name
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        from difflib import SequenceMatcher
        
        # Normalize names for comparison
        norm1 = self._normalize_company_name(name1)
        norm2 = self._normalize_company_name(name2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # If normalized names are identical, return perfect match
        if norm1 == norm2:
            return 1.0
        
        # Calculate basic similarity
        basic_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Bonus for exact word matches
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            # Weighted combination of character and word similarity
            combined_similarity = (basic_similarity * 0.6) + (word_overlap * 0.4)
        else:
            combined_similarity = basic_similarity
        
        return min(combined_similarity, 1.0)
    
    def _normalize_company_name(self, name: str) -> str:
        """
        Normalize company name for comparison.
        
        Args:
            name: Company name to normalize
            
        Returns:
            Normalized company name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common company suffixes (but only if they're clearly suffixes)
        suffixes = [
            'inc', 'inc.', 'incorporated',
            'llc', 'l.l.c.', 'limited liability company',
            'ltd', 'ltd.', 'limited',
            'corp', 'corp.', 'corporation',
            'pbc', 'p.b.c.', 'public benefit corporation'
        ]
        
        # Special handling for "company" and "co" - only remove if there are other words
        words = normalized.split()
        if len(words) > 1:
            if words[-1] in ['company', 'co', 'co.']:
                words = words[:-1]
                normalized = ' '.join(words)
        
        # Sort suffixes by length (longest first) to avoid partial matches
        suffixes.sort(key=len, reverse=True)
        
        for suffix in suffixes:
            suffix_pattern = ' ' + suffix
            if normalized.endswith(suffix_pattern):
                normalized = normalized[:-len(suffix_pattern)].strip()
                break  # Only remove one suffix
        
        # Remove punctuation and extra spaces
        import string
        normalized = ''.join(char if char not in string.punctuation else ' ' for char in normalized)
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        
        return normalized
    
    def save_discovery_results(self, discovery_companies: List[Any], 
                             create_backup: bool = True) -> ProcessingResult:
        """
        Save discovery results to CSV with proper backup and validation.
        
        This method implements the complete workflow for saving discovery results:
        1. Create backup of existing data
        2. Load and merge with existing data
        3. Perform deduplication
        4. Validate merged data
        5. Save to CSV
        
        Args:
            discovery_companies: List of Company objects from discovery
            create_backup: Whether to create backup before saving
            
        Returns:
            ProcessingResult with operation details
        """
        start_time = datetime.now()
        errors = []
        
        try:
            logger.info(f"Saving {len(discovery_companies)} discovery results")
            
            # Step 1: Create backup if requested
            if create_backup:
                try:
                    if os.path.exists(self.csv_file_path):
                        backup_path = self.create_manual_backup("discovery")
                        logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to create backup: {e}")
                    errors.append(f"Backup creation failed: {e}")
            
            # Step 2: Load existing data
            existing_df = self.load_existing_data()
            original_count = len(existing_df)
            
            # Step 3: Merge with discovery results
            merged_df = self.merge_discovery_results(discovery_companies, existing_df)
            
            # Step 4: Validate merged data
            cleaned_df, warnings = self.validate_and_clean_data(merged_df)
            if warnings:
                errors.extend([f"Data cleaning: {w}" for w in warnings])
            
            is_valid, validation_errors = self.validate_data_integrity(cleaned_df)
            if not is_valid:
                errors.extend([f"Validation: {e}" for e in validation_errors])
                # Don't save if data is invalid
                raise ValueError("Data validation failed")
            
            # Step 5: Save the merged data
            self.save_data(cleaned_df)
            
            new_count = len(cleaned_df) - original_count
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                new_companies_count=max(0, new_count),
                total_companies_count=len(cleaned_df),
                errors=errors,
                processing_time=processing_time,
                success=True
            )
            
            logger.info(f"Discovery results saved successfully: "
                       f"{result.new_companies_count} new, "
                       f"{result.total_companies_count} total companies")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Failed to save discovery results: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return ProcessingResult(
                new_companies_count=0,
                total_companies_count=0,
                errors=errors,
                processing_time=processing_time,
                success=False
            )
    
    def deduplicate_discovery_results(self, discovery_companies: List[Any]) -> Tuple[List[Any], List[str]]:
        """
        Deduplicate discovery results before merging with existing data.
        
        This method removes duplicates within the discovery results themselves,
        which is useful when multiple search queries return overlapping results.
        
        Args:
            discovery_companies: List of Company objects from discovery
            
        Returns:
            Tuple of (deduplicated_companies, list_of_warnings)
        """
        logger.info(f"Deduplicating {len(discovery_companies)} discovery results")
        
        if not discovery_companies:
            return [], []
        
        warnings = []
        
        # Convert to DataFrame for easier processing
        discovery_data = []
        for company in discovery_companies:
            if hasattr(company, 'to_dict'):
                discovery_data.append(company.to_dict())
            elif isinstance(company, dict):
                discovery_data.append(company)
            else:
                warnings.append(f"Skipping invalid company object: {type(company)}")
                continue
        
        if not discovery_data:
            return [], warnings
        
        discovery_df = pd.DataFrame(discovery_data)
        
        # Deduplicate by LinkedIn URL first (exact matches)
        linkedin_duplicates = discovery_df[
            discovery_df.duplicated(subset=['linkedin_url'], keep='first') & 
            (discovery_df['linkedin_url'] != '')
        ]
        
        if not linkedin_duplicates.empty:
            warnings.append(f"Removed {len(linkedin_duplicates)} LinkedIn URL duplicates")
            discovery_df = discovery_df.drop(linkedin_duplicates.index)
        
        # Deduplicate by name similarity
        similarity_duplicates = []
        processed_indices = set()
        
        for i, row_i in discovery_df.iterrows():
            if i in processed_indices:
                continue
                
            name_i = row_i.get('name', '').strip().lower()
            if not name_i:
                continue
            
            for j, row_j in discovery_df.iterrows():
                if j <= i or j in processed_indices:
                    continue
                    
                name_j = row_j.get('name', '').strip().lower()
                if not name_j:
                    continue
                
                similarity = self._calculate_name_similarity(name_i, name_j)
                if similarity >= 0.9:  # Very high threshold for discovery deduplication
                    # Keep the one with more complete data
                    completeness_i = self._calculate_record_completeness(row_i)
                    completeness_j = self._calculate_record_completeness(row_j)
                    
                    if completeness_i >= completeness_j:
                        similarity_duplicates.append(j)
                    else:
                        similarity_duplicates.append(i)
                        break  # Don't process this record further
                    
                    processed_indices.add(j)
        
        if similarity_duplicates:
            warnings.append(f"Removed {len(similarity_duplicates)} name similarity duplicates")
            discovery_df = discovery_df.drop(similarity_duplicates)
        
        # Convert back to Company objects
        from .models import Company
        deduplicated_companies = []
        
        for _, row in discovery_df.iterrows():
            try:
                if hasattr(discovery_companies[0], '__class__') and discovery_companies[0].__class__.__name__ == 'Company':
                    # Create Company object
                    company = Company.from_dict(row.to_dict())
                    deduplicated_companies.append(company)
                else:
                    # Keep as dictionary
                    deduplicated_companies.append(row.to_dict())
            except Exception as e:
                warnings.append(f"Error creating deduplicated company: {e}")
                continue
        
        logger.info(f"Deduplication complete: {len(deduplicated_companies)} unique companies")
        return deduplicated_companies, warnings
    
    def _calculate_record_completeness(self, record: pd.Series) -> int:
        """
        Calculate completeness score for a company record.
        
        Args:
            record: Company record as pandas Series
            
        Returns:
            Completeness score (higher is more complete)
        """
        score = 0
        
        # Check important fields
        important_fields = ['name', 'website', 'description', 'linkedin_url']
        for field in important_fields:
            value = record.get(field, '').strip()
            if value:
                score += 1
                # Bonus for longer descriptions
                if field == 'description' and len(value) > 50:
                    score += 1
        
        return score
    
    def _is_valid_url_format(self, url: str) -> bool:
        """Check if URL has valid format."""
        from urllib.parse import urlparse
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def _is_valid_timestamp_format(self, timestamp: str) -> bool:
        """Check if timestamp has valid ISO format with time component."""
        try:
            parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            # Check if it has time component (not just date)
            return 'T' in timestamp or ' ' in timestamp
        except (ValueError, AttributeError):
            return False
    
    def _check_data_completeness(self, df: pd.DataFrame) -> List[str]:
        """Check for data completeness issues."""
        issues = []
        
        if df.empty:
            return issues
        
        # Check for missing required fields
        required_fields = ['name', 'yc_page']
        for field in required_fields:
            if field in df.columns:
                empty_count = (df[field] == '').sum()
                if empty_count > 0:
                    issues.append(f"{empty_count} companies missing {field}")
        
        # Check for missing optional but important fields
        optional_fields = ['website', 'description', 'linkedin_url']
        for field in optional_fields:
            if field in df.columns:
                empty_count = (df[field] == '').sum()
                empty_percentage = (empty_count / len(df)) * 100
                if empty_percentage > 50:
                    issues.append(f"{empty_percentage:.1f}% of companies missing {field}")
        
        return issues
    
    def _detect_data_anomalies(self, df: pd.DataFrame) -> List[str]:
        """Detect potential data anomalies."""
        anomalies = []
        
        if df.empty:
            return anomalies
        
        # Check for suspiciously short company names
        if 'name' in df.columns:
            short_names = df[df['name'].str.len() < 3]
            if len(short_names) > 0:
                anomalies.append(f"{len(short_names)} companies with very short names (< 3 characters)")
        
        # Check for duplicate URLs
        url_fields = ['website', 'yc_page', 'linkedin_url']
        for field in url_fields:
            if field in df.columns:
                non_empty = df[df[field] != '']
                if len(non_empty) > 0:
                    duplicates = non_empty[non_empty[field].duplicated()]
                    if len(duplicates) > 0:
                        anomalies.append(f"{len(duplicates)} duplicate {field} URLs found")
        
        # Check for unusual timestamp patterns
        if 'last_updated' in df.columns:
            try:
                timestamps = pd.to_datetime(df['last_updated'])
                # Check if all timestamps are the same (suspicious)
                if timestamps.nunique() == 1 and len(df) > 1:
                    anomalies.append("All companies have identical timestamps")
                
                # Check for future timestamps
                future_timestamps = timestamps > datetime.now()
                if future_timestamps.any():
                    anomalies.append(f"{future_timestamps.sum()} companies have future timestamps")
                    
            except Exception:
                anomalies.append("Invalid timestamp formats detected")
        
        return anomalies