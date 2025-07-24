"""
Streamlit App for YC S25 Company Parser

Provides the web interface for viewing and refreshing company data.
"""

import streamlit as st
import pandas as pd
import traceback
from datetime import datetime
from typing import Optional, Tuple
from yc_parser.data_manager import DataManager
from yc_parser.company_processor import CompanyProcessor
from yc_parser.logging_config import setup_application_logging, get_logger, log_exception

# Initialize logging for the Streamlit app
try:
    setup_application_logging(environment="production", log_directory="logs")
    logger = get_logger(__name__)
    logger.info("Streamlit application started")
except Exception as e:
    # Fallback logging if setup fails
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to setup application logging: {e}")

# Global error tracking
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'last_error' not in st.session_state:
    st.session_state.last_error = None

def load_company_data() -> pd.DataFrame:
    """
    Load existing company data from CSV file with comprehensive error handling.
    
    Returns:
        DataFrame with company data, empty DataFrame if loading fails
    """
    try:
        logger.info("Loading company data from CSV file")
        data_manager = DataManager()
        df = data_manager.load_existing_data()
        
        if df.empty:
            logger.info("No existing company data found")
            return df
        
        logger.info(f"Successfully loaded {len(df)} companies from CSV")
        return df
        
    except FileNotFoundError:
        logger.warning("CSV file not found, starting with empty dataset")
        st.info("📄 No data file found. Click 'Refresh Data' to fetch companies for the first time.")
        return pd.DataFrame()
        
    except pd.errors.EmptyDataError:
        logger.warning("CSV file is empty")
        st.warning("📄 Data file is empty. Click 'Refresh Data' to fetch companies.")
        return pd.DataFrame()
        
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        st.error("📄 Data file appears to be corrupted. Try refreshing the data or check the logs.")
        _track_error("CSV parsing error", str(e))
        return pd.DataFrame()
        
    except PermissionError:
        logger.error("Permission denied accessing CSV file")
        st.error("🔒 Permission denied accessing data file. Check file permissions.")
        _track_error("Permission error", "Cannot access CSV file")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        log_exception(logger, "Failed to load company data")
        st.error(f"❌ Unexpected error loading data: {_get_user_friendly_error(e)}")
        _track_error("Data loading error", str(e))
        return pd.DataFrame()


def _track_error(error_type: str, error_message: str):
    """Track errors for debugging and user feedback."""
    st.session_state.error_count += 1
    st.session_state.last_error = {
        'type': error_type,
        'message': error_message,
        'timestamp': datetime.now().isoformat()
    }
    logger.error(f"Error tracked: {error_type} - {error_message}")


def _get_user_friendly_error(error: Exception) -> str:
    """Convert technical errors to user-friendly messages."""
    error_str = str(error).lower()
    
    if 'connection' in error_str or 'network' in error_str:
        return "Network connection issue. Please check your internet connection and try again."
    elif 'timeout' in error_str:
        return "Request timed out. The service might be temporarily unavailable."
    elif 'permission' in error_str or 'access' in error_str:
        return "Permission denied. Please check file permissions or try running as administrator."
    elif 'not found' in error_str or '404' in error_str:
        return "Resource not found. The requested data might have moved or been deleted."
    elif 'rate limit' in error_str or '429' in error_str:
        return "Too many requests. Please wait a moment before trying again."
    elif 'memory' in error_str:
        return "Insufficient memory. Try processing smaller batches of data."
    elif 'disk' in error_str or 'space' in error_str:
        return "Insufficient disk space. Please free up some space and try again."
    else:
        return f"Technical error: {str(error)[:100]}{'...' if len(str(error)) > 100 else ''}"


def _show_error_recovery_options():
    """Display error recovery options to the user."""
    if st.session_state.error_count > 0:
        with st.expander("🔧 Error Recovery Options", expanded=False):
            st.write("If you're experiencing issues, try these recovery options:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Clear Error State"):
                    st.session_state.error_count = 0
                    st.session_state.last_error = None
                    st.success("Error state cleared!")
                    st.rerun()
                
                if st.button("🧹 Validate Data"):
                    _validate_and_repair_data()
            
            with col2:
                if st.button("💾 Create Backup"):
                    _create_manual_backup()
                
                if st.button("📊 Check System Status"):
                    _show_system_status()


def _validate_and_repair_data():
    """Validate data integrity and attempt repairs."""
    try:
        with st.spinner("Validating data integrity..."):
            processor = CompanyProcessor()
            result = processor.validate_data_integrity()
            
            if result.success:
                st.success("✅ Data integrity check passed!")
                logger.info("Data integrity validation passed successfully")
                
                # Show additional validation details
                if result.total_companies_count > 0:
                    st.info(f"📊 Validated {result.total_companies_count} company records")
                    st.info(f"⏱️ Validation completed in {result.processing_time:.2f} seconds")
            else:
                st.warning(f"⚠️ Found {len(result.errors)} data integrity issues")
                logger.warning(f"Data integrity validation found {len(result.errors)} issues")
                
                # Show detailed error information
                with st.expander("🔍 View Integrity Issues", expanded=False):
                    for i, error in enumerate(result.errors[:10], 1):  # Show first 10 errors
                        st.caption(f"{i}. {error}")
                    if len(result.errors) > 10:
                        st.caption(f"... and {len(result.errors) - 10} more issues")
                
                if st.button("🔧 Attempt Automatic Repair"):
                    with st.spinner("Attempting to repair data..."):
                        try:
                            success, actions = processor.repair_data_issues()
                            
                            if success:
                                st.success("✅ Data repair completed!")
                                logger.info("Data repair completed successfully")
                                for action in actions:
                                    st.info(f"• {action}")
                                    logger.info(f"Repair action: {action}")
                                
                                # Re-validate after repair
                                st.info("🔄 Re-validating data after repair...")
                                revalidation_result = processor.validate_data_integrity()
                                if revalidation_result.success:
                                    st.success("✅ Data validation now passes!")
                                else:
                                    st.warning(f"⚠️ {len(revalidation_result.errors)} issues remain after repair")
                            else:
                                st.error("❌ Automatic repair failed")
                                logger.error("Data repair failed")
                                for action in actions:
                                    st.error(f"• {action}")
                                    logger.error(f"Repair failure: {action}")
                                
                                st.info("💡 Manual intervention may be required:")
                                st.info("• Check the logs directory for detailed error information")
                                st.info("• Consider restoring from a backup")
                                st.info("• Contact support if issues persist")
                                
                        except Exception as repair_error:
                            logger.error(f"Data repair process failed: {repair_error}")
                            log_exception(logger, "Data repair process encountered an error")
                            st.error(f"❌ Repair process failed: {_get_user_friendly_error(repair_error)}")
                            _track_error("Data repair failure", str(repair_error))
                            
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        log_exception(logger, "Data validation process failed")
        st.error(f"❌ Data validation failed: {_get_user_friendly_error(e)}")
        _track_error("Data validation failure", str(e))


def _create_manual_backup():
    """Create a manual backup of the current data."""
    try:
        with st.spinner("Creating backup..."):
            processor = CompanyProcessor()
            backup_name = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = processor.create_data_backup(backup_name)
            st.success(f"✅ Backup created: {backup_path}")
            
    except FileNotFoundError:
        st.info("ℹ️ No data file exists to backup")
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        st.error(f"❌ Backup creation failed: {_get_user_friendly_error(e)}")


def _show_system_status():
    """Display system status and diagnostics."""
    try:
        with st.spinner("Checking system status..."):
            processor = CompanyProcessor()
            stats = processor.get_processing_statistics()
            
            st.subheader("📊 System Status")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Companies", stats.get('total_companies', 0))
                st.metric("Companies with LinkedIn", stats.get('companies_with_linkedin', 0))
                st.metric("YC S25 Mentions", stats.get('yc_s25_mentions', 0))
            
            with col2:
                st.metric("Data Integrity Issues", stats.get('data_integrity_issues', 0))
                if stats.get('last_updated'):
                    st.info(f"Last Updated: {stats['last_updated']}")
                else:
                    st.info("Last Updated: Never")
                
                if stats.get('error'):
                    st.error(f"System Error: {stats['error']}")
            
            # Show resume status if available
            resume_status = processor.get_resume_status()
            if resume_status.get('has_resume_data'):
                st.warning("⚠️ Interrupted processing session detected")
                st.info(f"Interrupted at: {resume_status['interrupted_at']}")
                st.info(f"Processed: {resume_status['processed_count']} companies")
                
                if st.button("🔄 Clear Resume Data"):
                    processor.clear_resume_data()
                    st.success("Resume data cleared")
                    st.rerun()
                    
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        st.error(f"❌ System status check failed: {_get_user_friendly_error(e)}")


def format_url_as_link(url, display_text=None):
    """Format URL as clickable HTML link."""
    if pd.isna(url) or url == '' or str(url).lower() in ['nan', 'none']:
        return ""
    
    # Clean the URL
    url_str = str(url).strip()
    if not url_str:
        return ""
    
    # Add protocol if missing
    if not url_str.startswith(('http://', 'https://')):
        url_str = 'https://' + url_str
    
    # Use display text or truncated URL
    if display_text:
        link_text = display_text
    else:
        # Truncate long URLs for display
        link_text = url_str
        if len(link_text) > 40:
            link_text = link_text[:37] + "..."
    
    return f'<a href="{url_str}" target="_blank">{link_text}</a>'

def apply_filters_and_sorting(df: pd.DataFrame, search_term: str, linkedin_filter: str, sort_by: str) -> pd.DataFrame:
    """Apply filters and sorting to the dataframe."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply text search filter
    if search_term:
        search_term = search_term.lower()
        mask = (
            filtered_df['name'].str.lower().str.contains(search_term, na=False) |
            filtered_df['description'].str.lower().str.contains(search_term, na=False) |
            filtered_df['website'].str.lower().str.contains(search_term, na=False)
        )
        filtered_df = filtered_df[mask]
    
    # Apply LinkedIn filter
    if linkedin_filter == "Has LinkedIn":
        filtered_df = filtered_df[(filtered_df['linkedin_url'] != '') & (filtered_df['linkedin_url'].notna())]
    elif linkedin_filter == "No LinkedIn":
        filtered_df = filtered_df[(filtered_df['linkedin_url'] == '') | (filtered_df['linkedin_url'].isna())]
    elif linkedin_filter == "YC S25 Mention":
        filtered_df = filtered_df[filtered_df['yc_s25_on_linkedin'] == True]
    elif linkedin_filter == "No YC Mention":
        filtered_df = filtered_df[filtered_df['yc_s25_on_linkedin'] == False]
    
    # Apply sorting
    if sort_by == "Name (A-Z)":
        filtered_df = filtered_df.sort_values('name', ascending=True)
    elif sort_by == "Name (Z-A)":
        filtered_df = filtered_df.sort_values('name', ascending=False)
    elif sort_by == "YC Mentions First":
        filtered_df = filtered_df.sort_values(['yc_s25_on_linkedin', 'name'], ascending=[False, True])
    elif sort_by == "LinkedIn First":
        # Sort by whether they have LinkedIn URL, then by name
        filtered_df['has_linkedin'] = (filtered_df['linkedin_url'] != '') & (filtered_df['linkedin_url'].notna())
        filtered_df = filtered_df.sort_values(['has_linkedin', 'name'], ascending=[False, True])
        filtered_df = filtered_df.drop('has_linkedin', axis=1)
    elif sort_by == "Recently Updated":
        if 'last_updated' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('last_updated', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('name', ascending=True)
    
    return filtered_df


def display_company_table(df: pd.DataFrame, original_count: int = None):
    """Display company data in a formatted table with enhanced formatting."""
    if df.empty:
        st.info("No company data available. Click 'Refresh Data' to fetch companies.")
        return
    
    # Prepare data for display
    display_df = df.copy()
    
    # Format YC S25 LinkedIn flag as Yes/No
    display_df['yc_s25_on_linkedin'] = display_df['yc_s25_on_linkedin'].map({
        True: "Yes", 
        False: "No"
    })
    
    # Format URLs as clickable links
    display_df['website'] = display_df['website'].apply(
        lambda x: format_url_as_link(x, "Visit Website") if pd.notna(x) and str(x).strip() != '' else ""
    )
    
    display_df['yc_page'] = display_df['yc_page'].apply(
        lambda x: format_url_as_link(x, "YC Profile") if pd.notna(x) and str(x).strip() != '' else ""
    )
    
    display_df['linkedin_url'] = display_df['linkedin_url'].apply(
        lambda x: format_url_as_link(x, "LinkedIn") if pd.notna(x) and str(x).strip() != '' else ""
    )
    
    # Truncate long descriptions for better table display
    display_df['description'] = display_df['description'].apply(
        lambda x: str(x)[:100] + "..." if pd.notna(x) and len(str(x)) > 100 else str(x) if pd.notna(x) else ""
    )
    
    # Reorder columns for better display
    column_order = ['name', 'website', 'description', 'yc_page', 'linkedin_url', 'yc_s25_on_linkedin']
    display_df = display_df.reindex(columns=column_order)
    
    # Rename columns for better readability
    display_df.columns = [
        'Company Name', 
        'Website', 
        'Description', 
        'YC Profile', 
        'LinkedIn', 
        'YC S25 Mention'
    ]
    
    # Display the table with HTML rendering for clickable links
    st.markdown("### Company Directory")
    if original_count and len(display_df) != original_count:
        st.markdown(f"*Showing {len(display_df)} of {original_count} companies (filtered)*")
    else:
        st.markdown(f"*Showing {len(display_df)} companies*")
    
    # Convert to HTML for better link rendering
    html_table = display_df.to_html(escape=False, index=False, classes='streamlit-table')
    
    # Add custom CSS for better table styling (dark mode compatible)
    st.markdown("""
    <style>
    .streamlit-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
        color: var(--text-color);
        background-color: var(--background-color);
    }
    
    /* Light mode colors */
    [data-theme="light"] .streamlit-table {
        --text-color: #262730;
        --background-color: #ffffff;
        --header-bg: #f0f2f6;
        --border-color: #e0e0e0;
        --hover-bg: #f8f9fa;
        --link-color: #1f77b4;
    }
    
    /* Dark mode colors */
    [data-theme="dark"] .streamlit-table,
    .stApp[data-theme="dark"] .streamlit-table {
        --text-color: #fafafa;
        --background-color: #0e1117;
        --header-bg: #262730;
        --border-color: #3d4043;
        --hover-bg: #262730;
        --link-color: #58a6ff;
    }
    
    /* Auto-detect dark mode if no theme attribute */
    @media (prefers-color-scheme: dark) {
        .streamlit-table {
            --text-color: #fafafa;
            --background-color: #0e1117;
            --header-bg: #262730;
            --border-color: #3d4043;
            --hover-bg: #262730;
            --link-color: #58a6ff;
        }
    }
    
    .streamlit-table th {
        background-color: var(--header-bg);
        color: var(--text-color);
        padding: 12px 8px;
        text-align: left;
        font-weight: bold;
        border-bottom: 2px solid var(--border-color);
    }
    
    .streamlit-table td {
        padding: 10px 8px;
        border-bottom: 1px solid var(--border-color);
        vertical-align: top;
        color: var(--text-color);
        background-color: var(--background-color);
    }
    
    .streamlit-table tr:hover td {
        background-color: var(--hover-bg) !important;
        color: var(--text-color) !important;
    }
    
    .streamlit-table a {
        color: var(--link-color) !important;
        text-decoration: none;
        font-weight: 500;
    }
    
    .streamlit-table a:hover {
        text-decoration: underline;
        opacity: 0.8;
    }
    
    /* Ensure text is visible in all cases */
    .streamlit-table * {
        color: inherit !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the HTML table
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Show enhanced statistics
    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Companies", len(df))
    
    with col2:
        companies_with_linkedin = len(df[(df['linkedin_url'] != '') & (df['linkedin_url'].notna())])
        linkedin_percentage = (companies_with_linkedin / len(df) * 100) if len(df) > 0 else 0
        st.metric("With LinkedIn", companies_with_linkedin, f"{linkedin_percentage:.1f}%")
    
    with col3:
        yc_mentions = len(df[df['yc_s25_on_linkedin'] == True])
        mention_percentage = (yc_mentions / len(df) * 100) if len(df) > 0 else 0
        st.metric("YC S25 Mentions", yc_mentions, f"{mention_percentage:.1f}%")
    
    with col4:
        companies_with_websites = len(df[(df['website'] != '') & (df['website'].notna())])
        website_percentage = (companies_with_websites / len(df) * 100) if len(df) > 0 else 0
        st.metric("With Websites", companies_with_websites, f"{website_percentage:.1f}%")
    
    # Additional insights
    if not df.empty:
        st.markdown("### 🔍 Quick Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            if yc_mentions > 0:
                mention_rate = (yc_mentions / companies_with_linkedin * 100) if companies_with_linkedin > 0 else 0
                st.info(f"📈 **{mention_rate:.1f}%** of companies with LinkedIn mention YC S25")
            
            if companies_with_websites > 0:
                st.info(f"🌐 **{website_percentage:.1f}%** of companies have websites listed")
        
        with insight_col2:
            # Show top companies with YC mentions (if any)
            yc_mention_companies = df[df['yc_s25_on_linkedin'] == True]
            if len(yc_mention_companies) > 0:
                st.success(f"✅ **{len(yc_mention_companies)}** companies actively promote their YC S25 status")
            
            # Data completeness
            complete_profiles = len(df[
                (df['website'] != '') & (df['website'].notna()) &
                (df['linkedin_url'] != '') & (df['linkedin_url'].notna()) &
                (df['description'] != '') & (df['description'].notna())
            ])
            completeness_rate = (complete_profiles / len(df) * 100) if len(df) > 0 else 0
            st.info(f"📋 **{completeness_rate:.1f}%** have complete profiles")

def refresh_data(season: str = "Summer", year: int = 2025) -> bool:
    """
    Refresh company data by processing new companies with comprehensive error handling.
    
    Args:
        season: YC batch season ("Summer" or "Winter")
        year: YC batch year (e.g., 2025, 2024, etc.)
    
    Returns:
        True if refresh was successful, False otherwise
    """
    progress_bar = None
    status_text = None
    
    try:
        batch_code = f"{'S' if season == 'Summer' else 'W'}{str(year)[2:]}"
        logger.info(f"Starting data refresh process for {batch_code} ({season} {year})")
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize processor
        status_text.text(f"🔧 Initializing data processor for {batch_code}...")
        progress_bar.progress(10)
        
        try:
            # Generate dynamic CSV filename based on batch
            csv_filename = f"yc_{batch_code.lower()}_companies.csv"
            processor = CompanyProcessor(
                csv_file_path=csv_filename,
                season=season,
                year=year
            )
            logger.info(f"CompanyProcessor initialized successfully for {batch_code}")
        except Exception as e:
            logger.error(f"Failed to initialize CompanyProcessor: {e}")
            raise Exception(f"Failed to initialize data processor: {_get_user_friendly_error(e)}")
        
        # Step 2: Check for resume data
        status_text.text("🔍 Checking for interrupted sessions...")
        progress_bar.progress(20)
        
        resume_status = processor.get_resume_status()
        if resume_status.get('has_resume_data'):
            st.info(f"📋 Resuming interrupted session from {resume_status['interrupted_at']}")
            logger.info(f"Resuming processing from interrupted session")
        
        # Step 3: Process new companies
        status_text.text("🚀 Fetching and processing new companies...")
        progress_bar.progress(30)
        
        try:
            result = processor.process_new_companies()
            logger.info(f"Processing completed: success={result.success}, new_companies={result.new_companies_count}")
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            st.warning("⚠️ Processing was interrupted. You can resume later.")
            return False
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            log_exception(logger, "Company processing failed")
            raise Exception(f"Data processing failed: {_get_user_friendly_error(e)}")
        
        # Step 4: Complete processing
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Display results with enhanced error handling
        if result.success:
            if result.new_companies_count > 0:
                st.success(f"✅ Successfully processed {result.new_companies_count} new companies!")
                st.info(f"📊 Total companies in dataset: {result.total_companies_count}")
                st.info(f"⏱️ Processing time: {result.processing_time:.1f} seconds")
                
                # Show LinkedIn mention statistics
                try:
                    data_manager = DataManager()
                    df = data_manager.load_existing_data()
                    if not df.empty:
                        yc_mentions = len(df[df['yc_s25_on_linkedin'] == True])
                        if yc_mentions > 0:
                            st.info(f"🎯 Found {yc_mentions} companies with YC S25 LinkedIn mentions!")
                except Exception as e:
                    logger.warning(f"Failed to calculate LinkedIn statistics: {e}")
                
                # Show warnings if any
                if result.errors:
                    st.warning(f"⚠️ {len(result.errors)} warnings occurred during processing:")
                    for error in result.errors[:3]:  # Show first 3 errors
                        st.caption(f"• {error}")
                    if len(result.errors) > 3:
                        st.caption(f"• ... and {len(result.errors) - 3} more warnings")
                        
                    # Log all errors for debugging
                    for error in result.errors:
                        logger.warning(f"Processing warning: {error}")
            else:
                st.info("ℹ️ No new companies found. Dataset is up to date!")
                st.info(f"📊 Total companies in dataset: {result.total_companies_count}")
                logger.info("No new companies found during refresh")
        else:
            st.error("❌ Data refresh failed!")
            _track_error("Data refresh failed", f"Processing returned failure with {len(result.errors)} errors")
            
            if result.errors:
                st.error("🔍 Errors encountered:")
                for error in result.errors[:5]:  # Show first 5 errors
                    st.caption(f"• {error}")
                if len(result.errors) > 5:
                    st.caption(f"• ... and {len(result.errors) - 5} more errors")
                
                # Log all errors
                for error in result.errors:
                    logger.error(f"Processing error: {error}")
            
            # Suggest recovery actions
            st.info("💡 Try these recovery options:")
            st.info("• Check your internet connection")
            st.info("• Wait a few minutes and try again")
            st.info("• Use the Error Recovery Options below")
        
        logger.info(f"Data refresh completed: success={result.success}")
        return result.success
        
    except KeyboardInterrupt:
        logger.warning("Data refresh interrupted by user")
        st.warning("⚠️ Data refresh was interrupted")
        return False
        
    except Exception as e:
        error_msg = f"Unexpected error during data refresh: {e}"
        logger.error(error_msg)
        log_exception(logger, "Data refresh failed with unexpected error")
        
        st.error(f"❌ {_get_user_friendly_error(e)}")
        _track_error("Data refresh error", str(e))
        
        # Provide specific recovery suggestions based on error type
        error_str = str(e).lower()
        if 'network' in error_str or 'connection' in error_str:
            st.info("🌐 Network issue detected. Please check your internet connection and try again.")
        elif 'timeout' in error_str:
            st.info("⏰ Request timed out. The service might be busy. Try again in a few minutes.")
        elif 'rate limit' in error_str or '429' in error_str:
            st.info("🚦 Rate limit reached. Please wait a few minutes before trying again.")
        else:
            st.info("🔧 Use the Error Recovery Options below to diagnose and fix the issue.")
        
        return False
        
    finally:
        # Clean up progress indicators
        if progress_bar is not None:
            progress_bar.empty()
        if status_text is not None:
            status_text.empty()

def main():
    st.set_page_config(
        page_title="YC Company Parser",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': 'https://github.com/ivrejchik/YC-Scraper',
            'Report a bug': 'https://github.com/ivrejchik/YC-Scraper/issues',
            'About': """
            # YC Company Parser
            
            A comprehensive tool to fetch and analyze Y Combinator companies 
            with advanced filtering, sorting, and LinkedIn integration.
            
            **GitHub**: https://github.com/ivrejchik/YC-Scraper
            """
        }
    )
    
    st.title("🚀 YC Company Parser")
    st.markdown("**Y Combinator Company Dashboard**")
    
    # Add deployment info (safe for both local and cloud)
    # Only show cloud info if we're actually on Streamlit Cloud
    import os
    if os.getenv('STREAMLIT_SHARING') or os.getenv('STREAMLIT_CLOUD'):
        st.info("🌐 **Live on Streamlit Cloud** - Analyze YC companies from any batch!")
    
    st.markdown("---")
    
    # Season and Year Selection
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        season = st.selectbox(
            "📅 Season",
            options=["Summer", "Winter"],
            index=0,  # Default to Summer
            help="Select YC batch season"
        )
    
    with col2:
        current_year = datetime.now().year
        year = st.selectbox(
            "📆 Year", 
            options=list(range(current_year + 1, 2019, -1)),  # 2025 down to 2020
            index=0,  # Default to current/next year
            help="Select YC batch year"
        )
    
    with col3:
        batch_code = f"{'S' if season == 'Summer' else 'W'}{str(year)[2:]}"
        st.info(f"**Selected Batch:** {batch_code}")
    
    with col4:
        if st.button("🔄 Refresh Data", type="primary"):
            with st.spinner("Processing..."):
                success = refresh_data(season=season, year=year)
                if success:
                    st.rerun()
    
    st.caption(f"Fetching data for {season} {year} batch ({batch_code})")
    
    # Note about current API limitation
    if batch_code != "S25":
        st.warning(f"⚠️ **Note**: Currently only S25 (Summer 2025) data is available. The API endpoint for {batch_code} may not exist yet.")
        st.info("💡 The app will attempt to fetch data for the selected batch, but may fall back to available data.")
    
    # Show error recovery options if there have been errors
    _show_error_recovery_options()
    
    st.markdown("---")
    
    # Load and display data
    st.subheader("Company Data")
    
    # Load data with error handling
    try:
        df = load_company_data()
        
        if not df.empty:
            # Filtering and Sorting Controls
            st.markdown("### 🔍 Filters & Controls")
            
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 2])
            
            with filter_col1:
                # Text search filter
                search_term = st.text_input(
                    "🔍 Search Companies",
                    placeholder="Search by name, description, or website...",
                    help="Search across company name, description, and website"
                )
            
            with filter_col2:
                # LinkedIn filter
                linkedin_filter = st.selectbox(
                    "🔗 LinkedIn Status",
                    options=["All", "Has LinkedIn", "No LinkedIn", "YC S25 Mention", "No YC Mention"],
                    help="Filter by LinkedIn presence and YC mentions"
                )
            
            with filter_col3:
                # Sort options
                sort_by = st.selectbox(
                    "📊 Sort By",
                    options=["Name (A-Z)", "Name (Z-A)", "YC Mentions First", "LinkedIn First", "Recently Updated"],
                    help="Choose how to sort the companies"
                )
            
            with filter_col4:
                # Download button
                filtered_df = apply_filters_and_sorting(df, search_term, linkedin_filter, sort_by)
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"yc_{batch_code.lower()}_companies_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download filtered data as CSV file"
                )
            
            st.markdown("---")
            
            # Apply filters and display table
            filtered_df = apply_filters_and_sorting(df, search_term, linkedin_filter, sort_by)
            display_company_table(filtered_df, original_count=len(df))
            
        else:
            st.info("No company data available. Click 'Refresh Data' to fetch companies.")
        
        # Show last updated information if data exists
        if not df.empty and 'last_updated' in df.columns:
            try:
                last_updated = pd.to_datetime(df['last_updated']).max()
                st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                logger.warning(f"Failed to parse last_updated timestamp: {e}")
                st.caption("Last updated: Unknown")
        
        # Show error summary if there have been errors
        if st.session_state.error_count > 0:
            st.markdown("---")
            with st.expander("⚠️ Error Summary", expanded=False):
                st.warning(f"Total errors encountered: {st.session_state.error_count}")
                if st.session_state.last_error:
                    st.info(f"Last error: {st.session_state.last_error['type']} at {st.session_state.last_error['timestamp']}")
                    st.caption(f"Details: {st.session_state.last_error['message']}")
                
                st.info("💡 If you continue to experience issues:")
                st.info("• Check the logs directory for detailed error information")
                st.info("• Try the Error Recovery Options above")
                st.info("• Restart the application if problems persist")
    
    except Exception as e:
        logger.error(f"Critical error in main application: {e}")
        log_exception(logger, "Critical application error")
        st.error("❌ Critical application error occurred")
        st.error(f"Error: {_get_user_friendly_error(e)}")
        
        # Show emergency recovery options
        st.markdown("### 🚨 Emergency Recovery")
        st.info("The application encountered a critical error. Try these recovery steps:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart Application"):
                st.session_state.clear()
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All Data"):
                try:
                    st.session_state.clear()
                    st.success("Session data cleared")
                except:
                    st.error("Failed to clear session data")
        
        with col3:
            if st.button("📋 Show Debug Info"):
                st.code(f"Error: {str(e)}")
                st.code(f"Traceback: {traceback.format_exc()}")
        
        _track_error("Critical application error", str(e))

if __name__ == "__main__":
    main()