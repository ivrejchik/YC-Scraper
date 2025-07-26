#!/usr/bin/env python3
"""
Command Line Interface for YC Company Parser

Provides command-line access to data collection and processing functionality
outside of the Streamlit interface.
"""

import argparse
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .company_processor import CompanyProcessor
from .data_manager import DataManager
from .config import load_config
from .logging_config import setup_application_logging, get_logger
from .health_check import run_health_check_cli
from .linkedin_discovery import LinkedInDiscoveryService


def setup_cli_logging(config: Dict[str, Any], verbose: bool = False) -> None:
    """Setup logging for CLI operations."""
    log_directory = config.get('logging', {}).get('log_directory', 'logs')
    
    setup_application_logging(
        environment=config.get('app', {}).get('environment', 'production'),
        log_directory=log_directory
    )
    
    # Set log level after setup if verbose
    if verbose:
        logger = get_logger(__name__)
        logger.setLevel("DEBUG")
        # Also set root logger level
        logging.getLogger().setLevel("DEBUG")


def process_companies(args) -> int:
    """Process new companies and update the dataset."""
    config = load_config()
    setup_cli_logging(config, args.verbose)
    logger = get_logger(__name__)
    
    # Get batch parameters
    if hasattr(args, 'batch') and args.batch:
        from .config import parse_batch_code, format_config_paths
        try:
            season, year = parse_batch_code(args.batch)
            batch_code = args.batch.upper()
            config = format_config_paths(config, batch_code)
        except ValueError as e:
            print(f"❌ Invalid batch code '{args.batch}': {e}")
            return 1
    else:
        # Use default batch from config
        season = config.get('batch', {}).get('default_season', 'Summer')
        year = config.get('batch', {}).get('default_year', 2025)
        batch_code = config.get('batch', {}).get('default_batch_code', 'S25')
        config = format_config_paths(config, batch_code)
    
    logger.info(f"Starting CLI company processing for {season} {year} ({batch_code})")
    
    try:
        processor = CompanyProcessor(
            csv_file_path=config['data']['csv_file_path'],
            season=season,
            year=year
        )
        
        # Show processing configuration if verbose
        if args.verbose:
            print(f"Configuration loaded from: {config.get('_config_path', 'default')}")
            print(f"Max workers: {config.get('processing', {}).get('max_workers', 3)}")
            print(f"Batch size: {config.get('processing', {}).get('batch_size', 10)}")
            print(f"LinkedIn delay: {config.get('rate_limiting', {}).get('linkedin_delay_min', 1.0)}-{config.get('rate_limiting', {}).get('linkedin_delay_max', 2.0)}s")
            print()
        
        # Check for resume data
        resume_status = processor.get_resume_status()
        if resume_status.get('has_resume_data'):
            print(f"⚠️  Interrupted session detected from {resume_status['interrupted_at']}")
            print(f"   Processed: {resume_status['processed_count']} companies")
            
            if not args.force:
                response = input("Resume interrupted session? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    processor.clear_resume_data()
                    print("Resume data cleared. Starting fresh processing.")
            print()
        
        # Process companies
        print("🚀 Processing companies...")
        start_time = time.time()
        
        result = processor.process_new_companies()
        
        processing_time = time.time() - start_time
        
        # Display results
        if result.success:
            print(f"✅ Processing completed successfully!")
            print(f"   New companies processed: {result.new_companies_count}")
            print(f"   Total companies in dataset: {result.total_companies_count}")
            print(f"   Processing time: {processing_time:.1f} seconds")
            
            if result.errors:
                print(f"⚠️  {len(result.errors)} warnings occurred:")
                for error in result.errors[:5]:  # Show first 5 errors
                    print(f"   • {error}")
                if len(result.errors) > 5:
                    print(f"   • ... and {len(result.errors) - 5} more warnings")
            
            logger.info(f"CLI processing completed successfully: {result.new_companies_count} new companies")
            return 0
        else:
            print(f"❌ Processing failed!")
            if result.errors:
                print("Errors:")
                for error in result.errors:
                    print(f"   • {error}")
            
            logger.error(f"CLI processing failed with {len(result.errors)} errors")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        logger.warning("CLI processing interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"CLI processing failed with unexpected error: {e}")
        return 1


def validate_data(args) -> int:
    """Validate data integrity and optionally repair issues."""
    config = load_config()
    setup_cli_logging(config, args.verbose)
    logger = get_logger(__name__)
    
    # Get batch parameters
    if hasattr(args, 'batch') and args.batch:
        from .config import parse_batch_code, format_config_paths
        try:
            season, year = parse_batch_code(args.batch)
            batch_code = args.batch.upper()
            config = format_config_paths(config, batch_code)
        except ValueError as e:
            print(f"❌ Invalid batch code '{args.batch}': {e}")
            return 1
    else:
        # Use default batch from config
        season = config.get('batch', {}).get('default_season', 'Summer')
        year = config.get('batch', {}).get('default_year', 2025)
        batch_code = config.get('batch', {}).get('default_batch_code', 'S25')
        config = format_config_paths(config, batch_code)
    
    logger.info(f"Starting CLI data validation for {season} {year} ({batch_code})")
    
    try:
        processor = CompanyProcessor(
            csv_file_path=config['data']['csv_file_path'],
            season=season,
            year=year
        )
        
        print("🔍 Validating data integrity...")
        result = processor.validate_data_integrity()
        
        if result.success:
            print(f"✅ Data integrity check passed!")
            print(f"   Validated {result.total_companies_count} company records")
            print(f"   Validation time: {result.processing_time:.2f} seconds")
            
            logger.info(f"Data validation passed: {result.total_companies_count} records")
            return 0
        else:
            print(f"⚠️  Found {len(result.errors)} data integrity issues:")
            for i, error in enumerate(result.errors[:10], 1):
                print(f"   {i}. {error}")
            if len(result.errors) > 10:
                print(f"   ... and {len(result.errors) - 10} more issues")
            
            if args.repair:
                print("\n🔧 Attempting automatic repair...")
                success, actions = processor.repair_data_issues()
                
                if success:
                    print("✅ Data repair completed!")
                    for action in actions:
                        print(f"   • {action}")
                    
                    # Re-validate after repair
                    print("\n🔄 Re-validating data after repair...")
                    revalidation_result = processor.validate_data_integrity()
                    if revalidation_result.success:
                        print("✅ Data validation now passes!")
                        logger.info("Data repair and revalidation successful")
                        return 0
                    else:
                        print(f"⚠️  {len(revalidation_result.errors)} issues remain after repair")
                        logger.warning(f"Data repair incomplete: {len(revalidation_result.errors)} issues remain")
                        return 1
                else:
                    print("❌ Automatic repair failed:")
                    for action in actions:
                        print(f"   • {action}")
                    logger.error("Data repair failed")
                    return 1
            else:
                print("\nUse --repair to attempt automatic repair of these issues.")
                logger.warning(f"Data validation found {len(result.errors)} issues")
                return 1
                
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        logger.error(f"CLI data validation failed: {e}")
        return 1


def show_stats(args) -> int:
    """Display dataset statistics and system status."""
    config = load_config()
    setup_cli_logging(config, args.verbose)
    logger = get_logger(__name__)
    
    # Get batch parameters
    if hasattr(args, 'batch') and args.batch:
        from .config import parse_batch_code, format_config_paths
        try:
            season, year = parse_batch_code(args.batch)
            batch_code = args.batch.upper()
            config = format_config_paths(config, batch_code)
        except ValueError as e:
            print(f"❌ Invalid batch code '{args.batch}': {e}")
            return 1
    else:
        # Use default batch from config
        season = config.get('batch', {}).get('default_season', 'Summer')
        year = config.get('batch', {}).get('default_year', 2025)
        batch_code = config.get('batch', {}).get('default_batch_code', 'S25')
        config = format_config_paths(config, batch_code)
    
    try:
        data_manager = DataManager(config['data']['csv_file_path'])
        processor = CompanyProcessor(
            csv_file_path=config['data']['csv_file_path'],
            season=season,
            year=year
        )
        
        # Load data
        df = data_manager.load_existing_data()
        
        if df.empty:
            print("📄 No data available. Run 'process' command to fetch companies.")
            return 0
        
        # Basic statistics
        print("📊 Dataset Statistics")
        print("=" * 50)
        print(f"Total companies: {len(df)}")
        
        # LinkedIn statistics
        companies_with_linkedin = len(df[(df['linkedin_url'] != '') & (df['linkedin_url'].notna())])
        linkedin_percentage = (companies_with_linkedin / len(df) * 100) if len(df) > 0 else 0
        print(f"Companies with LinkedIn: {companies_with_linkedin} ({linkedin_percentage:.1f}%)")
        
        # YC batch mentions (use dynamic field name)
        yc_field = 'yc_batch_on_linkedin' if 'yc_batch_on_linkedin' in df.columns else 'yc_s25_on_linkedin'
        yc_mentions = len(df[df[yc_field] == True]) if yc_field in df.columns else 0
        mention_percentage = (yc_mentions / len(df) * 100) if len(df) > 0 else 0
        print(f"YC {batch_code} LinkedIn mentions: {yc_mentions} ({mention_percentage:.1f}%)")
        
        # Website statistics
        companies_with_websites = len(df[(df['website'] != '') & (df['website'].notna())])
        website_percentage = (companies_with_websites / len(df) * 100) if len(df) > 0 else 0
        print(f"Companies with websites: {companies_with_websites} ({website_percentage:.1f}%)")
        
        # Data completeness
        complete_profiles = len(df[
            (df['website'] != '') & (df['website'].notna()) &
            (df['linkedin_url'] != '') & (df['linkedin_url'].notna()) &
            (df['description'] != '') & (df['description'].notna())
        ])
        completeness_rate = (complete_profiles / len(df) * 100) if len(df) > 0 else 0
        print(f"Complete profiles: {complete_profiles} ({completeness_rate:.1f}%)")
        
        # Last updated
        if 'last_updated' in df.columns:
            try:
                last_updated = df['last_updated'].max()
                print(f"Last updated: {last_updated}")
            except:
                print("Last updated: Unknown")
        
        # System status
        if args.verbose:
            print("\n🔧 System Status")
            print("=" * 50)
            
            stats = processor.get_processing_statistics()
            for key, value in stats.items():
                print(f"{key}: {value}")
            
            # Resume status
            resume_status = processor.get_resume_status()
            if resume_status.get('has_resume_data'):
                print(f"\n⚠️  Interrupted session detected:")
                print(f"   Interrupted at: {resume_status['interrupted_at']}")
                print(f"   Processed: {resume_status['processed_count']} companies")
        
        logger.info(f"Stats displayed: {len(df)} companies")
        return 0
        
    except Exception as e:
        print(f"❌ Failed to display statistics: {e}")
        logger.error(f"CLI stats failed: {e}")
        return 1


def backup_data(args) -> int:
    """Create a backup of the current dataset."""
    config = load_config()
    setup_cli_logging(config, args.verbose)
    logger = get_logger(__name__)
    
    # Get batch parameters
    if hasattr(args, 'batch') and args.batch:
        from .config import parse_batch_code, format_config_paths
        try:
            season, year = parse_batch_code(args.batch)
            batch_code = args.batch.upper()
            config = format_config_paths(config, batch_code)
        except ValueError as e:
            print(f"❌ Invalid batch code '{args.batch}': {e}")
            return 1
    else:
        # Use default batch from config
        season = config.get('batch', {}).get('default_season', 'Summer')
        year = config.get('batch', {}).get('default_year', 2025)
        batch_code = config.get('batch', {}).get('default_batch_code', 'S25')
        config = format_config_paths(config, batch_code)
    
    try:
        processor = CompanyProcessor(
            csv_file_path=config['data']['csv_file_path'],
            season=season,
            year=year
        )
        
        # Generate backup name
        if args.name:
            backup_name = args.name
        else:
            backup_name = f"cli_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"💾 Creating backup: {backup_name}")
        backup_path = processor.create_data_backup(backup_name)
        
        print(f"✅ Backup created successfully: {backup_path}")
        logger.info(f"Backup created: {backup_path}")
        return 0
        
    except FileNotFoundError:
        print("📄 No data file exists to backup")
        return 0
        
    except Exception as e:
        print(f"❌ Backup creation failed: {e}")
        logger.error(f"CLI backup failed: {e}")
        return 1


def clear_resume(args) -> int:
    """Clear resume data from interrupted processing sessions."""
    config = load_config()
    setup_cli_logging(config, args.verbose)
    logger = get_logger(__name__)
    
    # Get batch parameters
    if hasattr(args, 'batch') and args.batch:
        from .config import parse_batch_code, format_config_paths
        try:
            season, year = parse_batch_code(args.batch)
            batch_code = args.batch.upper()
            config = format_config_paths(config, batch_code)
        except ValueError as e:
            print(f"❌ Invalid batch code '{args.batch}': {e}")
            return 1
    else:
        # Use default batch from config
        season = config.get('batch', {}).get('default_season', 'Summer')
        year = config.get('batch', {}).get('default_year', 2025)
        batch_code = config.get('batch', {}).get('default_batch_code', 'S25')
        config = format_config_paths(config, batch_code)
    
    try:
        processor = CompanyProcessor(
            csv_file_path=config['data']['csv_file_path'],
            season=season,
            year=year
        )
        
        resume_status = processor.get_resume_status()
        if not resume_status.get('has_resume_data'):
            print("ℹ️  No resume data found")
            return 0
        
        print(f"🗑️  Clearing resume data from {resume_status['interrupted_at']}")
        processor.clear_resume_data()
        
        print("✅ Resume data cleared successfully")
        logger.info("Resume data cleared via CLI")
        return 0
        
    except Exception as e:
        print(f"❌ Failed to clear resume data: {e}")
        logger.error(f"CLI clear resume failed: {e}")
        return 1


def discover_linkedin(args) -> int:
    """Discover LinkedIn profiles for YC companies and merge with existing data."""
    config = load_config()
    setup_cli_logging(config, args.verbose)
    logger = get_logger(__name__)
    
    logger.info(f"Starting CLI LinkedIn discovery for batch {args.batch}")
    
    try:
        # Override configuration with command-line options if provided
        if hasattr(args, 'search_delay') and args.search_delay:
            config['linkedin_discovery']['search_delay_min'] = args.search_delay
            config['linkedin_discovery']['search_delay_max'] = args.search_delay + 1.0
        
        if hasattr(args, 'max_results') and args.max_results:
            config['linkedin_discovery']['max_results_per_search'] = args.max_results
        
        # Show discovery configuration if verbose
        if args.verbose:
            print(f"Configuration loaded from: {config.get('_config_path', 'default')}")
            print(f"Batch: {args.batch}")
            print(f"Search delay: {config.get('linkedin_discovery', {}).get('search_delay_min', 2.0)}-{config.get('linkedin_discovery', {}).get('search_delay_max', 4.0)}s")
            print(f"Max results per search: {config.get('linkedin_discovery', {}).get('max_results_per_search', 10)}")
            print(f"Dry run: {args.dry_run}")
            print()
        
        # Initialize discovery service
        discovery_service = LinkedInDiscoveryService(config)
        
        # Show progress header
        print(f"🔍 Discovering LinkedIn profiles for YC {args.batch}...")
        print("=" * 60)
        
        # Track progress with real-time updates
        start_time = time.time()
        
        # Perform discovery
        result = discovery_service.discover_and_merge(args.batch)
        
        processing_time = time.time() - start_time
        
        # Display results
        if result.get('success'):
            print(f"\n✅ LinkedIn discovery completed successfully!")
            print(f"   Batch processed: YC {result['batch']}")
            print(f"   YC companies found: {result['yc_companies_count']}")
            print(f"   LinkedIn profiles discovered: {result['linkedin_results_count']}")
            print(f"   Total merged companies: {result['merged_companies_count']}")
            print(f"   Processing time: {processing_time:.1f} seconds")
            
            if not args.dry_run:
                # Save results to CSV
                data_manager = DataManager()
                
                # Create backup before saving
                backup_name = f"pre_discovery_{args.batch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    backup_path = data_manager.create_backup(backup_name)
                    print(f"   Backup created: {backup_path}")
                except Exception as e:
                    print(f"   ⚠️  Backup creation failed: {e}")
                
                # Save merged data
                try:
                    companies = result['companies']
                    save_result = data_manager.append_companies(companies)
                    
                    if save_result.get('success'):
                        print(f"   ✅ Data saved successfully!")
                        print(f"   New companies added: {save_result.get('new_companies_count', 0)}")
                        print(f"   Total companies in dataset: {save_result.get('total_companies_count', 0)}")
                    else:
                        print(f"   ❌ Failed to save data: {save_result.get('error', 'Unknown error')}")
                        return 1
                        
                except Exception as e:
                    print(f"   ❌ Error saving data: {e}")
                    logger.error(f"CLI discovery data save failed: {e}")
                    return 1
            else:
                print(f"   🔍 Dry run completed - no data was saved")
            
            logger.info(f"CLI LinkedIn discovery completed successfully: {result['merged_companies_count']} companies")
            return 0
            
        else:
            print(f"❌ LinkedIn discovery failed!")
            error_msg = result.get('error', 'Unknown error')
            failed_step = result.get('step_name', 'unknown step')
            print(f"   Error: {error_msg}")
            print(f"   Failed at: {failed_step}")
            print(f"   Processing time: {processing_time:.1f} seconds")
            
            logger.error(f"CLI LinkedIn discovery failed: {error_msg}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Discovery interrupted by user")
        logger.warning("CLI LinkedIn discovery interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"❌ Unexpected error during discovery: {e}")
        logger.error(f"CLI LinkedIn discovery failed with unexpected error: {e}")
        return 1


def health_check(args) -> int:
    """Perform system health checks."""
    config = load_config()
    setup_cli_logging(config, args.verbose)
    logger = get_logger(__name__)
    
    try:
        print("🏥 Performing health checks...")
        
        health_checks = []
        
        # Check data file exists
        data_manager = DataManager()
        csv_path = Path(config.get('data', {}).get('csv_file_path', 'yc_s25_companies.csv'))
        if csv_path.exists():
            health_checks.append(("Data file exists", True, str(csv_path)))
        else:
            health_checks.append(("Data file exists", False, f"File not found: {csv_path}"))
        
        # Check CSV readable
        try:
            df = data_manager.load_existing_data()
            health_checks.append(("CSV readable", True, f"{len(df)} records loaded"))
        except Exception as e:
            health_checks.append(("CSV readable", False, str(e)))
        
        # Check log directory writable
        log_dir = Path(config.get('logging', {}).get('log_directory', 'logs'))
        try:
            log_dir.mkdir(exist_ok=True)
            test_file = log_dir / "health_check_test.tmp"
            test_file.write_text("test")
            test_file.unlink()
            health_checks.append(("Log directory writable", True, str(log_dir)))
        except Exception as e:
            health_checks.append(("Log directory writable", False, str(e)))
        
        # Check network connectivity
        try:
            import requests
            response = requests.get("https://www.ycombinator.com", timeout=10)
            if response.status_code == 200:
                health_checks.append(("Network connectivity", True, "YC website accessible"))
            else:
                health_checks.append(("Network connectivity", False, f"HTTP {response.status_code}"))
        except Exception as e:
            health_checks.append(("Network connectivity", False, str(e)))
        
        # Display results
        all_passed = True
        for check_name, passed, details in health_checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}: {details}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All health checks passed!")
            logger.info("All health checks passed")
            return 0
        else:
            print("\n⚠️  Some health checks failed")
            logger.warning("Some health checks failed")
            return 1
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        logger.error(f"CLI health check failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YC S25 Company Parser - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process                    # Process new companies
  %(prog)s process --verbose          # Process with detailed output
  %(prog)s validate --repair          # Validate and repair data
  %(prog)s stats                      # Show dataset statistics
  %(prog)s backup --name my_backup    # Create named backup
  %(prog)s discover --batch S25       # Discover LinkedIn profiles for YC S25
  %(prog)s discover --dry-run         # Test discovery without saving
  %(prog)s health                     # Run health checks
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process new companies')
    process_parser.add_argument(
        '--batch', '-b',
        type=str,
        help='YC batch to process (e.g., S25, W25, S24, W24). Defaults to config default.'
    )
    process_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force processing without prompting for resume'
    )
    process_parser.set_defaults(func=process_companies)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data integrity')
    validate_parser.add_argument(
        '--batch', '-b',
        type=str,
        help='YC batch to validate (e.g., S25, W25, S24, W24). Defaults to config default.'
    )
    validate_parser.add_argument(
        '--repair', '-r',
        action='store_true',
        help='Attempt to repair data issues automatically'
    )
    validate_parser.set_defaults(func=validate_data)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument(
        '--batch', '-b',
        type=str,
        help='YC batch to show stats for (e.g., S25, W25, S24, W24). Defaults to config default.'
    )
    stats_parser.set_defaults(func=show_stats)
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create data backup')
    backup_parser.add_argument(
        '--batch', '-b',
        type=str,
        help='YC batch to backup (e.g., S25, W25, S24, W24). Defaults to config default.'
    )
    backup_parser.add_argument(
        '--name', '-n',
        type=str,
        help='Backup name (default: auto-generated)'
    )
    backup_parser.set_defaults(func=backup_data)
    
    # Clear resume command
    clear_parser = subparsers.add_parser('clear-resume', help='Clear resume data')
    clear_parser.add_argument(
        '--batch', '-b',
        type=str,
        help='YC batch to clear resume data for (e.g., S25, W25, S24, W24). Defaults to config default.'
    )
    clear_parser.set_defaults(func=clear_resume)
    
    # Discovery command
    discovery_parser = subparsers.add_parser('discover', help='Discover LinkedIn profiles for YC companies')
    discovery_parser.add_argument(
        '--batch', '-b',
        type=str,
        default='S25',
        help='YC batch to process (e.g., S25, W25, S24, W24)'
    )
    discovery_parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Perform discovery without saving results'
    )
    discovery_parser.add_argument(
        '--search-delay',
        type=float,
        help='Override search delay between requests (seconds)'
    )
    discovery_parser.add_argument(
        '--max-results',
        type=int,
        help='Override maximum results per search query'
    )
    discovery_parser.set_defaults(func=discover_linkedin)
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Run health checks')
    health_parser.set_defaults(func=health_check)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set config path if provided
    if args.config:
        import os
        os.environ['YC_PARSER_CONFIG'] = args.config
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())