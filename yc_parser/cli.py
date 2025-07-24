#!/usr/bin/env python3
"""
Command Line Interface for YC S25 Company Parser

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
    
    logger.info("Starting CLI company processing")
    
    try:
        processor = CompanyProcessor()
        
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
    
    logger.info("Starting CLI data validation")
    
    try:
        processor = CompanyProcessor()
        
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
    
    try:
        data_manager = DataManager()
        processor = CompanyProcessor()
        
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
        
        # YC S25 mentions
        yc_mentions = len(df[df['yc_s25_on_linkedin'] == True])
        mention_percentage = (yc_mentions / len(df) * 100) if len(df) > 0 else 0
        print(f"YC S25 LinkedIn mentions: {yc_mentions} ({mention_percentage:.1f}%)")
        
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
    
    try:
        processor = CompanyProcessor()
        
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
    
    try:
        processor = CompanyProcessor()
        
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
        '--force', '-f',
        action='store_true',
        help='Force processing without prompting for resume'
    )
    process_parser.set_defaults(func=process_companies)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data integrity')
    validate_parser.add_argument(
        '--repair', '-r',
        action='store_true',
        help='Attempt to repair data issues automatically'
    )
    validate_parser.set_defaults(func=validate_data)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.set_defaults(func=show_stats)
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create data backup')
    backup_parser.add_argument(
        '--name', '-n',
        type=str,
        help='Backup name (default: auto-generated)'
    )
    backup_parser.set_defaults(func=backup_data)
    
    # Clear resume command
    clear_parser = subparsers.add_parser('clear-resume', help='Clear resume data')
    clear_parser.set_defaults(func=clear_resume)
    
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